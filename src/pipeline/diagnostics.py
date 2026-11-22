"""Diagnostics only: volume errors, Stooq cross-check, USO split, GARCH variants.

Report-only: nothing here changes data/raw, data/clean or results/volume_corrections.csv
(the team edits that file by hand). Run: python -m pipeline.diagnostics
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

from pipeline.config import PROJECT_ROOT, clean_dir, load_config, raw_dir, results_dir
from pipeline.examples import load_feature_dates

VOLUME_WINDOW = 63          # median of the previous 63 trading days
LOW_RATIO, HIGH_RATIO = 0.05, 20.0
USO_SPLIT_DATE = "2020-04-28"
SPLIT_WINDOW = 20
STOOQ_FILE = PROJECT_ROOT / "data" / "external" / "stooq_2026-11-22" / "iwm_us_d.csv"


# ---------- 1. volume errors ----------

def volume_flags(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Days whose volume is far from the median of the previous 63 days (past data only)."""
    v = raw[("Volume", ticker)]
    median = v.shift(1).rolling(VOLUME_WINDOW).median()
    ratio = v / median
    hit = (ratio < LOW_RATIO) | (ratio > HIGH_RATIO)
    o, h, l, c = (raw[(f, ticker)] for f in ["Open", "High", "Low", "Close"])
    out = pd.DataFrame({"fund": ticker, "volume": v, "median_prev63": median, "ratio": ratio,
                        "open": o, "high": h, "low": l, "close": c, "prev_close": c.shift(1),
                        "simple_return_pct": raw[("Adj Close", ticker)].pct_change() * 100})
    out = out[hit].copy()
    out["direction"] = np.where(out["ratio"] < LOW_RATIO, "low", "high")
    out["ohlc_all_equal"] = (out[["open", "high", "low", "close"]].nunique(axis=1) == 1)
    out["close_equals_prev"] = out["close"] == out["prev_close"]
    out["stale_row"] = out["ohlc_all_equal"] | out["close_equals_prev"]
    return out.rename_axis("date").reset_index()


def volume_check(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """All flagged fund-days for all funds."""
    cols = ["fund", "date", "direction", "volume", "median_prev63", "ratio", "open", "high", "low", "close",
            "prev_close", "simple_return_pct", "ohlc_all_equal", "close_equals_prev", "stale_row"]
    return pd.concat([volume_flags(raw, t) for t in tickers], ignore_index=True)[cols]


# ---------- 2. Stooq cross-check ----------

def stooq_compare(flags: pd.DataFrame, path: Path) -> pd.DataFrame | None:
    """Second source for flagged IWM days: is the low volume in Yahoo only?"""
    if not path.exists():
        return None
    s = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    iwm = flags[flags["fund"] == "IWM"].set_index("date")
    out = pd.DataFrame({"yahoo_volume": iwm["volume"], "stooq_volume": s["Volume"].reindex(iwm.index),
                        "yahoo_close": iwm["close"], "stooq_close": s["Close"].reindex(iwm.index)})
    out["volume_ratio_yahoo_over_stooq"] = out["yahoo_volume"] / out["stooq_volume"]
    out["close_ratio_yahoo_over_stooq"] = out["yahoo_close"] / out["stooq_close"]
    return out.rename_axis("date").reset_index()


def stooq_volume_basis(raw: pd.DataFrame, path: Path, ticker: str, exclude: pd.DatetimeIndex) -> dict:
    """Stooq adjusts for dividends; check whether its volume is scaled too (dollar volume kept) or left raw."""
    s = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    days = raw.index.intersection(s.index).difference(exclude)
    yv, yc = raw[("Volume", ticker)].reindex(days), raw[("Close", ticker)].reindex(days)
    sv, sc = s["Volume"].reindex(days), s["Close"].reindex(days)
    return {"fund": ticker, "shared_days_used": len(days), "excluded_flagged_days": len(exclude),
            "median_yahoo_over_stooq_unadjusted": float((yv / (sv * sc / yc)).median()),
            "median_yahoo_over_stooq_as_is": float((yv / sv).median())}


def stooq_volume_corrections(raw: pd.DataFrame, path: Path, ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Replacement volumes for confirmed bad days, with Stooq's dividend scaling undone."""
    s = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").reindex(dates)
    yc = raw[("Close", ticker)].reindex(dates)
    new = (s["Volume"] * s["Close"] / yc).round().astype("int64")
    return pd.DataFrame({"fund": ticker, "date": dates, "field": "Volume",
                         "old_value": raw[("Volume", ticker)].reindex(dates).to_numpy(), "new_value": new.to_numpy(),
                         "source": "stooq " + STOOQ_FILE.parent.name.removeprefix("stooq_"),
                         "method": "round(stooq_volume * stooq_close / yahoo_raw_close)",
                         "reason": "volume < 5% of prior 63-day median; Stooq shows normal volume"})


# ---------- 3. USO split continuity ----------

def split_continuity(raw: pd.DataFrame, ticker: str, date: str, window: int) -> dict:
    """If volume is split-adjusted, average volume should not jump by the split ratio."""
    i = raw.index.get_loc(pd.Timestamp(date))
    before, after = raw.iloc[i - window: i], raw.iloc[i: i + window]
    vb, va = before[("Volume", ticker)].mean(), after[("Volume", ticker)].mean()
    return {"fund": ticker, "split_date": date, "window_days": window,
            "before": f"{before.index[0].date()}..{before.index[-1].date()}",
            "after": f"{after.index[0].date()}..{after.index[-1].date()}",
            "mean_volume_before": float(vb), "mean_volume_after": float(va), "after_over_before": float(va / vb),
            "close_day_before": float(raw[("Close", ticker)].iloc[i - 1]),
            "close_split_day": float(raw[("Close", ticker)].iloc[i]),
            "adj_close_return_split_day_pct": float(raw[("Adj Close", ticker)].pct_change().iloc[i] * 100)}


# ---------- 4. GARCH variants on training days ----------

GARCH_SPECS = {"garch_normal": dict(p=1, o=0, q=1, dist="normal"),
               "garch_t": dict(p=1, o=0, q=1, dist="t"),
               "gjr_normal": dict(p=1, o=1, q=1, dist="normal")}


def garch_row(r: np.ndarray, fund: str, spec: str) -> dict:
    """alpha + beta is the persistence; for GJR the 'effective' one adds gamma / 2."""
    res = arch_model(r, mean="Constant", vol="GARCH", **GARCH_SPECS[spec]).fit(disp="off")
    p = res.params
    a, b, g = p["alpha[1]"], p["beta[1]"], p.get("gamma[1]", 0.0)
    return {"fund": fund, "model": spec, "alpha": a, "beta": b, "gamma": g if spec == "gjr_normal" else np.nan,
            "alpha_plus_beta": a + b, "persistence": a + b + g / 2, "nu": p.get("nu", np.nan),
            "loglik": res.loglikelihood, "converged": res.convergence_flag == 0}


def garch_table(R: np.ndarray, tickers: list[str]) -> pd.DataFrame:
    """Every fund, every model, training days only."""
    return pd.DataFrame([garch_row(R[:, j], t, s) for j, t in enumerate(tickers) for s in GARCH_SPECS])


def main() -> None:
    cfg = load_config()
    tickers = cfg["tickers"]
    raw = pd.read_parquet(raw_dir() / "funds.parquet")     # read only

    flags = volume_check(raw, tickers)
    flags.to_csv(results_dir() / "volume_check.csv", index=False, date_format="%Y-%m-%d")
    per_fund = flags.groupby(["fund", "direction"]).size().unstack(fill_value=0).reindex(tickers, fill_value=0)
    per_year = flags.groupby([flags["date"].dt.year, "direction"]).size().unstack(fill_value=0)
    per_fund.to_csv(results_dir() / "volume_check_per_fund.csv")
    per_year.to_csv(results_dir() / "volume_check_per_year.csv", index_label="year")

    stooq = stooq_compare(flags, STOOQ_FILE)
    if stooq is not None:
        stooq.to_csv(results_dir() / "volume_check_iwm_stooq.csv", index=False, date_format="%Y-%m-%d")
        bad = pd.DatetimeIndex(flags.loc[(flags["fund"] == "IWM") & (flags["direction"] == "low"), "date"])
        basis = stooq_volume_basis(raw, STOOQ_FILE, "IWM", bad)
        (results_dir() / "stooq_volume_basis.json").write_text(json.dumps(basis, indent=2), encoding="utf-8")
        print(json.dumps(basis, indent=2))
        if abs(basis["median_yahoo_over_stooq_unadjusted"] - 1) < 0.01:
            # report only: the team copies approved rows into results/volume_corrections.csv by hand
            fixes = stooq_volume_corrections(raw, STOOQ_FILE, "IWM", bad)
            fixes.to_csv(results_dir() / "volume_corrections_proposed.csv", index=False, date_format="%Y-%m-%d")
            print("Proposed corrections (not applied):\n" + fixes.to_string())

    uso = split_continuity(raw, "USO", USO_SPLIT_DATE, SPLIT_WINDOW)
    (results_dir() / "uso_split_check.json").write_text(json.dumps(uso, indent=2), encoding="utf-8")

    R = np.load(clean_dir() / "returns_pct.npy")
    train = np.asarray(load_feature_dates() <= cfg["train_end"])
    garch = garch_table(R[train], tickers)
    garch.round(6).to_csv(results_dir() / "garch_variants_train.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"flagged fund-days: {len(flags)}\n", per_fund[per_fund.sum(axis=1) > 0], "\n", per_year)
    print(flags.drop(columns=["open", "high", "low"]).to_string())
    print(stooq.to_string() if stooq is not None else "Stooq file not found")
    print(json.dumps(uso, indent=2))
    print(garch.round(4).to_string())


if __name__ == "__main__":
    main()
