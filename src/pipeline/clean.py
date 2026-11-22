"""Step 2 — report missing days (2a), then clean prices with the team's missing-days rule (2b)."""
import json

import pandas as pd

from pipeline.config import clean_dir, load_config, raw_dir, results_dir

FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
SAVED_FIELDS = ["Adj Close", "High", "Low", "Volume"]
RULES = ("ffill2", "drop")


# ---------- 2a: missing-days report ----------

def trading_calendar(adj_close: pd.DataFrame) -> pd.DatetimeIndex:
    """A day counts as a trading day if at least one fund traded; otherwise it is a holiday, not a gap."""
    return adj_close.index[adj_close.notna().any(axis=1)]


def find_gaps(series: pd.Series) -> pd.DataFrame:
    """Runs of consecutive missing values, because a 1-day gap and a 10-day gap need different treatment."""
    missing = series.isna()
    run_id = (missing != missing.shift()).cumsum()
    rows = [{"start": run.index[0], "end": run.index[-1], "length": len(run)}
            for _, run in series[missing].groupby(run_id[missing])]
    return pd.DataFrame(rows, columns=["start", "end", "length"])


def missing_days_report(adj_close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-fund counts and the full gap list, so the team can choose the missing-days rule."""
    adj_close = adj_close.reindex(trading_calendar(adj_close))
    gap_tables = []
    for ticker in adj_close.columns:
        g = find_gaps(adj_close[ticker])
        g.insert(0, "ticker", ticker)
        gap_tables.append(g)
    gaps = pd.concat(gap_tables, ignore_index=True)
    per_fund = pd.DataFrame({
        "missing_days": adj_close.isna().sum(),
        "n_gaps": gaps.groupby("ticker").size().reindex(adj_close.columns, fill_value=0),
        "longest_gap": gaps.groupby("ticker")["length"].max().reindex(adj_close.columns, fill_value=0),
    })
    per_fund.index.name = "ticker"
    return per_fund, gaps


# ---------- 2b: cleaning ----------

def apply_missing_rule(df: pd.DataFrame, rule: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the team's rule to every (field, ticker) column and record each filled cell or dropped date."""
    if rule == "ffill2":
        out = df.ffill(limit=2)
    elif rule == "drop":
        out = df.copy()
    else:
        raise ValueError(f"missing_days_rule must be one of {RULES}, got {rule!r}")
    filled = (df.isna() & out.notna()).stack(level=[0, 1], future_stack=True)
    filled = filled[filled]
    flags = [pd.DataFrame({"date": filled.index.get_level_values(0), "series": filled.index.get_level_values(2),
                           "field": filled.index.get_level_values(1), "flag": "filled"})]
    still_missing = out.isna().any(axis=1)
    for date in out.index[still_missing]:
        tickers = sorted(set(out.columns[out.loc[date].isna()].get_level_values(1)))
        flags.append(pd.DataFrame([{"date": date, "series": ";".join(tickers), "field": "all", "flag": "date_dropped"}]))
    return out.loc[~still_missing], pd.concat(flags, ignore_index=True)


def apply_corrections(df: pd.DataFrame, corrections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace values proven wrong by a second source; each must still match the raw value it replaces."""
    out = df.copy()
    for row in corrections.itertuples(index=False):
        date, col = pd.Timestamp(row.date), (row.field, row.fund)
        if out.at[date, col] != row.old_value:
            raise ValueError(f"correction {row.fund} {row.field} {row.date}: raw value {out.at[date, col]} "
                             f"!= old_value {row.old_value}")
        out.at[date, col] = row.new_value
    flags = pd.DataFrame({"date": pd.to_datetime(corrections["date"]), "series": corrections["fund"],
                          "field": corrections["field"], "flag": "corrected"})
    return out, flags


def load_corrections() -> pd.DataFrame:
    """Team-approved corrections (results/volume_corrections.csv); none if the file is absent."""
    path = results_dir() / "volume_corrections.csv"
    cols = ["fund", "date", "field", "old_value", "new_value"]
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)


def _long(mask: pd.DataFrame, values: pd.DataFrame, check: str) -> pd.DataFrame:
    """Turn a dates x tickers boolean mask into rows (date, ticker, check, value)."""
    s = values.where(mask).stack(future_stack=True).dropna()
    return pd.DataFrame({"date": s.index.get_level_values(0), "ticker": s.index.get_level_values(1),
                         "check": check, "value": s.values})


def sanity_violations(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                      close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Impossible raw values point to data errors, which would distort features such as hl_range."""
    parts = [_long(p <= 0, p, f"{name}<=0") for name, p in
             [("open", open_), ("high", high), ("low", low), ("close", close)]]
    parts += [_long(close > high, close, "close>high"), _long(close < low, close, "close<low"),
              _long(volume < 0, volume, "volume<0")]
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker", "check"], ignore_index=True)


def extreme_moves(adj_close: pd.DataFrame, close: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Huge daily moves are kept, but listed with raw and adjusted prices so the team can judge each one."""
    ret = adj_close.pct_change(fill_method=None)
    mask = ret.abs() > threshold
    out = _long(mask, ret, "extreme_move").rename(columns={"value": "simple_return"}).drop(columns="check")
    for name, frame in [("adj_prev", adj_close.shift(1)), ("adj_now", adj_close),
                        ("close_prev", close.shift(1)), ("close_now", close)]:
        out[name] = [frame.at[d, t] for d, t in zip(out["date"], out["ticker"])]
    return out.sort_values(["date", "ticker"], ignore_index=True)


def split_suspects(adj_close: pd.DataFrame, close: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """A jump in Adj Close / Close means a large distribution or a split handled differently in the two series."""
    ratio = adj_close / close
    change = ratio.pct_change(fill_method=None).abs()
    out = _long(change > threshold, change, "ratio_change").rename(columns={"value": "ratio_change"}).drop(columns="check")
    for name, frame in [("ratio_prev", ratio.shift(1)), ("ratio_now", ratio),
                        ("close_prev", close.shift(1)), ("close_now", close)]:
        out[name] = [frame.at[d, t] for d, t in zip(out["date"], out["ticker"])]
    return out.sort_values(["date", "ticker"], ignore_index=True)


def align_to_calendar(series: pd.Series, calendar: pd.DatetimeIndex) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Put VIX/IRX on the fund calendar using only the latest value known on each day (forward fill)."""
    valid = series.dropna()
    aligned = valid.reindex(valid.index.union(calendar)).ffill().reindex(calendar)
    filled_dates = calendar.difference(valid.index)
    if aligned.isna().any():
        raise ValueError(f"{series.name}: no earlier value to carry forward on {aligned.index[aligned.isna()][0].date()}")
    return aligned, filled_dates


# ---------- main ----------

def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Raw snapshot on the fund trading calendar, columns (field, ticker) in config order."""
    funds = pd.read_parquet(raw_dir() / "funds.parquet")
    cols = pd.MultiIndex.from_product([FIELDS, cfg["tickers"]])
    wide = funds[cols]
    wide = wide.reindex(trading_calendar(wide["Adj Close"]))
    vix = pd.read_parquet(raw_dir() / "vix.parquet").iloc[:, 0]
    irx = pd.read_parquet(raw_dir() / "irx.parquet").iloc[:, 0]
    return wide, vix, irx


def print_missing_summary(per_fund: pd.DataFrame, gaps: pd.DataFrame, extra: dict[str, int]) -> None:
    """Short console summary of the 2a report."""
    print("Missing days per fund (Adj Close, on the fund trading calendar):")
    print(per_fund.to_string())
    longest = gaps.loc[gaps["length"].idxmax()] if len(gaps) else None
    if longest is not None:
        print(f"Longest gap: {longest['ticker']} {longest['start'].date()} .. {longest['end'].date()} ({longest['length']} days)")
    lengths = gaps["length"]
    print(f"Gaps: total {len(gaps)}; length 1: {(lengths == 1).sum()}, length 2: {(lengths == 2).sum()}, "
          f"length > 2: {(lengths > 2).sum()}")
    for name, n in extra.items():
        print(f"{name}: {n}")


def write_clean(wide, vix, irx, flags, cfg) -> dict:
    """2b: checks and outputs once the rule is applied; returns the quality report."""
    thr_ext, thr_split = cfg["extreme_move_threshold"], cfg["split_check_threshold"]
    cal = wide.index
    vix_c, vix_filled = align_to_calendar(vix, cal)
    irx_c, irx_filled = align_to_calendar(irx, cal)
    flags = pd.concat([flags] + [pd.DataFrame({"date": d, "series": name, "field": "Close", "flag": "filled"})
                                 for name, d in [(cfg["vix_ticker"], vix_filled), (cfg["rate_ticker"], irx_filled)]],
                      ignore_index=True).sort_values(["date", "series", "field"], ignore_index=True)
    violations = sanity_violations(*(wide[f] for f in ["Open", "High", "Low", "Close", "Volume"]))
    extremes = extreme_moves(wide["Adj Close"], wide["Close"], thr_ext)
    splits = split_suspects(wide["Adj Close"], wide["Close"], thr_split)
    wide[SAVED_FIELDS].to_parquet(clean_dir() / "prices.parquet")
    vix_c.to_frame().to_parquet(clean_dir() / "vix.parquet")
    irx_c.to_frame().to_parquet(clean_dir() / "irx.parquet")
    flags.to_csv(results_dir() / "flags.csv", index=False)
    extremes.to_csv(results_dir() / "extreme_moves.csv", index=False)
    splits.to_csv(results_dir() / "split_suspects.csv", index=False)
    violations.to_csv(results_dir() / "sanity_violations.csv", index=False)
    return {
        "missing_days_rule": cfg["missing_days_rule"], "rows": len(cal),
        "first_date": str(cal[0].date()), "last_date": str(cal[-1].date()),
        "cells_filled": int((flags["flag"] == "filled").sum() - len(vix_filled) - len(irx_filled)),
        "dates_dropped": int((flags["flag"] == "date_dropped").sum()),
        "vix_days_filled": len(vix_filled), "irx_days_filled": len(irx_filled),
        "sanity_violations": violations["check"].value_counts().to_dict(),
        "extreme_moves": len(extremes), "split_suspects": len(splits),
        "missing_values_left": int(wide[SAVED_FIELDS].isna().sum().sum() + vix_c.isna().sum() + irx_c.isna().sum()),
        "same_dates": bool(wide.index.equals(vix_c.index) and wide.index.equals(irx_c.index)),
    }


def main() -> None:
    cfg = load_config()
    wide, vix, irx = load_raw(cfg)
    per_fund, gaps = missing_days_report(wide["Adj Close"])
    per_fund.to_csv(results_dir() / "missing_days_per_fund.csv")
    gaps.to_csv(results_dir() / "missing_gaps.csv", index=False)
    cal = wide.index
    other = int((wide.drop(columns="Adj Close", level=0).isna().T.groupby(level=1).any().T & wide["Adj Close"].notna()).sum().sum())
    print_missing_summary(per_fund, gaps, {
        "Trading days (calendar)": len(cal),
        "Fund-days with Adj Close but another field missing": other,
        f"{cfg['vix_ticker']} days missing on the fund calendar": len(cal.difference(vix.dropna().index)),
        f"{cfg['rate_ticker']} days missing on the fund calendar": len(cal.difference(irx.dropna().index)),
    })
    rule = cfg["missing_days_rule"]
    if rule is None:
        raise SystemExit("\nStep 2a done. missing_days_rule is not set in config.yaml: the team must choose "
                         f"one of {RULES} after reading the report. Step 2b (cleaning) not run.")
    corrections = load_corrections()
    wide, fix_flags = apply_corrections(wide, corrections)
    clean, flags = apply_missing_rule(wide, rule)
    report = write_clean(clean, vix, irx, pd.concat([flags, fix_flags], ignore_index=True), cfg)
    report["cells_corrected"] = len(corrections)
    (results_dir() / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
