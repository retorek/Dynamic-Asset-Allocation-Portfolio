"""Step 1 — download Yahoo Finance data once and freeze it, so every later result is reproducible."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from pipeline.config import eda_dir, load_config, raw_dir, results_dir
from pipeline.plotstyle import apply_style, save

FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def download_funds(cfg: dict) -> pd.DataFrame:
    """Unadjusted OHLCV plus Adj Close for all funds; columns are (field, ticker)."""
    return yf.download(cfg["tickers"], start=cfg["start"], end=cfg["end"],
                       auto_adjust=False, progress=False)


def download_close(ticker: str, cfg: dict) -> pd.Series:
    """VIX and IRX are indices: only the close is meaningful."""
    df = yf.download(ticker, start=cfg["start"], end=cfg["end"], auto_adjust=False, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance keeps a ticker level even for one ticker
        close = close[ticker]
    return close.rename(ticker)


def sha256(path: Path) -> str:
    """Fingerprint a file so the team can verify everyone uses the same snapshot."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def series_info(s: pd.Series) -> dict:
    """Rows and first/last valid date of one series, for the manifest."""
    valid = s.dropna()
    return {"rows": int(valid.size), "first": str(valid.index.min().date()), "last": str(valid.index.max().date())}


def write_manifest(files: dict[str, Path], series: dict[str, pd.Series], cfg: dict) -> None:
    """Record when, how and what was downloaded, with checksums."""
    manifest = {
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "yfinance_version": yf.__version__,
        "tickers": cfg["tickers"], "vix_ticker": cfg["vix_ticker"], "rate_ticker": cfg["rate_ticker"],
        "start": cfg["start"], "end_exclusive": cfg["end"],
        "series": {name: series_info(s) for name, s in series.items()},
        "sha256": {name: sha256(p) for name, p in files.items()},
    }
    (raw_dir() / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def snapshot_checks(funds: pd.DataFrame, vix: pd.Series, irx: pd.Series, cfg: dict) -> dict:
    """Facts the team must confirm before trusting the snapshot."""
    tickers = cfg["tickers"]
    missing = {f: sorted(set(tickers) - set(funds[f].columns)) for f in FIELDS if f in funds.columns.levels[0]}
    first_valid = funds["Adj Close"].apply(lambda s: s.first_valid_index()).dt.date.astype(str)
    return {
        "fields_present": sorted(set(funds.columns.get_level_values(0))),
        "tickers_missing_per_field": missing,
        "first_valid_date_per_fund": first_valid.to_dict(),
        "latest_first_valid": {"ticker": first_valid.idxmax(), "date": first_valid.max()},
        "rows": int(len(funds)), "first_date": str(funds.index.min().date()),
        "last_date": str(funds.index.max().date()),
        "all_empty_columns": [f"{f}/{t}" for f, t in funds.columns[funds.isna().all()]],
        "vix": series_info(vix), "irx": series_info(irx),
    }


def plot_spy(adj_close: pd.Series, path: Path) -> None:
    """Visual sanity check: SPY should show 2008, 2020 and 2022 drawdowns and no jumps."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(adj_close.index, adj_close.values)
    ax.set_title(f"SPY adjusted close, {adj_close.index.min():%Y-%m-%d} to {adj_close.index.max():%Y-%m-%d}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted close (USD)")
    save(fig, path)


def main() -> None:
    cfg = load_config()
    funds_path = raw_dir() / "funds.parquet"
    if funds_path.exists():
        raise SystemExit(f"{funds_path} exists: the snapshot is frozen. A new download goes to a new dated folder.")
    funds = download_funds(cfg)
    vix = download_close(cfg["vix_ticker"], cfg)
    irx = download_close(cfg["rate_ticker"], cfg)
    files = {"funds": funds_path, "vix": raw_dir() / "vix.parquet", "irx": raw_dir() / "irx.parquet"}
    funds.to_parquet(files["funds"])
    vix.to_frame().to_parquet(files["vix"])
    irx.to_frame().to_parquet(files["irx"])
    series = {t: funds["Adj Close"][t] for t in cfg["tickers"]} | {"vix": vix, "irx": irx}
    write_manifest(files, series, cfg)
    checks = snapshot_checks(funds, vix, irx, cfg)
    (results_dir() / "download_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    plot_spy(funds["Adj Close"]["SPY"].dropna(), eda_dir() / "spy_adj_close.png")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
