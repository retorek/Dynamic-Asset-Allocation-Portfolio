"""Step 6 — exploratory tables and plots, using dates up to val_end only (the test period stays sealed)."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from pipeline.config import TRADING_DAYS_PER_YEAR as YEAR, clean_dir, eda_dir, load_config
from pipeline.examples import load_feature_dates
from pipeline.plotstyle import DIVERGING, INK_SECONDARY, MUTED, SERIES, apply_style, save

VOL_TICKERS = ["SPY", "TLT", "GLD", "USO"]
STOCK_FUNDS = ["SPY", "IWM", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLU", "XLP", "EFA", "EEM"]
BOND_FUNDS = ["TLT", "IEF"]
CALM = ("2017-01-01", "2017-12-31")
CRISIS = ("2008-09-01", "2009-03-31")


def _span(index: pd.Index) -> str:
    """Date range for figure titles."""
    return f"{index.min():%Y-%m-%d} to {index.max():%Y-%m-%d}"


# 1. summary statistics
def summary_stats(R: pd.DataFrame) -> pd.DataFrame:
    """How each fund behaves on average and in the tails (daily simple returns in %)."""
    return pd.DataFrame({
        "ann_return_pct": R.mean() * YEAR, "ann_vol_pct": R.std() * np.sqrt(YEAR),
        "worst_day_pct": R.min(), "worst_day": R.idxmin().dt.strftime("%Y-%m-%d"),
        "skew": R.skew(), "excess_kurtosis": R.kurt(),
    }).round(3)


# 2. rolling volatility
def rolling_vol(R: pd.DataFrame, tickers: list[str], window: int) -> pd.DataFrame:
    """Volatility changes over time: the reason a time-varying forecast is needed."""
    return (R[tickers].rolling(window).std() * np.sqrt(YEAR)).dropna()


def plot_rolling_vol(vol: pd.DataFrame, window: int, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k, t in enumerate(vol.columns):
        ax.plot(vol.index, vol[t], color=SERIES[k], label=t, linewidth=1.2)
    ax.set_title(f"{window}-day rolling volatility (annualized), {_span(vol.index)}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (% per year)")
    ax.legend(ncol=len(vol.columns), loc="upper right")
    save(fig, path)


# 3. autocorrelation
def acf_table(r: pd.Series, nlags: int) -> pd.DataFrame:
    """Squared returns stay autocorrelated (volatility clustering) while plain returns do not."""
    r = r.dropna()
    return pd.DataFrame({"lag": np.arange(1, nlags + 1),
                         "acf_returns": acf(r, nlags=nlags)[1:], "acf_squared_returns": acf(r ** 2, nlags=nlags)[1:]})


def plot_acf(table: pd.DataFrame, n_obs: int, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    band = 1.96 / np.sqrt(n_obs)
    ax.axhspan(-band, band, color=MUTED, alpha=0.15, linewidth=0, label="95% band, no autocorrelation")
    ax.plot(table["lag"], table["acf_squared_returns"], color=SERIES[0], label="Squared returns")
    ax.plot(table["lag"], table["acf_returns"], color=SERIES[1], label="Returns")
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("Lag (trading days)")
    ax.set_ylabel("Autocorrelation")
    ax.legend(loc="upper right")
    save(fig, path)


# 4. correlation tables
def corr_table(R: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Correlations in one period."""
    return R.loc[start:end].corr()


def average_correlations(C: pd.DataFrame) -> dict:
    """Diversification summary: stock-stock and stock-Treasury averages."""
    s = C.loc[STOCK_FUNDS, STOCK_FUNDS].to_numpy()
    off_diag = s[~np.eye(len(STOCK_FUNDS), dtype=bool)]
    return {"stock_vs_stock": round(float(off_diag.mean()), 3),
            "stock_vs_TLT_IEF": round(float(C.loc[STOCK_FUNDS, BOND_FUNDS].to_numpy().mean()), 3)}


def plot_corr_pair(c1: pd.DataFrame, t1: str, c2: pd.DataFrame, t2: str, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, c, title in zip(axes, [c1, c2], [t1, t2]):
        im = ax.imshow(c.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(len(c)), c.columns, rotation=90)
        ax.set_yticks(range(len(c)), c.index)
        ax.set_title(title)
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Correlation of daily returns")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# 5. rolling SPY-TLT correlation
def rolling_corr(R: pd.DataFrame, a: str, b: str, window: int) -> pd.Series:
    """Stock-bond correlation is not constant: a fixed correlation forecast is wrong."""
    return R[a].rolling(window).corr(R[b]).dropna().rename(f"{a}_{b}_corr")


def plot_rolling_corr(c: pd.Series, window: int, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(c.index, c.values, color=SERIES[0])
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.6)
    ax.set_ylim(-1, 1)
    ax.set_title(f"{window}-day rolling correlation, SPY vs TLT, {_span(c.index)}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    save(fig, path)


# 6. VIX vs next-month realized volatility
def vix_vs_future_vol(vix: pd.Series, r: pd.Series, horizon: int) -> pd.DataFrame:
    """Does today's VIX predict next month's realized volatility? (future by design: evaluation quantity)."""
    rv = r.rolling(horizon).std() * np.sqrt(YEAR)          # value at day k covers days k-H+1 .. k
    future = pd.Series(rv.to_numpy()[horizon:], index=rv.index[:-horizon])  # moved to day t = k-H: covers t+1 .. t+H
    df = pd.DataFrame({"vix": vix, "spy_realized_vol_next": future}).dropna()
    return df


def plot_vix_scatter(df: pd.DataFrame, horizon: int, path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(df["vix"], df["spy_realized_vol_next"], s=8, alpha=0.35, color=SERIES[0], linewidths=0)
    lim = [0, max(df.max()) * 1.05]
    ax.plot(lim, lim, color=MUTED, linewidth=0.8, linestyle="--", label="Realized = VIX")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    corr = df.corr().iloc[0, 1]
    ax.set_title(f"VIX on day t vs SPY realized vol, t+1..t+{horizon}\n{_span(df.index)}, correlation {corr:.2f}")
    ax.set_xlabel("VIX on day t (% per year)")
    ax.set_ylabel(f"SPY realized volatility next {horizon} days (% per year)")
    ax.legend(loc="upper left")
    save(fig, path)


# 7. scaled features
def feature_ranges(z: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Min/max of each scaled feature on training days; extreme values hint at data problems."""
    flat = z.reshape(-1, len(names))
    return pd.DataFrame({"min": flat.min(axis=0), "max": flat.max(axis=0),
                         "mean": flat.mean(axis=0), "std": flat.std(axis=0)}, index=names).round(3)


def plot_feature_hists(z: np.ndarray, names: list[str], title: str, path) -> None:
    flat = z.reshape(-1, len(names))
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.5))
    for k, ax in enumerate(axes.ravel()):
        if k >= len(names):
            ax.axis("off")
            continue
        ax.hist(flat[:, k], bins=80, color=SERIES[0])
        ax.set_title(names[k])
        ax.set_xlabel("z-score")
        ax.set_ylabel("Fund-days")
    fig.suptitle(title)
    save(fig, path)


def main() -> None:
    cfg = load_config()
    apply_style()
    out = eda_dir()
    L, H, val_end = cfg["lookback"], cfg["horizon"], cfg["val_end"]
    R = pd.read_parquet(clean_dir() / "ret_simple_pct.parquet")[cfg["tickers"]].loc[:val_end].dropna()
    vix = pd.read_parquet(clean_dir() / "vix.parquet").iloc[:, 0].loc[:val_end]
    facts = {"period": _span(R.index)}

    summary_stats(R).to_csv(out / "summary_stats.csv")

    vol = rolling_vol(R, VOL_TICKERS, H)
    vol.round(4).to_csv(out / "rolling_vol.csv")
    plot_rolling_vol(vol, H, out / "rolling_vol.png")

    a = acf_table(R["SPY"], L)
    a.round(4).to_csv(out / "acf_spy.csv", index=False)
    plot_acf(a, len(R), f"Autocorrelation of SPY daily returns, {_span(R.index)}", out / "acf_spy.png")

    c_calm, c_crisis = corr_table(R, *CALM), corr_table(R, *CRISIS)
    c_calm.round(3).to_csv(out / "corr_2017.csv")
    c_crisis.round(3).to_csv(out / "corr_2008_crisis.csv")
    facts["avg_corr_2017"] = average_correlations(c_calm)
    facts["avg_corr_crisis"] = average_correlations(c_crisis)
    plot_corr_pair(c_calm, f"Calm: {CALM[0]} to {CALM[1]}", c_crisis, f"Crisis: {CRISIS[0]} to {CRISIS[1]}",
                   out / "corr_heatmaps.png")

    rc = rolling_corr(R, "SPY", "TLT", L)
    rc.round(4).to_csv(out / "rolling_corr_spy_tlt.csv")
    plot_rolling_corr(rc, L, out / "rolling_corr_spy_tlt.png")

    vv = vix_vs_future_vol(vix, R["SPY"], H)
    vv.round(4).to_csv(out / "vix_vs_future_vol.csv")
    facts["vix_vs_future_vol_corr"] = round(float(vv.corr().iloc[0, 1]), 3)
    plot_vix_scatter(vv, H, out / "vix_vs_future_vol.png")

    names = json.loads((clean_dir() / "feature_names.json").read_text(encoding="utf-8"))
    dates = load_feature_dates()
    train = dates <= cfg["train_end"]
    z = np.load(clean_dir() / "features_scaled.npy")[train]
    feature_ranges(z, names).to_csv(out / "scaled_feature_ranges.csv")
    plot_feature_hists(z, names, f"Scaled features on training days, {_span(dates[train])} (all funds pooled)",
                       out / "scaled_feature_hists.png")

    (out / "eda_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
