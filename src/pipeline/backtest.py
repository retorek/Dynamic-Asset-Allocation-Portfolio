"""Step 8 — monthly minimum-variance backtest on the validation period (2016-2019)."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pipeline.config import TRADING_DAYS_PER_YEAR as YEAR, clean_dir, eda_dir, forecast_dir, load_config, results_dir
from pipeline.examples import load_feature_dates
from pipeline.plotstyle import SERIES, apply_style, save

FORECAST_METHODS = ["historical", "ewma", "ledoit_wolf", "dcc_garch"]


def rebalance_positions(dates: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp, horizon: int) -> np.ndarray:
    """First trading day on/after start, then every H days while a full H-day holding period fits before end."""
    inside = np.flatnonzero((dates >= start) & (dates <= end))
    first, last = inside[0], inside[-1]
    return np.arange(first, last - horizon + 1, horizon)


def min_variance_weights(sigma: np.ndarray) -> np.ndarray:
    """Long-only, fully invested portfolio with the lowest forecast variance (no maximum weight yet)."""
    n = len(sigma)
    res = minimize(lambda w: w @ sigma @ w, np.full(n, 1 / n), jac=lambda w: 2 * sigma @ w, method="SLSQP",
                   bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-12, "maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"min-variance optimization failed: {res.message}")
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def equal_weights(n: int) -> np.ndarray:
    """Reference portfolio that ignores any forecast."""
    return np.full(n, 1 / n)


def run_backtest(weights_on_dates: np.ndarray, returns_pct: np.ndarray, rebalance_idx: np.ndarray,
                 horizon: int, trading_cost: float) -> dict:
    """Buy and hold between rebalancing days; weights drift with prices.

    Rebalancing happens at the close of day t, so the cost (trading_cost x turnover, in % of wealth)
    is charged on the first day of the new holding period. The first rebalance buys from cash (turnover 1).
    """
    held = np.zeros(returns_pct.shape[1])
    daily, days, turnovers = [], [], []
    for target, i in zip(weights_on_dates, rebalance_idx):
        turnover = float(np.abs(target - held).sum())
        w = target.copy()
        for d in range(i + 1, i + 1 + horizon):
            r = returns_pct[d]
            port = float(w @ r) - (trading_cost * turnover * 100 if d == i + 1 else 0.0)
            daily.append(port)
            days.append(d)
            w = w * (1 + r / 100)
            w = w / w.sum()
        held = w
        turnovers.append(turnover)
    return {"daily": np.array(daily), "days": np.array(days), "turnovers": np.array(turnovers),
            "weights": np.asarray(weights_on_dates)}


def summarize(daily: pd.Series, rf_daily: pd.Series, turnovers: np.ndarray, weights: np.ndarray) -> dict:
    """Volatility is the main target of a minimum-variance portfolio; the rest is context."""
    excess = daily - rf_daily
    wealth = (1 + daily / 100).cumprod()
    return {
        "ann_vol_pct": daily.std() * np.sqrt(YEAR),
        "ann_return_pct": daily.mean() * YEAR,
        "sharpe": excess.mean() / excess.std() * np.sqrt(YEAR),
        "max_drawdown_pct": float((wealth / wealth.cummax() - 1).min() * 100),
        "mean_turnover": float(turnovers[1:].mean()),      # first rebalance is the initial purchase
        "largest_weight": float(weights.max()),
        "n_days": len(daily),
    }


def plot_wealth(daily: pd.DataFrame, path) -> None:
    """Growth of 1 USD, to see when portfolios diverge."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    wealth = (1 + daily / 100).cumprod()
    for k, col in enumerate(wealth.columns):
        ax.plot(wealth.index, wealth[col], color=SERIES[k], label=col, linewidth=1.2)
    ax.set_title(f"Validation backtest: growth of 1 USD, {wealth.index.min():%Y-%m-%d} to {wealth.index.max():%Y-%m-%d}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth (USD)")
    ax.legend(ncol=3, loc="upper left")
    save(fig, path)


def main() -> None:
    cfg = load_config()
    H = cfg["horizon"]
    R = np.load(clean_dir() / "returns_pct.npy")
    dates = load_feature_dates()
    index = pd.read_csv(forecast_dir() / "index.csv")
    row_of = dict(zip(index["i"], range(len(index))))
    start = dates[dates > pd.Timestamp(cfg["train_end"])][0]
    reb = rebalance_positions(dates, start, pd.Timestamp(cfg["val_end"]), H)
    assert all(index["split"].iloc[row_of[i]] == "validation" for i in reb)
    irx = pd.read_parquet(clean_dir() / "irx.parquet").iloc[:, 0].reindex(dates)
    rf_pct = irx / YEAR                                      # annual % -> daily %, same units as returns

    weights = {m: np.stack([min_variance_weights(np.load(forecast_dir() / f"{m}.npy", mmap_mode="r")[row_of[i]])
                            for i in reb]) for m in FORECAST_METHODS}
    weights["equal_weight"] = np.tile(equal_weights(R.shape[1]), (len(reb), 1))

    daily, summary, wrows = {}, {}, []
    for name, W in weights.items():
        out = run_backtest(W, R, reb, H, cfg["trading_cost"])
        s = pd.Series(out["daily"], index=dates[out["days"]])
        daily[name] = s
        summary[name] = summarize(s, rf_pct.iloc[out["days"]].set_axis(s.index), out["turnovers"], W)
        for k, i in enumerate(reb):
            wrows.append({"portfolio": name, "date": dates[i].date(), **dict(zip(cfg["tickers"], W[k].round(6)))})
    daily = pd.DataFrame(daily)
    daily.to_csv(results_dir() / "backtest_daily.csv", index_label="date", date_format="%Y-%m-%d")
    table = pd.DataFrame(summary).T.round(4).sort_values("ann_vol_pct")
    table.to_csv(results_dir() / "backtest_summary.csv", index_label="portfolio")
    pd.DataFrame(wrows).to_csv(results_dir() / "backtest_weights.csv", index=False)
    plot_wealth(daily, eda_dir().parent / "backtest_wealth.png")

    sums = np.concatenate([W.sum(axis=1) for W in weights.values()])
    checks = {"n_rebalances": len(reb), "first_rebalance": str(dates[reb[0]].date()),
              "last_rebalance": str(dates[reb[-1]].date()), "n_daily_returns": len(daily),
              "weights_sum_max_error": float(np.abs(sums - 1).max()),
              "min_weight": float(min(W.min() for W in weights.values()))}
    (results_dir() / "backtest_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(table.to_string())


if __name__ == "__main__":
    main()
