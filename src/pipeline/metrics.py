"""Step 7 — score covariance forecasts (Gaussian NLL, Frobenius gap) and compare methods (Diebold-Mariano)."""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pipeline.config import clean_dir, forecast_dir, load_config, results_dir

METHODS = ["historical", "ewma", "ledoit_wolf", "dcc_garch"]
LOSSES = ["nll", "frobenius"]


def gaussian_nll(sigma: np.ndarray, Y: np.ndarray) -> float:
    """How unlikely next month's daily returns are under the forecast; rewards right size AND right shape."""
    n = sigma.shape[0]
    sign, logdet = np.linalg.slogdet(sigma)
    if sign != 1:
        raise ValueError("forecast covariance is not positive definite")
    quad = np.einsum("hn,nh->h", Y, np.linalg.solve(sigma, Y.T))
    return float(np.mean(0.5 * (logdet + quad + n * np.log(2 * np.pi))))


def frobenius_gap(sigma: np.ndarray, Y: np.ndarray) -> float:
    """Distance to the realized average outer product, a simple scale-sensitive error."""
    realized = Y.T @ Y / len(Y)
    return float(np.linalg.norm(sigma - realized, "fro"))


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, max_lags: int) -> tuple[float, float]:
    """Is the mean loss difference real? HAC errors because monthly labels overlap day to day."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    res = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    return float(res.tvalues[0]), float(res.pvalues[0])


def labels(R: np.ndarray, i: int, horizon: int) -> np.ndarray:
    """Next H days of simple % returns after day i."""
    return R[i + 1: i + 1 + horizon]


def example_losses(forecasts: dict[str, np.ndarray], R: np.ndarray, idx: np.ndarray, horizon: int) -> pd.DataFrame:
    """One row per example, one column per (method, loss)."""
    rows = {}
    for m, F in forecasts.items():
        rows[f"{m}_nll"] = [gaussian_nll(F[k], labels(R, i, horizon)) for k, i in enumerate(idx)]
        rows[f"{m}_frobenius"] = [frobenius_gap(F[k], labels(R, i, horizon)) for k, i in enumerate(idx)]
    return pd.DataFrame(rows)


def dm_table(losses: pd.DataFrame, max_lags: int) -> pd.DataFrame:
    """Each method against the best one (lowest mean loss), for each loss."""
    rows = []
    for loss in LOSSES:
        means = {m: losses[f"{m}_{loss}"].mean() for m in METHODS}
        best = min(means, key=means.get)
        for m in METHODS:
            if m == best:
                continue
            t, p = dm_test(losses[f"{m}_{loss}"], losses[f"{best}_{loss}"], max_lags)
            rows.append({"loss": loss, "method": m, "best_method": best,
                         "mean_diff": means[m] - means[best], "t_stat": t, "p_value": p})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    R = np.load(clean_dir() / "returns_pct.npy")
    index = pd.read_csv(forecast_dir() / "index.csv")
    forecasts = {m: np.load(forecast_dir() / f"{m}.npy") for m in METHODS}
    losses = example_losses(forecasts, R, index["i"].to_numpy(), cfg["horizon"])
    losses.insert(0, "split", index["split"])
    losses.insert(0, "date_t", index["date_t"])
    losses.to_csv(forecast_dir() / "losses.csv", index=False)

    val = losses[losses["split"] == "validation"].reset_index(drop=True)
    scores = pd.DataFrame({
        "method": METHODS,
        "mean_nll": [val[f"{m}_nll"].mean() for m in METHODS],
        "mean_frobenius": [val[f"{m}_frobenius"].mean() for m in METHODS],
        "n_examples": len(val),
    }).sort_values("mean_nll")
    scores.to_csv(results_dir() / "scores.csv", index=False)
    dm = dm_table(val, cfg["dm_max_lags"])
    dm.to_csv(results_dir() / "dm_tests.csv", index=False)
    print("Validation scores (daily returns in %, variance in %^2):")
    print(scores.to_string(index=False))
    print("\nDiebold-Mariano vs best method (positive t = worse than best):")
    print(dm.to_string(index=False))


if __name__ == "__main__":
    main()
