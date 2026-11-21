"""Step 7 — traditional covariance forecasts for next month, each from data dated <= t only.

Units everywhere: simple daily returns in %, so covariances are daily variances in %^2.
"""
import json

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from pipeline.config import clean_dir, forecast_dir, load_config, results_dir
from pipeline.examples import load_feature_dates
from pipeline.metrics import gaussian_nll, labels

JITTER = 1e-6  # added to the diagonal only if a historical forecast is not positive definite


# ---------- simple baselines ----------

def historical_cov(R: np.ndarray, i: int, L: int) -> np.ndarray:
    """Assume next month looks like the last L days."""
    return np.cov(R[i - L + 1: i + 1], rowvar=False)


def ewma_covs(R: np.ndarray, lam: float, init_days: int) -> np.ndarray:
    """RiskMetrics: recent days weigh more. Returns one matrix per day (NaN before init_days - 1)."""
    T, N = R.shape
    out = np.full((T, N, N), np.nan)
    S = np.cov(R[:init_days], rowvar=False)
    out[init_days - 1] = S
    for t in range(init_days, T):
        S = lam * S + (1 - lam) * np.outer(R[t], R[t])
        out[t] = S
    return out


def ledoit_wolf_cov(R: np.ndarray, i: int, L: int) -> np.ndarray:
    """Sample covariance shrunk toward a simple target: less noise with only L days for 20 funds."""
    return LedoitWolf().fit(R[i - L + 1: i + 1]).covariance_


# ---------- DCC-GARCH ----------

def garch_filter(R: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Per fund GARCH(1,1) fitted on training days, then run over all days with those fixed parameters."""
    eps, sig, params = np.empty_like(R), np.empty_like(R), []
    for j in range(R.shape[1]):
        p = arch_model(R[train, j], mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off").params
        fixed = arch_model(R[:, j], mean="Constant", vol="GARCH", p=1, q=1).fix(p.to_numpy())
        eps[:, j], sig[:, j] = fixed.resid, fixed.conditional_volatility
        params.append(p.to_numpy())
    return eps, sig, pd.DataFrame(params, columns=["mu", "omega", "alpha", "beta"])


def monthly_variance(eps: np.ndarray, sig: np.ndarray, params: pd.DataFrame, horizon: int) -> np.ndarray:
    """Average GARCH variance over days t+1..t+H, mean-reverting toward the long-run level.

    sigma^2_{t+h} = LR + p^(h-1) (sigma^2_{t+1} - LR), LR = omega / (1 - p), p = alpha + beta, is written
    in the identical form omega * (1 + p + ... + p^(h-2)) + p^(h-1) sigma^2_{t+1}, which stays finite when
    a fit lands on p = 1 (HYG does).
    """
    w, a, b = (params[c].to_numpy() for c in ["omega", "alpha", "beta"])
    next_var = w + a * eps ** 2 + b * sig ** 2              # sigma^2_{t+1}, known at t
    powers = (a + b)[None, :] ** np.arange(horizon)[:, None]  # p^(h-1), h = 1..H
    geometric = np.cumsum(powers, axis=0) - powers            # 1 + p + ... + p^(h-2)
    return w * geometric.mean(axis=0) + powers.mean(axis=0) * next_var


def _normalize(Q: np.ndarray) -> np.ndarray:
    """Turn Q into a correlation matrix (unit diagonal)."""
    d = 1 / np.sqrt(np.diag(Q))
    return Q * np.outer(d, d)


def dcc_nll(ab: np.ndarray, z: np.ndarray, Qbar: np.ndarray) -> float:
    """Correlation part of the DCC likelihood: 0.5 * sum(ln det R_t + z_t' R_t^-1 z_t)."""
    a, b = ab
    Q, total = Qbar.copy(), 0.0
    for t in range(1, len(z)):
        Q = (1 - a - b) * Qbar + a * np.outer(z[t - 1], z[t - 1]) + b * Q
        Rt = _normalize(Q)
        sign, logdet = np.linalg.slogdet(Rt)
        if sign <= 0:
            return 1e12
        total += logdet + z[t] @ np.linalg.solve(Rt, z[t])
    return 0.5 * total


def fit_dcc(z_train: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Estimate a, b >= 0 with a + b < 1 on training days."""
    Qbar = np.corrcoef(z_train, rowvar=False)
    res = minimize(dcc_nll, x0=np.array([0.02, 0.95]), args=(z_train, Qbar), method="SLSQP",
                   bounds=[(0, 1), (0, 1)],
                   constraints=[{"type": "ineq", "fun": lambda x: 1 - 1e-6 - x[0] - x[1]}])
    return float(res.x[0]), float(res.x[1]), Qbar


def dcc_next_correlations(z: np.ndarray, Qbar: np.ndarray, a: float, b: float) -> np.ndarray:
    """R_{t+1} for every day t, built from z up to t only."""
    T, N = z.shape
    out, Q = np.empty((T, N, N)), Qbar.copy()
    for t in range(T):
        Q = (1 - a - b) * Qbar + a * np.outer(z[t], z[t]) + b * Q   # this is Q_{t+1}
        out[t] = _normalize(Q)
    return out


def dcc_garch_forecasts(R: np.ndarray, dates: pd.DatetimeIndex, train_end: str, horizon: int) -> tuple[np.ndarray, dict]:
    """Sigma_t = D R_{t+1} D with D from the H-day-average GARCH variances.

    Simplification: the correlation forecast for every day of the month is the one-step
    R_{t+1}; we do not iterate the DCC correlation forward over the horizon. Parameters are
    estimated on training days, so training-period forecasts are in-sample; validation ones are not.
    """
    train = np.asarray(dates <= train_end)
    eps, sig, params = garch_filter(R, train)
    z = eps / sig
    a, b, Qbar = fit_dcc(z[train])
    corr_next = dcc_next_correlations(z, Qbar, a, b)
    D = np.sqrt(monthly_variance(eps, sig, params, horizon))
    sigmas = corr_next * D[:, :, None] * D[:, None, :]
    sigmas = (sigmas + sigmas.transpose(0, 2, 1)) / 2   # remove float round-off asymmetry
    info = {"dcc_a": a, "dcc_b": b, "a_plus_b": a + b, "garch": params.round(6).to_dict(orient="list")}
    return sigmas, info


# ---------- checks and main ----------

def ensure_pd(S: np.ndarray, name: str, log: list) -> np.ndarray:
    """Covariances must be positive definite for the NLL; repair with a tiny ridge and record it."""
    for k in range(len(S)):
        if np.linalg.eigvalsh(S[k]).min() <= 0:
            S[k] = S[k] + JITTER * np.eye(S.shape[1])
            log.append(f"{name}: example {k} not PD, added {JITTER}*I")
    return S


def forecast_checks(F: np.ndarray, R: np.ndarray, idx: np.ndarray, is_train: np.ndarray, horizon: int) -> dict:
    """Symmetry, positive definiteness, and a units check (predicted vs realized variance on training)."""
    eig_min = np.linalg.eigvalsh(F).min(axis=1)
    realized = np.mean([np.mean(labels(R, i, horizon) ** 2) for i in idx[is_train]])
    predicted = np.mean(np.diagonal(F[is_train], axis1=1, axis2=2))
    return {"max_asymmetry": float(np.abs(F - F.transpose(0, 2, 1)).max()), "min_eigenvalue": float(eig_min.min()),
            "train_mean_predicted_var": round(float(predicted), 4), "train_mean_realized_var": round(float(realized), 4),
            "ratio_pred_over_realized": round(float(predicted / realized), 3)}


def main() -> None:
    cfg = load_config()
    L, H = cfg["lookback"], cfg["horizon"]
    R = np.load(clean_dir() / "returns_pct.npy")
    dates = load_feature_dates()
    table = pd.read_csv(clean_dir() / "examples.csv")
    use = table[table["split"].isin(["train", "validation"])].reset_index(drop=True)
    idx, is_train = use["i"].to_numpy(), (use["split"] == "train").to_numpy()
    use[["i", "date_t", "split"]].to_csv(forecast_dir() / "index.csv", index=False)
    log: list[str] = []

    forecasts = {"historical": ensure_pd(np.stack([historical_cov(R, i, L) for i in idx]), "historical", log)}
    for lam in cfg["ewma_baseline_lambdas"]:
        forecasts[f"ewma_{lam}"] = ewma_covs(R, lam, L)[idx]
    forecasts["ledoit_wolf"] = np.stack([ledoit_wolf_cov(R, i, L) for i in idx])
    dcc_all, dcc_info = dcc_garch_forecasts(R, dates, cfg["train_end"], H)
    forecasts["dcc_garch"] = dcc_all[idx]

    train_nll = {lam: float(np.mean([gaussian_nll(forecasts[f"ewma_{lam}"][k], labels(R, i, H))
                                     for k, i in enumerate(idx) if is_train[k]]))
                 for lam in cfg["ewma_baseline_lambdas"]}
    best_lam = min(train_nll, key=train_nll.get)
    forecasts["ewma"] = forecasts[f"ewma_{best_lam}"]

    for name, F in forecasts.items():
        np.save(forecast_dir() / f"{name}.npy", F)
    checks = {name: forecast_checks(F, R, idx, is_train, H) for name, F in forecasts.items()}
    summary = {"ewma_train_mean_nll": train_nll, "ewma_chosen_lambda": best_lam, "dcc": dcc_info,
               "pd_repairs": log, "checks": checks}
    (results_dir() / "baseline_checks.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "dcc"} |
                     {"dcc_a_b": [dcc_info["dcc_a"], dcc_info["dcc_b"], dcc_info["a_plus_b"]]}, indent=2))


if __name__ == "__main__":
    main()
