"""Scores match textbook definitions; EWMA recursion and DCC pieces behave as specified."""
import numpy as np
from scipy.stats import multivariate_normal

from pipeline.baselines import _normalize, ewma_covs, historical_cov
from pipeline.metrics import dm_test, frobenius_gap, gaussian_nll


def test_gaussian_nll_matches_scipy():
    rng = np.random.default_rng(42)
    A = rng.normal(size=(4, 4))
    sigma = A @ A.T + 4 * np.eye(4)
    Y = rng.normal(size=(21, 4))
    expected = -np.mean(multivariate_normal(mean=np.zeros(4), cov=sigma).logpdf(Y))
    assert np.isclose(gaussian_nll(sigma, Y), expected)


def test_frobenius_zero_when_exact():
    Y = np.array([[1.0, 2.0], [3.0, -1.0]])
    assert np.isclose(frobenius_gap(Y.T @ Y / 2, Y), 0.0)


def test_ewma_recursion_and_historical():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(30, 3))
    E = ewma_covs(R, 0.9, 10)
    assert np.allclose(E[9], np.cov(R[:10], rowvar=False))
    assert np.allclose(E[10], 0.9 * E[9] + 0.1 * np.outer(R[10], R[10]))
    assert np.allclose(historical_cov(R, 9, 10), E[9])


def test_normalize_unit_diagonal_and_dm_sign():
    Q = np.array([[4.0, 1.0], [1.0, 9.0]])
    C = _normalize(Q)
    assert np.allclose(np.diag(C), 1) and np.isclose(C[0, 1], 1 / 6)
    rng = np.random.default_rng(1)
    worse, better = rng.normal(1.0, 0.1, 200), rng.normal(0.0, 0.1, 200)
    t, p = dm_test(worse, better, 5)
    assert t > 0 and p < 0.01


def test_monthly_variance_matches_long_run_formula_and_handles_p_equal_1():
    import pandas as pd
    from pipeline.baselines import monthly_variance
    H = 21
    params = pd.DataFrame({"mu": [0.0, 0.0], "omega": [0.02, 0.01], "alpha": [0.1, 0.1], "beta": [0.85, 0.9]})
    eps, sig = np.array([[1.5, 0.5]]), np.array([[1.2, 0.8]])
    got = monthly_variance(eps, sig, params, H)
    w, a, b = 0.02, 0.1, 0.85
    nxt, lr = w + a * 1.5 ** 2 + b * 1.2 ** 2, w / (1 - a - b)
    expected = np.mean([lr + (a + b) ** (h - 1) * (nxt - lr) for h in range(1, H + 1)])
    assert np.isclose(got[0, 0], expected)
    nxt1 = 0.01 + 0.1 * 0.25 + 0.9 * 0.64      # p = 1: variance grows by omega each day
    assert np.isclose(got[0, 1], np.mean([nxt1 + 0.01 * (h - 1) for h in range(1, H + 1)]))
