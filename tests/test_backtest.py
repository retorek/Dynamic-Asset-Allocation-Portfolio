"""Min-variance weights for a diagonal covariance, and buy-and-hold drift in the backtest."""
import numpy as np
import pandas as pd

from pipeline.backtest import min_variance_weights, rebalance_positions, run_backtest


def test_diagonal_min_variance_is_inverse_variance():
    var = np.array([1.0, 2.0, 4.0, 0.5])
    w = min_variance_weights(np.diag(var))
    expected = (1 / var) / (1 / var).sum()
    np.testing.assert_allclose(w, expected, atol=1e-6)
    assert np.isclose(w.sum(), 1) and (w >= 0).all()


def test_backtest_drift_and_turnover():
    R = np.array([[0, 0], [10.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # % returns
    out = run_backtest(np.array([[0.5, 0.5], [0.5, 0.5]]), R, np.array([0, 2]), horizon=2, trading_cost=0.0)
    assert np.isclose(out["daily"][0], 5.0)                     # 0.5 * 10%
    drifted = np.array([0.55, 0.5]) / 1.05
    assert np.isclose(out["turnovers"][1], np.abs(np.array([0.5, 0.5]) - drifted).sum())
    assert out["turnovers"][0] == 1.0                           # initial purchase from cash


def test_rebalance_positions_leave_full_holding_period():
    dates = pd.bdate_range("2016-01-01", periods=50)
    pos = rebalance_positions(dates, dates[0], dates[-1], 10)
    assert pos[0] == 0 and (np.diff(pos) == 10).all() and pos[-1] + 10 <= 49
