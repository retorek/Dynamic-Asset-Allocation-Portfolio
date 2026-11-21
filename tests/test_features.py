"""Proves the features have no look-ahead: changing the future never changes the past."""
import numpy as np
import pandas as pd

from pipeline.config import load_config
from pipeline.features import build_feature_array, feature_inputs, returns


def synthetic_market(n_days: int, tickers: list[str], seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Random-walk prices with consistent High/Low/Volume and a positive VIX."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, len(tickers))), axis=0)),
                         index=idx, columns=tickers)
    spread = pd.DataFrame(rng.uniform(0.001, 0.02, close.shape), index=idx, columns=tickers)
    volume = pd.DataFrame(rng.integers(1_000, 100_000, close.shape).astype(float), index=idx, columns=tickers)
    prices = pd.concat({"Adj Close": close, "High": close * (1 + spread), "Low": close * (1 - spread),
                        "Volume": volume}, axis=1)
    vix = pd.Series(20 * np.exp(np.cumsum(rng.normal(0, 0.05, n_days))), index=idx)
    return prices, vix


def test_no_look_ahead():
    cfg = load_config()
    tickers = ["A", "B", "C"]
    prices, vix = synthetic_market(200, tickers, cfg["seed"])
    k = 150
    arr1, dates1 = build_feature_array(feature_inputs(prices, vix, tickers), cfg)

    future = prices.index > prices.index[k]
    prices2, vix2 = prices.copy(), vix.copy()
    prices2.loc[future] = prices2.loc[future] * 1.7   # every field changes after day k
    vix2.loc[future] = vix2.loc[future] * 3.0
    arr2, dates2 = build_feature_array(feature_inputs(prices2, vix2, tickers), cfg)

    assert dates1.equals(dates2)
    past = dates1 <= prices.index[k]
    assert past.sum() > 50
    np.testing.assert_array_equal(arr1[past], arr2[past])
    assert not np.array_equal(arr1[~past], arr2[~past])   # sanity: the change did reach the future


def test_returns_units():
    p = pd.DataFrame({"A": [100.0, 110.0]})
    log_ret, simple_pct = returns(p)
    assert np.isclose(simple_pct.iloc[1, 0], 10.0)
    assert np.isclose(log_ret.iloc[1, 0], np.log(1.1))
