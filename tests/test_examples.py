"""Inputs end at day t and labels start at t+1: no shared day, labels are simple % returns."""
import numpy as np
import pandas as pd

from pipeline.examples import example_table, get_example
from pipeline.features import returns

L, H, N, F = 5, 3, 2, 4


def test_window_boundaries():
    days = 20
    dates = pd.bdate_range("2020-01-01", periods=days)
    features = np.broadcast_to(np.arange(days)[:, None, None], (days, N, F)).astype(float)  # value = day index
    rets = np.broadcast_to(np.arange(days)[:, None], (days, N)).astype(float)
    table = example_table(dates, L, H)
    assert table["i"].iloc[0] == L - 1 and table["i"].iloc[-1] == days - 1 - H
    for i in table["i"]:
        x, y = get_example(i, features, rets, L, H)
        assert x.shape == (N, L, F) and y.shape == (H, N)
        assert x[0, -1, 0] == i and x[0, 0, 0] == i - L + 1      # last input day is t
        assert y[0, 0] == i + 1 and y[-1, 0] == i + H            # first label day is t+1
        assert set(x[0, :, 0]).isdisjoint(set(y[:, 0]))
    row = table.iloc[0]
    assert row["label_start"] > row["date_t"]


def test_labels_are_simple_percent_returns():
    p = pd.DataFrame({"A": [100.0, 101.0, 99.0, 102.0, 100.0, 103.0, 104.0],
                      "B": [50.0, 51.0, 52.0, 50.0, 49.0, 50.0, 51.0]})
    _, pct = returns(p)
    R = pct.to_numpy()
    _, y = get_example(L - 1, np.zeros((7, 2, 1)), R, L, 2)
    expected = (p.iloc[L:L + 2].to_numpy() / p.iloc[L - 1:L + 1].to_numpy() - 1) * 100
    np.testing.assert_allclose(y, expected)
