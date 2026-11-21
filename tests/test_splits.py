"""The scaler only sees training days, and no training label crosses train_end."""
import numpy as np
import pandas as pd

from pipeline.examples import example_table
from pipeline.splits import apply_scaler, assign_splits, fit_scaler

CFG = {"train_end": "2015-12-31", "val_end": "2019-12-31", "log_features": ["vol"], "log_floor": 1e-8}
NAMES = ["ret", "vol"]


def _data(seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-06-01", "2020-06-30")
    feats = np.stack([rng.normal(0, 1, (len(dates), 3)), rng.uniform(0.01, 0.05, (len(dates), 3))], axis=-1)
    return feats, dates


def test_validation_data_does_not_change_scaler():
    feats, dates = _data()
    s1 = fit_scaler(feats, dates, NAMES, CFG)
    changed = feats.copy()
    changed[dates > CFG["train_end"]] *= 50
    s2 = fit_scaler(changed, dates, NAMES, CFG)
    assert s1 == s2
    z = apply_scaler(feats, NAMES, s1)[dates <= CFG["train_end"]].reshape(-1, 2)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-10)


def test_split_boundaries():
    _, dates = _data()
    table = example_table(dates, lookback=10, horizon=5)
    split = assign_splits(table, CFG["train_end"], CFG["val_end"])
    assert (table.loc[split == "train", "label_end"] <= CFG["train_end"]).all()
    val = table[split == "validation"]
    assert (val["date_t"] > CFG["train_end"]).all() and (val["label_end"] <= CFG["val_end"]).all()
    assert (table.loc[split == "test", "date_t"] > CFG["val_end"]).all()
    assert (split == "gap").sum() == 2 * 5   # H examples at each of the two boundaries
