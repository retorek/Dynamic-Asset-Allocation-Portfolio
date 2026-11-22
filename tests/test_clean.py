"""Gap detection and the ffill2 rule on a tiny synthetic table."""
import numpy as np
import pandas as pd

from pipeline.clean import align_to_calendar, apply_missing_rule, find_gaps


def test_find_gaps_runs():
    s = pd.Series([1, np.nan, 2, np.nan, np.nan, np.nan, 3], index=pd.bdate_range("2020-01-01", periods=7))
    gaps = find_gaps(s)
    assert gaps["length"].tolist() == [1, 3]


def test_ffill2_fills_two_then_drops():
    idx = pd.bdate_range("2020-01-01", periods=6)
    a = [1.0, np.nan, np.nan, np.nan, 5.0, 6.0]  # gap of 3: two filled, third date dropped
    df = pd.DataFrame({("Adj Close", "A"): a, ("Adj Close", "B"): np.arange(6.0)}, index=idx)
    out, flags = apply_missing_rule(df, "ffill2")
    assert len(out) == 5 and idx[3] not in out.index
    assert out.loc[idx[2], ("Adj Close", "A")] == 1.0
    assert (flags["flag"] == "filled").sum() == 2 and (flags["flag"] == "date_dropped").sum() == 1


def test_align_uses_previous_value_only():
    cal = pd.bdate_range("2020-01-01", periods=4)
    s = pd.Series([10.0, 30.0], index=[cal[0], cal[3]])
    aligned, filled = align_to_calendar(s, cal)
    assert aligned.tolist() == [10.0, 10.0, 10.0, 30.0] and len(filled) == 2


def test_apply_corrections_replaces_and_guards_old_value():
    import pytest
    from pipeline.clean import apply_corrections
    idx = pd.to_datetime(["2008-03-24", "2008-03-25"])
    df = pd.DataFrame({("Volume", "IWM"): [90.0, 1.0]}, index=idx)
    fix = pd.DataFrame({"fund": ["IWM"], "date": ["2008-03-25"], "field": ["Volume"], "old_value": [1], "new_value": [70]})
    out, flags = apply_corrections(df, fix)
    assert out.at[idx[1], ("Volume", "IWM")] == 70 and df.at[idx[1], ("Volume", "IWM")] == 1
    assert flags["flag"].tolist() == ["corrected"]
    with pytest.raises(ValueError):
        apply_corrections(df, fix.assign(old_value=5))
