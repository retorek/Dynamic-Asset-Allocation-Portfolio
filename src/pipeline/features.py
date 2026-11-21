"""Step 3 — returns and the per-fund daily features; every value for day t uses data dated <= t only."""
import json
from functools import partial

import numpy as np
import pandas as pd

from pipeline.config import clean_dir, load_config, results_dir

REL_VOLUME_WINDOW = 63  # part of the rel_volume definition (task file), not the model look-back


def returns(adj_close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Log returns feed the features; simple % returns are the labels and portfolio returns."""
    return np.log(adj_close).diff(), adj_close.pct_change(fill_method=None) * 100


# ---------- one function per feature: (inputs, cfg) -> dates x tickers ----------
# inputs: log_ret, high, low, volume (dates x tickers) and vix (Series)

def f_log_ret(inp: dict, cfg: dict) -> pd.DataFrame:
    """Today's move."""
    return inp["log_ret"]


def f_abs_ret(inp: dict, cfg: dict) -> pd.DataFrame:
    """Size of today's move, regardless of sign."""
    return inp["log_ret"].abs()


def f_swing(inp: dict, cfg: dict, window: int) -> pd.DataFrame:
    """Recent realized volatility over a trailing window."""
    return inp["log_ret"].rolling(window).std()


def f_ewma_swing(inp: dict, cfg: dict) -> pd.DataFrame:
    """RiskMetrics-style volatility that reacts faster than a flat window."""
    alpha = 1 - cfg["ewma_feature_lambda"]
    return np.sqrt((inp["log_ret"] ** 2).ewm(alpha=alpha, adjust=False).mean())


def f_ret_sum(inp: dict, cfg: dict, window: int) -> pd.DataFrame:
    """Trailing momentum."""
    return inp["log_ret"].rolling(window).sum()


def f_rel_volume(inp: dict, cfg: dict) -> pd.DataFrame:
    """Unusual trading activity relative to the recent average."""
    v = inp["volume"]
    return np.log(v) - np.log(v.rolling(REL_VOLUME_WINDOW).mean())


def _broadcast(s: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    """Market-wide series copied to every fund column."""
    return pd.DataFrame(np.repeat(s.to_numpy()[:, None], like.shape[1], axis=1), index=like.index, columns=like.columns)


def f_log_vix(inp: dict, cfg: dict) -> pd.DataFrame:
    """Market fear level, the same for every fund."""
    return _broadcast(np.log(inp["vix"]), inp["log_ret"])


def f_vix_change(inp: dict, cfg: dict, window: int) -> pd.DataFrame:
    """Whether fear is rising or falling."""
    lv = np.log(inp["vix"])
    return _broadcast(lv - lv.shift(window), inp["log_ret"])


def f_hl_range(inp: dict, cfg: dict) -> pd.DataFrame:
    """Intraday range: a volatility estimate from a single day."""
    return np.log(inp["high"] / inp["low"])


FEATURE_FUNCS = {
    "log_ret": f_log_ret,
    "abs_ret": f_abs_ret,
    "swing_5": partial(f_swing, window=5),
    "swing_21": partial(f_swing, window=21),
    "swing_63": partial(f_swing, window=63),
    "ewma_swing": f_ewma_swing,
    "ret_21": partial(f_ret_sum, window=21),
    "rel_volume": f_rel_volume,
    "log_vix": f_log_vix,
    "vix_change_21": partial(f_vix_change, window=21),
    "hl_range": f_hl_range,
}


def feature_inputs(prices: pd.DataFrame, vix: pd.Series, tickers: list[str]) -> dict:
    """Split the cleaned price table into the inputs the feature functions use."""
    log_ret, _ = returns(prices["Adj Close"][tickers])
    return {"log_ret": log_ret, "high": prices["High"][tickers], "low": prices["Low"][tickers],
            "volume": prices["Volume"][tickers], "vix": vix}


def build_feature_array(inputs: dict, cfg: dict) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Stack features in config order into (days, N, F) and drop the warm-up rows."""
    frames = [FEATURE_FUNCS[name](inputs, cfg) for name in cfg["features"]]
    arr = np.stack([f.to_numpy(dtype=float) for f in frames], axis=-1)
    dates = frames[0].index
    complete = ~np.isnan(arr).any(axis=(1, 2))
    first = int(np.argmax(complete))
    return arr[first:], dates[first:]


def feature_checks(arr: np.ndarray, dates: pd.DatetimeIndex, names: list[str], volume: pd.DataFrame) -> dict:
    """Facts to confirm before using the features."""
    i_vix, i_chg = names.index("log_vix"), names.index("vix_change_21")
    return {
        "shape": list(arr.shape), "first_date": str(dates[0].date()), "last_date": str(dates[-1].date()),
        "nan_after_warmup": int(np.isnan(arr).sum()), "inf_after_warmup": int(np.isinf(arr).sum()),
        "vix_features_identical_across_funds": bool((arr[:, :, i_vix] == arr[:, :1, i_vix]).all()
                                                     and (arr[:, :, i_chg] == arr[:, :1, i_chg]).all()),
        "zero_volume_days": {t: int(n) for t, n in (volume == 0).sum().items() if n > 0},
        "high_equals_low_days": int((arr[:, :, names.index("hl_range")] == 0).sum()),
    }


def main() -> None:
    cfg = load_config()
    prices = pd.read_parquet(clean_dir() / "prices.parquet")
    vix = pd.read_parquet(clean_dir() / "vix.parquet").iloc[:, 0]
    tickers, names = cfg["tickers"], cfg["features"]
    log_ret, simple_pct = returns(prices["Adj Close"][tickers])
    log_ret.to_parquet(clean_dir() / "ret_log.parquet")
    simple_pct.to_parquet(clean_dir() / "ret_simple_pct.parquet")
    inputs = feature_inputs(prices, vix, tickers)
    arr, dates = build_feature_array(inputs, cfg)
    np.save(clean_dir() / "features.npy", arr)
    pd.Series(dates.strftime("%Y-%m-%d"), name="date").to_csv(clean_dir() / "feature_dates.csv", index=False)
    (clean_dir() / "feature_names.json").write_text(json.dumps(names), encoding="utf-8")
    checks = feature_checks(arr, dates, names, inputs["volume"])
    (results_dir() / "feature_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
