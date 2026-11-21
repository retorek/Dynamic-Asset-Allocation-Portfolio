"""Step 4 — table of eligible example days; windows are cut on demand instead of stored."""
import json

import numpy as np
import pandas as pd

from pipeline.config import clean_dir, load_config, results_dir


def example_table(dates: pd.DatetimeIndex, lookback: int, horizon: int) -> pd.DataFrame:
    """Day t is usable only with a full look-back before it and a full label month after it."""
    i = np.arange(lookback - 1, len(dates) - horizon)
    return pd.DataFrame({
        "i": i,
        "date_t": dates[i],
        "window_start": dates[i - lookback + 1],
        "label_start": dates[i + 1],
        "label_end": dates[i + horizon],
    })


def get_example(i: int, features: np.ndarray, returns_pct: np.ndarray,
                lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """X = the L days ending at t, per fund (N, L, F); Y = simple % returns of days t+1 .. t+H (H, N)."""
    x = features[i - lookback + 1: i + 1].transpose(1, 0, 2)
    y = returns_pct[i + 1: i + 1 + horizon]
    return x, y


def aligned_returns(returns_pct: pd.DataFrame, feature_dates: pd.DatetimeIndex, tickers: list[str]) -> np.ndarray:
    """Put labels on exactly the feature dates and fund order, so index i means the same day in both arrays."""
    r = returns_pct.reindex(index=feature_dates, columns=tickers)
    if r.isna().any().any():
        raise ValueError("returns missing on some feature dates")
    return r.to_numpy(dtype=np.float64)


def load_feature_dates() -> pd.DatetimeIndex:
    """Dates of the rows of features.npy."""
    return pd.DatetimeIndex(pd.read_csv(clean_dir() / "feature_dates.csv")["date"])


def main() -> None:
    cfg = load_config()
    L, H = cfg["lookback"], cfg["horizon"]
    features = np.load(clean_dir() / "features.npy")
    dates = load_feature_dates()
    R = aligned_returns(pd.read_parquet(clean_dir() / "ret_simple_pct.parquet"), dates, cfg["tickers"])
    np.save(clean_dir() / "returns_pct.npy", R)
    table = example_table(dates, L, H)
    table.to_csv(clean_dir() / "examples.csv", index=False, date_format="%Y-%m-%d")
    x, y = get_example(int(table["i"].iloc[0]), features, R, L, H)
    checks = {"n_examples": len(table), "first_t": str(table["date_t"].iloc[0].date()),
              "last_t": str(table["date_t"].iloc[-1].date()), "X_shape": list(x.shape), "Y_shape": list(y.shape)}
    (results_dir() / "example_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
