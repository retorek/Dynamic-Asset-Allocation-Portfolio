"""Step 5 — split examples by date and fit the feature scaler on training days only."""
import json

import numpy as np
import pandas as pd

from pipeline.config import clean_dir, load_config, results_dir
from pipeline.examples import load_feature_dates


def assign_splits(table: pd.DataFrame, train_end: str, val_end: str) -> pd.Series:
    """Labels may not cross a split boundary; examples whose label does are left unused ('gap')."""
    t, label_end = table["date_t"], table["label_end"]
    split = pd.Series("gap", index=table.index, name="split")
    split[label_end <= train_end] = "train"
    split[(t > train_end) & (label_end <= val_end)] = "validation"
    split[t > val_end] = "test"
    return split


def _log_step(features: np.ndarray, names: list[str], log_features: list[str], floor: float) -> np.ndarray:
    """Volatility-like features are right-skewed; their log is closer to normal."""
    out = features.astype(np.float64).copy()
    for name in log_features:
        j = names.index(name)
        out[..., j] = np.log(np.maximum(out[..., j], floor))
    return out


def fit_scaler(features: np.ndarray, dates: pd.DatetimeIndex, names: list[str], cfg: dict) -> dict:
    """One mean/std per feature, pooled over funds, from training days only, so no later data leaks in."""
    logged = _log_step(features, names, cfg["log_features"], cfg["log_floor"])
    train = logged[dates <= cfg["train_end"]].reshape(-1, len(names))
    return {"names": names, "log_features": cfg["log_features"], "log_floor": cfg["log_floor"],
            "fit_end": cfg["train_end"], "n_days": int((dates <= cfg["train_end"]).sum()),
            "mean": train.mean(axis=0).tolist(), "std": train.std(axis=0).tolist()}


def apply_scaler(features: np.ndarray, names: list[str], scaler: dict) -> np.ndarray:
    """Same log step, then z-score with the training numbers."""
    logged = _log_step(features, names, scaler["log_features"], scaler["log_floor"])
    return (logged - np.array(scaler["mean"])) / np.array(scaler["std"])


def main() -> None:
    cfg = load_config()
    features = np.load(clean_dir() / "features.npy")
    dates = load_feature_dates()
    names = json.loads((clean_dir() / "feature_names.json").read_text(encoding="utf-8"))
    table = pd.read_csv(clean_dir() / "examples.csv", parse_dates=["date_t", "window_start", "label_start", "label_end"])
    table["split"] = assign_splits(table, cfg["train_end"], cfg["val_end"])
    table.to_csv(clean_dir() / "examples.csv", index=False, date_format="%Y-%m-%d")
    scaler = fit_scaler(features, dates, names, cfg)
    scaled = apply_scaler(features, names, scaler)
    np.save(clean_dir() / "features_scaled.npy", scaled)
    (results_dir() / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")

    train_days = scaled[dates <= cfg["train_end"]].reshape(-1, len(names))
    seen = scaled[dates <= cfg["val_end"]].reshape(-1, len(names))  # test period stays sealed
    checks = {
        "split_counts": table["split"].value_counts().to_dict(),
        "train_mean": dict(zip(names, np.round(train_days.mean(axis=0), 6).tolist())),
        "train_std": dict(zip(names, np.round(train_days.std(axis=0), 6).tolist())),
        "max_abs_z_through_val_end": dict(zip(names, np.round(np.abs(seen).max(axis=0), 2).tolist())),
        "last_train_label_end": str(table.loc[table["split"] == "train", "label_end"].max().date()),
    }
    (results_dir() / "split_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
