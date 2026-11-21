"""Load config.yaml and build project paths, so every step reads settings from one place."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "config.yaml") -> dict:
    """Read the team's settings file; relative paths resolve from the project root so steps run from anywhere."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure(path: Path) -> Path:
    """Create a folder on first use so no step fails on a fresh checkout."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir() -> Path:
    """Frozen download snapshot; never modified once written."""
    return _ensure(PROJECT_ROOT / "data" / "raw")


def clean_dir() -> Path:
    """Cleaned prices, features and arrays (large, git-ignored)."""
    return _ensure(PROJECT_ROOT / "data" / "clean")


def forecast_dir() -> Path:
    """Covariance forecasts per method (large, git-ignored)."""
    return _ensure(PROJECT_ROOT / "data" / "forecasts")


def results_dir() -> Path:
    """Small tracked tables and figures used in the reports."""
    return _ensure(PROJECT_ROOT / "results")


def eda_dir() -> Path:
    """Exploratory-analysis tables and plots."""
    return _ensure(results_dir() / "eda")
