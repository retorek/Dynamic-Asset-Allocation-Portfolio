"""Config and environment sanity: guards against a broken settings file before any step runs."""
from pipeline.config import load_config, raw_dir, clean_dir, forecast_dir, results_dir, eda_dir
from pipeline.environment import LIBRARIES, library_versions


def test_config_shapes():
    cfg = load_config()
    assert len(cfg["tickers"]) == len(set(cfg["tickers"])) == 20
    assert len(cfg["features"]) == 11
    assert set(cfg["log_features"]) <= set(cfg["features"])
    assert cfg["train_end"] < cfg["val_end"]
    assert cfg["missing_days_rule"] in (None, "ffill2", "drop")


def test_path_helpers_create_folders():
    for helper in (raw_dir, clean_dir, forecast_dir, results_dir, eda_dir):
        assert helper().is_dir()


def test_all_libraries_import():
    assert set(library_versions()) == set(LIBRARIES)
