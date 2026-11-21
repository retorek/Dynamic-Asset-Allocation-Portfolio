"""Step 0 — record library versions so every result can be traced to the environment that made it."""
import importlib
import json
import platform
import sys

from pipeline.config import load_config, results_dir

# pip name -> import name
LIBRARIES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "yfinance": "yfinance",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "arch": "arch",
    "statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
    "pyyaml": "yaml",
    "pytest": "pytest",
}


def library_versions() -> dict[str, str]:
    """Import every required library (fails loudly if one is missing) and return its version."""
    return {pip_name: importlib.import_module(mod).__version__ for pip_name, mod in LIBRARIES.items()}


def main() -> None:
    cfg = load_config()
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": library_versions(),
    }
    out = results_dir() / "environment.json"
    out.write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"Python {env['python']} on {env['platform']}")
    for name, version in env["libraries"].items():
        print(f"  {name:<13} {version}")
    print(f"Config: {len(cfg['tickers'])} tickers, {len(cfg['features'])} features, "
          f"lookback {cfg['lookback']}, horizon {cfg['horizon']}, missing_days_rule={cfg['missing_days_rule']}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
