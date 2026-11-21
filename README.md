# CS495 Portfolio Risk Forecasting

Forecast next month's 20 × 20 covariance matrix of 20 US-listed ETFs and use it in a
minimum-variance portfolio. This repository holds the data pipeline and the traditional
baselines (Deliverable 2); the Transformer model is added later. Project rules: `CLAUDE.md`.
All settings live in `config.yaml`.

## Install

Python 3.10+.

```powershell
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` makes the `pipeline` package importable so `python -m pipeline.<step>` works
from the project root. `pytest` works without it (it adds `src/` to the path itself).

## Run order

Run from the project root, one step at a time:

| Step | Command | Notes |
|---|---|---|
| 0 | `python -m pipeline.environment` | library versions → `results/environment.json` |
| 1 | `python -m pipeline.download` | needs internet; refuses to run if `data/raw/funds.parquet` exists (frozen snapshot) |
| 2a | `python -m pipeline.clean` | missing-days report; then the **team** sets `missing_days_rule` in `config.yaml` |
| 2b | `python -m pipeline.clean` | cleaning (errors until `missing_days_rule` is set) |
| 3 | `python -m pipeline.features` | |
| 4 | `python -m pipeline.examples` | |
| 5 | `python -m pipeline.splits` | |
| 6 | `python -m pipeline.eda` | dates ≤ `val_end` only |
| 7 | `python -m pipeline.baselines` | train + validation forecasts and scores |
| 8 | `python -m pipeline.backtest` | validation period only |

Tests (no internet needed):

```powershell
pytest
```

The test period (2020 onward) is sealed in this phase: no step reads results from it.

## Folders

- `data/` — raw snapshot, cleaned data, arrays, forecasts (git-ignored).
- `results/` — small tables and figures used in the reports (tracked); EDA in `results/eda/`.
- `tests/` — pytest tests on small synthetic data.
- `notebooks/` — one-off exploration only, never pipeline code.

## Result files and where they are used in Deliverable 2

Filled in as each step is implemented.

| File | Step | Used in |
|---|---|---|
| `results/environment.json` | 0 | Reproducibility appendix |
