# CS495 Portfolio Risk Forecasting

Forecast next month's 20 × 20 covariance matrix of 20 US-listed ETFs and use it in a
minimum-variance portfolio. This repository holds the data pipeline and the traditional
baselines (Deliverable 2); the Transformer model is added later. Project rules and team decisions:
`PROJECT_RULES.md`.
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
| 2b | `python -m pipeline.clean` | same command; cleans once `missing_days_rule` is set (team chose `ffill2`) |
| 3 | `python -m pipeline.features` | |
| 4 | `python -m pipeline.examples` | |
| 5 | `python -m pipeline.splits` | |
| 6 | `python -m pipeline.eda` | dates ≤ `val_end` only |
| 7a | `python -m pipeline.baselines` | train + validation forecasts (DCC-GARCH takes ~20 s) |
| 7b | `python -m pipeline.metrics` | validation scores and Diebold–Mariano tests |
| 8 | `python -m pipeline.backtest` | validation period only |
| — | `python -m pipeline.diagnostics` | report-only data checks; never changes data or `results/volume_corrections.csv` |

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

Section names are suggestions; adjust to the document's final outline.

| File | Step | Used in |
|---|---|---|
| `results/environment.json` | 0 | Appendix: reproducibility |
| `data/raw/manifest.json` (git-ignored), `results/download_checks.json` | 1 | Data: source, period, snapshot checksums |
| `results/eda/spy_adj_close.png` | 1 | Data: visual sanity check |
| `results/missing_days_per_fund.csv`, `results/missing_gaps.csv` | 2a | Data quality: missing days and the team's rule |
| `results/quality_report.json`, `results/flags.csv` | 2b | Data quality: what was filled or dropped |
| `results/extreme_moves.csv`, `results/split_suspects.csv`, `results/sanity_violations.csv` | 2b | Data quality: moves > 25 %, split checks, impossible prices |
| `results/volume_corrections.csv` | 2b input | Data quality: hand-edited corrections applied by the cleaning step (7 IWM volumes) |
| `results/volume_check*.csv`, `stooq_volume_basis.json`, `volume_corrections_proposed.csv`, `uso_split_check.json`, `garch_variants_train.csv` | diagnostics | Data quality: volume errors, Stooq cross-check, USO split, GARCH variants |
| `results/feature_checks.json`, `results/volume_suspects.csv` | 3 | Features: definitions, warm-up, data issues |
| `results/example_checks.json` | 4 | Method: examples, window and label shapes |
| `results/split_checks.json`, `results/scaler.json` | 5 | Method: splits and scaling |
| `results/eda/summary_stats.csv` | 6 | EDA: summary table per fund |
| `results/eda/rolling_vol.png`, `acf_spy.png` | 6 | EDA: volatility changes and clusters |
| `results/eda/corr_heatmaps.png`, `rolling_corr_spy_tlt.png`, `eda_facts.json` | 6 | EDA: correlations change (calm vs crisis, stock–bond) |
| `results/eda/vix_vs_future_vol.png` | 6 | EDA: VIX predicts next-month volatility |
| `results/eda/scaled_feature_hists.png`, `scaled_feature_ranges.csv` | 6 | Features: scaled distributions |
| `results/baseline_checks.json` | 7 | Baselines: EWMA λ choice, DCC parameters, forecast checks |
| `results/scores.csv`, `results/dm_tests.csv` | 7 | Results: validation scores and significance |
| `results/backtest_summary.csv`, `results/backtest_wealth.png` | 8 | Results: portfolio backtest |
| `results/backtest_daily.csv`, `results/backtest_weights.csv` | 8 | Appendix: daily returns and weights |
