# CS495 Portfolio Risk Forecasting — project rules

## What this project is
A deep-learning course project. We forecast next month's 20 × 20 covariance matrix ("risk table")
of 20 US-listed ETFs with a small Transformer trained from scratch, then use it in a
minimum-variance portfolio. This repository starts with the data pipeline and the
traditional (non-deep-learning) baselines; the model is added later in the same repository.

## Fixed decisions (do not change without the team's approval)
- Data: Yahoo Finance via `yfinance`, adjusted close, plus `^VIX` and `^IRX` (13-week T-bill, annual %).
- Funds (this order everywhere): SPY, IWM, QQQ, XLK, XLF, XLE, XLV, XLU, XLP, TLT, IEF, LQD, HYG, GLD, SLV, USO, DBA, VNQ, EFA, EEM.
- Period: 2007-04-11 to 2026-10-30 (`end` in yfinance is exclusive, so config uses 2026-10-31).
- Look-back 63 trading days; horizon 21 trading days (= monthly rebalancing).
- 11 features per fund per day (order matters): log_ret, abs_ret, swing_5, swing_21, swing_63,
  ewma_swing (λ = 0.94), ret_21, rel_volume, log_vix, vix_change_21, hl_range.
- Labels: simple daily returns in percent for days t+1 … t+21.
- Splits by date: train = labels ending ≤ 2015-12-31; validation = day t in 2016 or later and labels ending
  ≤ 2019-12-31; test = day t from 2020. Examples whose label crosses a boundary are unused ("gap").
- Scaling: log of swing_5, swing_21, swing_63, ewma_swing, hl_range first; then z-score with one mean and std
  per feature, pooled across the 20 funds, computed on training days only (dates ≤ 2015-12-31).
- Baselines: historical (63 days), EWMA (λ 0.94 and 0.97, keep the better on training NLL), Ledoit–Wolf, DCC-GARCH.
- Backtest: rebalance every 21 trading days, long-only, fully invested, NO maximum weight yet,
  equal weight as reference, trading cost = 0 for now (kept as a config setting).
- The test period (2020 onward) is sealed: no code in this phase may read results, plots or settings from it.

## Data-quality decisions (team, 2026-11)
- **Missing fund-days:** `missing_days_rule: "ffill2"` in `config.yaml`. The report found no fund gaps,
  only 2 ^IRX bond-market holidays, which are forward-filled.
- **Extreme moves** (|simple return| > 25%) are flagged and kept unless proven to be data errors.
- **Corrections only through one file:** values are corrected ONLY through `results/volume_corrections.csv`
  (fund, date, field, old_value, new_value, source, method, reason). The team edits it by hand and the
  cleaning step applies it. Each row must match the raw value it replaces, or cleaning stops with an error.
  There is no automatic correction rule.
- **Current corrections:** 7 IWM volume errors (6 days in 2008, 2019-07-30), confirmed by Stooq, replaced
  with round(stooq_volume × stooq_close / yahoo_raw_close), because Stooq's volume is dividend-adjusted.
- **Volume check is report-only:** `python -m pipeline.diagnostics` may propose corrections in
  `results/volume_corrections_proposed.csv` but never changes data or the corrections file.
- **Real volume spikes, kept:** IEF (2014), USO (2026-03) and HYG (2007-09). No second source needed.

## Model decisions and open points
- DCC-GARCH stays as is for Deliverable 2: normal GARCH(1,1) and the current IGARCH-safe multi-day variance
  formula.
- Open, to test before the Transformer comparison: HYG persistence (α + β = 1) and Student-t / GJR GARCH.
  Do not change them before then.

## The one rule that matters most
Nothing computed for day t may use data dated after t (except labels and the "realized" evaluation
quantities, which are explicitly future by design). Negative shifts (`shift(-k)`) are forbidden in feature code.

## Coding style — simple first
- Plain Python 3.10+, `numpy`, `pandas`. Functions, not classes, unless a class is clearly simpler.
- One module per pipeline step in `src/pipeline/`. Each module has small pure functions plus a `main()`
  that reads inputs, calls the functions, and saves outputs. Run a step with `python -m pipeline.<module>`.
- Functions short (aim < 40 lines), one job each, type hints, a one-line docstring saying WHY, not what.
- No hard-coded numbers that live in `config.yaml` (dates, 63, 21, tickers, λ, thresholds, seed).
  Shapes come from the data or the config: N funds, L look-back, H horizon, F features.
- Each step reads the previous step's files and writes its own; never recompute an earlier step inside a later one.
- Clear names over clever code. No premature abstractions, no frameworks, no notebooks for pipeline code
  (notebooks are fine for one-off exploration in `notebooks/`).
- Use float64 for covariance and optimization code.
- Every random choice uses `config.yaml: seed`.

## Files and folders
- `config.yaml` — all settings. `src/pipeline/config.py` loads it and builds paths.
- `data/` — raw snapshot, cleaned data, arrays (git-ignored; large).
- `results/` — small tables (CSV/JSON) and figures (PNG) used in the reports (tracked).
- `tests/` — pytest tests on small synthetic data; they must run without internet.

## Working rules
- Work one step at a time. After each step, run its checks and the tests before moving on.
- Ask the team before adding any dependency not in `requirements.txt`.
- Ask the team before changing any fixed decision above, even if something seems better.
- Never modify or re-download `data/raw/` once it exists; a new download goes to a new dated folder.
- Commit after each step passes its checks, with a short message.
- When unsure, choose the simpler implementation and say what you simplified.
