# CS495 Portfolio Risk Forecasting

A deep-learning course project (CS495). It forecasts next month's risk table, the 20 × 20 covariance
matrix of daily returns for 20 US-listed exchange-traded funds (ETFs), and uses the forecast to build a
minimum-variance portfolio, rebalanced every month. The forecast will come from a small Transformer
trained from scratch, compared with four traditional forecasting methods.

**Status:** the pre-modeling phase is complete (data, checks, baselines, backtest); the Transformer is next.

| Asset class | Funds |
|---|---|
| US stocks | SPY (S&P 500), IWM (small caps), QQQ (Nasdaq-100) |
| US sectors | XLK (technology), XLF (financials), XLE (energy), XLV (health care), XLU (utilities), XLP (consumer staples) |
| US Treasuries | TLT (20+ years), IEF (7–10 years) |
| Corporate bonds | LQD (investment grade), HYG (high yield) |
| Commodities | GLD (gold), SLV (silver), USO (oil), DBA (agriculture) |
| Real estate | VNQ (US REITs) |
| International stocks | EFA (developed markets), EEM (emerging markets) |

- **Data:** daily prices from Yahoo Finance, 2007-04-11 to 2026-10-30, plus the VIX (expected stock-market
  volatility) and the 13-week T-bill rate (risk-free rate).
- **Each example:** for day t, the last 63 trading days × 20 funds × 11 features as input, and the next
  21 daily returns (one month) as target.
- **Splits by date:** training 2007–2015, validation 2016–2019, test 2020 onward.

All settings live in `config.yaml`.

## Where the project stands

**Done: the pre-modeling phase** (the course's Deliverable 2): data pipeline, data-quality checks,
exploratory analysis, four traditional baselines, and a portfolio backtest on the validation period.

**Next: the Transformer model**, added to this repository and compared with the baselines.

- **Test period sealed:** no result, plot or setting has been read from 2020 onward yet, so the final
  comparison stays honest.
- **Open points**, to test before that comparison: the HYG volatility fit that sits at its stability limit,
  and fat-tailed (Student-t) or asymmetric (GJR) GARCH variants. The DCC-GARCH baseline is kept as is until then.

## Pre-modeling outcomes

**Data:** 4,922 trading days for all 20 funds, with no missing fund-day, no impossible prices and no
split problems. 4,776 examples: 2,053 training, 985 validation, 1,696 test, 42 unused at split boundaries.

**What the exploratory analysis shows (2007–2019):**
- Volatility comes in clusters: SPY's squared returns stay autocorrelated for more than 60 days.
- Correlations jump in crises: the average stock–stock correlation is 0.38 in calm 2017 and 0.84 in
  Sep 2008–Mar 2009, while stocks vs Treasuries go from −0.15 to −0.40.
- The VIX on day t correlates 0.76 with SPY's realized volatility over the next 21 days.

**Baseline forecasts on validation** (985 examples; lower is better):

| Method | Gaussian NLL | Frobenius gap |
|---|---|---|
| DCC-GARCH | **13.87** | 9.50 |
| EWMA (λ = 0.97, chosen on training) | 15.16 | **8.73** |
| Ledoit–Wolf (63 days) | 16.82 | 9.07 |
| Historical (63 days) | 16.95 | 9.51 |

- **Gaussian NLL** (negative log-likelihood): how unlikely the next month's actual returns are under the
  forecast. It is the main score.
- **Frobenius gap:** distance between the forecast matrix and the matrix of actual next-month return products.
- **Diebold–Mariano test:** whether one method's average loss is significantly lower than another's.

DCC-GARCH has the best NLL, and its lead is significant against every other method (Diebold–Mariano,
p < 0.001). This is the score the Transformer must beat.

**Monthly minimum-variance backtest on validation** (2016-01 to 2019-12, long-only, no trading cost):

| Portfolio | Annual volatility | Annual return | Sharpe | Max drawdown |
|---|---|---|---|---|
| DCC-GARCH | **3.00 %** | 4.53 % | 1.08 | −3.79 % |
| EWMA | 3.04 % | 5.21 % | 1.29 | −3.67 % |
| Historical | 3.14 % | 5.39 % | 1.30 | −3.77 % |
| Ledoit–Wolf | 3.68 % | 6.43 % | 1.40 | −5.04 % |
| Equal weight | 8.32 % | 9.33 % | 0.97 | −12.70 % |

Minimum variance targets the lowest volatility, so volatility is the main measure here. Every
forecast-based portfolio has less than half the volatility of equal weight. With no maximum weight
yet, they hold mostly intermediate Treasuries (IEF, about 60 %) and high-yield bonds (HYG, 20–25 %).

## How inconsistent data was fixed

The raw download in `data/raw/` is frozen and never edited; every fix happens in the cleaning step and is
recorded in `results/flags.csv`.

| Issue found | Fix |
|---|---|
| **IWM volume errors:** 7 days (6 in 2008, 2019-07-30) with volume hundreds to thousands of times too low, e.g. 1,200 shares vs about 20 million. Stooq, a second source, shows normal volume on all 7. | Replaced with Stooq's volume. Stooq scales volume for dividends, so its scaling is undone: round(stooq_volume × stooq_close / yahoo_raw_close), which matches Yahoo on every other day (median ratio 1.000). The relative-volume outlier drops from 21 to 6.8 standard deviations; only IWM's relative-volume input changes. |
| **Volume spikes** on IEF (2014), USO (2026-03) and HYG (2007-09) | Checked; real trading, kept. |
| **Missing days:** none for the funds; 2 bond-market holidays for the T-bill rate | Previous day's rate carried forward (rule `ffill2` in `config.yaml`: fill gaps of up to 2 days with the last value). The VIX and T-bill series are aligned to the fund calendar the same way. |
| **Daily moves above 25 %:** 3 (USO twice in 2020, SLV in 2026), all in the test period | Real market events; kept and flagged. |
| **USO 1-for-8 reverse split** (2020-04-28) | Checked: prices and volume are both split-adjusted. No fix needed. |
| **HYG GARCH fit at persistence 1** (shocks never fade), where the usual long-run variance formula breaks | Same forecast written in an equivalent form that stays finite; tested against the original formula. |

Corrections happen **only** through `results/volume_corrections.csv`, edited by hand by the project team: each row
gives fund, date, field, old value, new value, source, method and reason, and must match the raw value it
replaces, or cleaning stops. `python -m pipeline.diagnostics` only reports and proposes corrections
(in `results/volume_corrections_proposed.csv`); it never changes data.

## Method notes

Choices that affect how the results should be read:

- **No look-ahead:** every input for day t uses data dated t or earlier; only the target and the
  evaluation use later data. A test changes all data after a day and checks that earlier features don't move.
- **Scaling:** volatility-type features are logged, then standardized with one mean and standard deviation
  per feature, pooled across funds and computed on training days only.
- **Validation examples:** day t from 2016-01-01, with the whole 21-day target ending by 2019-12-31. Examples
  whose target crosses a split boundary are not used.
- **DCC-GARCH:** one GARCH(1,1) per fund plus a DCC correlation model, all fitted on training days only, so
  its training-period forecasts are in-sample and its validation forecasts are out-of-sample. The monthly
  variance is the average of the GARCH forecasts over the next 21 days; the correlation forecast for the
  next day is used for the whole month (a simplification).
- **EWMA:** λ = 0.94 and 0.97 are both run; 0.97 is kept because it scores better on training data.
- **Backtest:** rebalanced every 21 trading days, long-only, fully invested, no maximum weight; weights drift
  with prices between rebalances. The trading cost (0 for now) would be charged on the first day after each
  rebalance; the first purchase from cash is left out of average turnover. The risk-free rate for the Sharpe
  ratio is the daily T-bill rate.

## Install

Python 3.10+.

```powershell
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` makes the `pipeline` package importable so `python -m pipeline.<step>` works
from the project root. `pytest` works without it (it adds `src/` to the path itself).

**Data is not in the repository** (`data/` is git-ignored). Step 1 downloads it from Yahoo Finance and
saves a frozen snapshot with checksums. Yahoo revises adjusted prices over time, so a new download can
differ slightly from the one behind the results here; `data/raw/manifest.json` records the snapshot.

## Run order

Run from the project root, one step at a time:

| Step | Command | Notes |
|---|---|---|
| 0 | `python -m pipeline.environment` | library versions → `results/environment.json` |
| 1 | `python -m pipeline.download` | needs internet; refuses to run if `data/raw/funds.parquet` exists (frozen snapshot) |
| 2 | `python -m pipeline.clean` | missing-days report, then cleaning: applies `results/volume_corrections.csv` and the `missing_days_rule` (`ffill2`) |
| 3 | `python -m pipeline.features` | |
| 4 | `python -m pipeline.examples` | |
| 5 | `python -m pipeline.splits` | |
| 6 | `python -m pipeline.eda` | dates ≤ `val_end` only |
| 7a | `python -m pipeline.baselines` | train + validation forecasts (DCC-GARCH takes ~20 s) |
| 7b | `python -m pipeline.metrics` | validation scores and Diebold–Mariano tests |
| 8 | `python -m pipeline.backtest` | validation period only |
| — | `python -m pipeline.diagnostics` | report-only data checks; never changes data or `results/volume_corrections.csv`. The IWM cross-check needs the Stooq file (below) |

Tests (no internet needed):

```powershell
pytest
```

The test period (2020 onward) is sealed in this phase: no step reads results from it.

**After editing `results/volume_corrections.csv`,** rerun steps 2 to 8 in order; each step reads the files
of the step before it.

**Stooq file for the diagnostics:** Stooq blocks scripted downloads, so get it by hand from
<https://stooq.com/q/d/l/?s=iwm.us&i=d> and save it as `data/external/stooq_2026-11-22/iwm_us_d.csv`.
Without it, the diagnostics skip the IWM cross-check; the rest of the pipeline does not need it.

## Folders

- `data/` — raw snapshot (`data/raw/`, frozen), cleaned data, arrays, forecasts (git-ignored).
- `data/external/` — second-source files downloaded by hand, one dated folder each (git-ignored).
- `results/` — small tables and figures (tracked); exploratory analysis in `results/eda/`.
- `src/pipeline/` — one module per step, run with `python -m pipeline.<module>`.
- `tests/` — pytest tests on small synthetic data; no internet needed.
- `notebooks/` — one-off exploration only, never pipeline code.

## Result files

All in `results/` (tracked), made by the step shown.

| File | Step | Shows |
|---|---|---|
| `results/environment.json` | 0 | Reproducibility: library versions |
| `data/raw/manifest.json` (git-ignored), `results/download_checks.json` | 1 | Data: source, period, snapshot checksums |
| `results/eda/spy_adj_close.png` | 1 | Data: visual sanity check |
| `results/missing_days_per_fund.csv`, `results/missing_gaps.csv` | 2 | Data quality: missing days and the fill rule |
| `results/quality_report.json`, `results/flags.csv` | 2 | Data quality: what was filled or dropped |
| `results/extreme_moves.csv`, `results/split_suspects.csv`, `results/sanity_violations.csv` | 2 | Data quality: moves > 25 %, split checks, impossible prices |
| `results/volume_corrections.csv` | 2 (input) | Data quality: hand-edited corrections applied by the cleaning step (7 IWM volumes) |
| `results/volume_check*.csv`, `stooq_volume_basis.json`, `volume_corrections_proposed.csv`, `uso_split_check.json`, `garch_variants_train.csv` | diagnostics | Data quality: volume errors, Stooq cross-check, USO split, GARCH variants |
| `results/feature_checks.json` | 3 | Features: definitions, warm-up, zero-volume days |
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
| `results/backtest_daily.csv`, `results/backtest_weights.csv` | 8 | Results: daily returns and weights |
