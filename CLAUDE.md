# CLAUDE.md — PM2.5 India Forecasting

> **Re-read this file at the start of every session.** Keep the Day checklist and the Decisions log current: update both after every confirmed decision and every completed, verified item.

## 1. Project context
**Study:** "Leakage-Free, Uncertainty-Aware PM2.5 Forecasting for Indian Cities". It is leakage-audited, cross-city and calibrated.
It is a **survey paper plus a small benchmark study**. We do NOT build a new ML algorithm. We build one reproducible forecasting pipeline and run three controlled experiments.

### RQ1 — Leakage
- **Protocol A:** random 80/20 split plus same-day AQI and co-pollutants. This is same-day *estimation*, kept as an intentionally invalid demonstration.
- **Protocol B:** chronological split plus same-day AQI and co-pollutants.
- **Protocol C:** chronological split plus strictly causal features. This is the genuine forecast.
- B and C are **two different tasks**. Never describe the B→C gap as the isolated causal effect of removing co-pollutants.

**Final RQ1 protocol definition (confirmed 2026-11-14):**

| | Task | Dataset (unit) | Inputs | Split |
|---|---|---|---|---|
| A | same-day estimation of PM2.5 on day d (invalid demo) | `rq1_same_day_dataset.parquet`, one row per (city, target_date) | same-day AQI + 11 co-pollutants + calendar(d) | random 80/20 over unique city-dates, stratified by city, seed from YAML; pool = target dates in 2015–2017 ∪ 2019; 2018 excluded |
| B | same-day estimation | same table, same rows | identical to A | chronological: train 2015–2017, test 2019; 2018 reserved, unused |
| C | genuine forecast, h = 1, 3, 7 (direct) | `forecast_dataset.parquet`, one row per (city, origin, h) | `causal_common` (city-ID-free) | chronological by target date: train 2015–2017, test 2019 |

- Cities: the core five, pooled. AQI_Bucket is never used, since it's a binning of AQI. Same-day inputs are not filled; NaNs go to LightGBM.
- **A vs B** use the *same city-date pool* (2015–2017 ∪ 2019), so their gap estimates *split-induced inflation*. Split type is **not** the only difference, though:
  1. **Assignment:** A is random; B is chronological.
  2. **Test proportion:** A tests on ≈20% of each city's pool, which is exact up to rounding per city. B's test is the full 2019 year, ≈25–27% of the pool (Delhi 25.0%, Lucknow 26.9%, Chennai 27.2%, Hyderabad 27.2%, Bengaluru 27.4%), so B also trains on ≈6–9% fewer city-dates.
  3. **Test period:** A's test rows are spread across 2015–2017 and 2019, while B's are all in 2019.
  - Report A vs B with this qualification (`outputs/tables/rq1_same_day_split_counts.csv` has the shares). Never say the split type is the sole difference.
- **B vs C** compares *same-day estimation with a chronological split* against *genuine forecasting*. These are two different tasks, scored on the common set of 2019 (city, target_date) pairs where both B has a row and C's `eval_mask` holds at that h. Full-set numbers are also reported. Never attribute the gap to co-pollutants alone.
- The random split must never be applied to the three-horizon long table. Identical (city, target_date) targets occur there under several horizons, and `random_split_unique` raises on duplicates.
- Under A, the same calendar date can appear in train for one city and in test for another city. That is part of what A demonstrates; the (city, date) pairs themselves never overlap.

### Model validation (revised 2026-11-14, superseding the earlier "fixed" choice)
- 2018 is used only for CQR calibration. 2019 is used only for final testing. Neither is ever used for hyperparameter selection.
- **Selection:** expanding-window CV with a small predeclared grid (`tuning.grid`).
  - Two folds by target date: train 2015 → validate 2016, then train 2015–2016 → validate 2017.
  - Score: validation MAE, averaged over folds and horizons.
  - Selection data: protocol C, core cities, `causal_common`, h = 1, 3, 7.
  - `assert_training_only` guards every row a fit or prediction touches.
- **Refit:** the selected parameters are refit on all of 2015–2017 (`split_main == train`).
- **Reuse:** one selected parameter set is stored in `outputs/tables/lightgbm_selected_params.json` and reused everywhere: RQ1 A/B/C, RQ2, and the RQ3 quantile models via `quantile_params`.
- No early stopping.
- The grid values and selection scope are **proposed** (`tuning.status`) until the user freezes them.

### RQ2 — Cross-city
- Pooled vs per-city vs leave-one-city-out (zero-shot).
- City ID is categorical. The held-out city is treated as unseen, or city ID is omitted.
- **Feature sets (confirmed 2026-11-14):**
  - The primary comparison uses `causal_common`, which has no city ID, for pooled, per-city and LOCO alike.
  - `causal_city_aware` (causal_common + city) is an optional ablation for the **pooled** model only. LOCO refuses it.
  - Any city absent from training is encoded as an explicit unseen value (NaN; categories = training cities only) by `encode_city`.
  - LOCO training partitions contain zero rows from the held-out city. Transfer cities are never in training.

### RQ3 — Uncertainty
- Quantile LightGBM (0.05 / 0.95) vs CQR (calibrated on 2018) vs CQR+ACI.
- ACI update: `alpha_{t+1} = alpha_t + gamma * (alpha - err_t)`, with alpha = 0.10 and alpha_t bounded.
- ACI runs per city × horizon with **delayed feedback**. Pending predictions sit in a queue, and state is updated only once the target is observed, i.e. h days later.
- Source-city calibration and held-out-city calibration are **distinct experiments**.
- ACI gives **long-run marginal coverage only**. Never claim per-season, per-city or extreme-event guarantees; measure those empirically.

### Models
- Persistence: `y_hat[t+h] = y[t]`.
- 7-day seasonal naive: `y_hat[t+h] = y[t+h-7]`.
- Pooled LightGBM point model.
- LightGBM quantile models.
- **Direct forecasting**, with a separate model per horizon h ∈ {1, 3, 7}.
- Out of scope, future work only: ARIMA, deep learning, SHAP, weather, AgACI, conformal PID.

### Metrics
- Point: MAE, RMSE, R², skill vs persistence.
- Interval: PICP, MPIW, Winkler score.
- Breakdowns:
  - horizon;
  - four Indian seasons: winter Dec–Feb, summer/pre-monsoon Mar–May, monsoon Jun–Sep, post-monsoon Oct–Nov;
  - top-10% pollution days, with the threshold taken from the **training-period** PM2.5 distribution, **per city**;
  - per city;
  - held-out cities.
- **Never paste other papers' numbers into our results tables.** Literature comparison belongs in Related Work / Discussion only.

## 2. Frozen design (do not change without asking the user)
- **Dataset:** Kaggle "Air Quality Data in India (2015-2020)" by Vopani, **version 12**, CC0, compiled from CPCB.
  - Only `city_day.csv` is used for modelling.
  - sha256 `0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847`; 29,531 rows; 26 cities; dates 2015-01-01 → 2020-07-01.
- **Raw data is IMMUTABLE.** Never edit, overwrite or re-save anything in `data/raw/`. Derived data goes in `data/processed/`.
- **Cities:**
  - Core (final selection to be verified from the completeness report): Delhi, Lucknow, Chennai, Hyderabad, Bengaluru.
  - Transfer-only targets: Kolkata, Mumbai (PM2.5 only from April 2018), and optionally Visakhapatnam.
  - Alternates for a sixth city: Patna or Gurugram.
- **AQI and AQI_Bucket are never features in protocol C**, because AQI is computed from same-day PM2.5.
- **Splits are assigned by TARGET date, never by forecast-origin date.** This stops h = 7 samples from crossing split boundaries.
  - Main: train 2015–2017 / calibration 2018 / test 2019.
  - Stress test, reported separately from primary results: train 2015–2018 / calibration 2019 / test 2020-01-01 → 2020-07-01. Pre-lockdown (< 2020-03-25) and lockdown are reported separately.
- **Forecast origin t:** X_t → y_{t+h}. Every feature must be knowable at date t.
  - PM2.5 lags of 1, 2, 3 and 7 days.
  - Rolling mean and std over 3, 7 and 14 days, computed on the series **shifted** so they use observations up to and including t only.
  - Calendar features of the **target** date (month, day of week, sin/cos of day of year). These are allowed because they are known in advance.
  - City ID as a categorical feature.
- **Missing values:**
  - Forward-fill at most 2 days, on **input features only**.
  - Never fill the target. Never backfill or interpolate.
  - Add an imputed flag.
  - Evaluate only on observed targets.
  - A sensitivity run with limit = 3 comes later.
- **Reproducibility:**
  - Fixed seed in `configs/experiments.yaml`.
  - All paths and parameters live in the YAML.
  - Every output table records the dataset sha256 and Kaggle version in its metadata, as a `<table>.meta.json` sidecar.

## 3. Ground rules
- Plan first (plan mode) for each work block and wait for user approval.
- Any architecture, pipeline, design or modelling decision not fixed above must be **proposed to the user via AskUserQuestion, with options and a recommendation, and confirmed before writing code that depends on it.** Never decide silently.
- Python only, local development, macOS, venv at `.venv` (Python 3.11). No Kaggle notebooks, no cloud.
- Verification must be executable (pytest), not narrative.

## 4. Repo map
```
CLAUDE.md                 this file (project memory and rules)
README.md                 setup and run commands
requirements.txt          pinned dependencies
configs/experiments.yaml  seed, paths, dataset identity, stage parameters
data/raw/city_day.csv     canonical input (read-only, git-ignored)
data/raw/unused/          other 4 Kaggle files, not used in this study
data/processed/           derived parquet (git-ignored)
notebooks/01_eda.ipynb    EDA (kernel "pm25")
src/data_audit.py         Day 1 items 1–3: config load, sha256, load, verify, completeness/gap reports, table+metadata writer
src/preprocessing.py      Day 1: city selection/roles, daily calendar series, input ffill (limit 2)
src/features.py           causal features h = 1,3,7, named feature sets, RQ1 same-day table, encode_city, exports
src/splits.py             target-date splits, RQ1 A/B splits, B-vs-C common index, tuning guard + expanding folds, LOCO
src/models.py             baselines (Day 1 item 8), fixed lightgbm_params; LightGBM training later (RQ1–3)
src/conformal.py          later: CQR, CQR+ACI (RQ3)
src/metrics.py            later: point and interval metrics, breakdowns
src/experiments.py        later: protocol A/B/C, cross-city, uncertainty runners
tests/                    executable verification (data audit, features/causality, splits, baselines/export)
outputs/tables|figures|predictions/
```

## 5. Run commands
```bash
source .venv/bin/activate
python -m src.data_audit            # Day 1 items 1–3: verify raw data, write audit tables
python -m src.features              # Day 1 items 4–9: build + export processed parquet and sample tables
python -m src.features rq1          # RQ1 A/B same-day table -> data/processed/rq1_same_day_dataset.parquet
pytest -q                           # all tests
jupyter lab                         # pick kernel "Python (pm25)"
shasum -a 256 data/raw/city_day.csv # must equal frozen sha256
```

## 6. Day 1 checklist (live status)
| # | Item | Status | Verified by |
|---|------|--------|-------------|
| 1 | Load the canonical city_day.csv | ✅ 2026-11-13 | `pytest tests/test_data_audit.py` (load, dtypes, raw hash unchanged) |
| 2 | Verify SHA-256, dimensions, dates and columns | ✅ 2026-11-13 | `verify_raw` + tamper tests (rows/cols/cities/dupes/dates/hash) |
| 3 | Generate city-wise PM2.5 completeness report | ✅ 2026-11-13 | `python -m src.data_audit` → outputs/tables/pm25_completeness_by_city_{year,period}.csv, pm25_gap_stats.csv (+ .meta.json); accounting tests |
| 4 | Select the five core cities | ✅ 2026-11-13 | `tests/test_features.py::test_city_roles`, `test_core_cities_meet_training_coverage` (≥85% train coverage) |
| 5 | Build causal features for h = 1, 3, 7 | ✅ 2026-11-13 | `src/features.py`; `test_feature_values_match_definitions`, `test_ffill_limit_is_respected` |
| 6 | Verify every feature is available at its forecast origin | ✅ 2026-11-13 | future-perturbation tests (8 cities + end-to-end) + control test proving leaky features are caught |
| 7 | Verify target-date-based chronological splitting | ✅ 2026-11-13 | `tests/test_splits.py` (ranges, no train target ≥ cal start, straddling samples, stress periods) |
| 8 | Implement persistence and seasonal-naive baselines | ✅ 2026-11-13 | `src/models.py`; `tests/test_baselines_export.py` (common `eval_mask`) |
| 9 | Export processed dataset and audit report | ✅ 2026-11-13 | `python -m src.features` → data/processed/forecast_dataset.parquet (+ provenance), outputs/tables/usable_samples_*, feature_missingness_*; export round-trip test |

**Day 1 complete** (committed 31a4836). Do NOT start Day 2 until the user says so.

### Pre-Day-2 fixes (2026-11-14), not yet committed and awaiting user review
| Item | Status | Verified by |
|---|---|---|
| RQ1 same-day table (one row per city-date), A random-unique / B chronological | ✅ | `tests/test_rq1.py` |
| RQ1 documentation: A ≈20% vs B full-2019 (≈25–27%) test shares; same pool | ✅ | CLAUDE.md §1; `rq1_same_day_split_counts.csv` (`share_of_pool`) |
| Hyperparameter rule: expanding-window CV, predeclared grid, val MAE, refit 2015–2017, one reused set | ✅ code; ☐ grid/scope freeze | `tests/test_validation_rule.py` (spy model, no training) |
| LightGBM params: subsample_freq 1, bagging_seed 42, deterministic + force_col_wise | ✅ | `tests/test_lightgbm_params.py` |
| Return types: `lightgbm_params` / `ffill_inputs` / `baseline_predictions` (no trailing commas found) | ✅ | `tests/test_return_types.py`, `tests/test_lightgbm_params.py` |
| RQ2 common feature sets, LOCO isolation, unseen-city encoding | ✅ | `tests/test_rq2_features.py` |
| Pylance issues (features.py, test_features.py, splits.py, test_rq1.py) | ✅ code changed | pyright not run (CLI permission denied); confirm in editor |
| Hyperparameter selection actually run | ☐ not run (counts as training; waits for Day 2 go-ahead) | |

Total: 102 tests pass, 0 warnings.

Setup: ✅ scaffold · ✅ venv (Py 3.11) + pinned requirements · ✅ imports verified (all 13) · ✅ Jupyter kernel `pm25` · ✅ git init · ☐ initial git commit (awaiting user approval of skeleton)

Known gotcha: in YAML, quote column names — unquoted `NO` parses as boolean False (caught by `verify_raw`).

## 7. Decisions log
| Decision | Chosen option | Date | Confirmed by user |
|----------|---------------|------|-------------------|
| Repo root | Current dir `~/Projects/pm2.5-india-forecasting` (no nested folder) | 2026-11-13 | yes |
| Python for .venv | Homebrew Python 3.11 | 2026-11-13 | yes |
| Table metadata format | Sidecar `<table>.meta.json` (sha256, Kaggle version, git commit, timestamp) | 2026-11-13 | yes (approved in plan) |
| Raw copies protection | `chmod a-w` on files in data/raw after copying | 2026-11-13 | yes (approved in plan) |
| Core cities | Delhi, Lucknow, Chennai, Hyderabad, Bengaluru only (train-period PM2.5 coverage 88–100%); no Patna/Gurugram (≈57%) | 2026-11-13 | yes |
| Transfer cities | Kolkata, Mumbai = transfer-only; Visakhapatnam = optional external held-out transfer, never in core training | 2026-11-13 | yes |
| Processed table scope | All 8 cities in one table with `city_role` = core / transfer / transfer_optional | 2026-11-13 | yes |
| Missing origin y_t | Drop the sample. Never impute y_t or the target | 2026-11-13 | yes |
| `pm25_origin` feature | y_t is an explicit feature; lags 1,2,3,7 mean y_{t-1}, y_{t-2}, y_{t-3}, y_{t-7} | 2026-11-13 | yes |
| Missing lag inputs | Forward-fill historical lag inputs ≤2 days, then keep NaN (LightGBM handles it natively); per-feature imputation flags | 2026-11-13 | yes |
| Rolling stats | Computed on genuinely observed (raw, unfilled) values over [t-w+1, t]; min observations 2/3, 4/7, 7/14; otherwise NaN | 2026-11-13 | yes |
| Processed layout | One long parquet, one row per (city, origin_date, horizon): target_date, horizon, y_origin, causal features, y_target, imputation flags, main and stress split labels | 2026-11-13 | yes |
| Seasonal-naive input | Raw observed y[t+h-7] only (no fill) | 2026-11-13 | yes |
| Common evaluation sample | `eval_mask` = target observed AND y_t observed AND raw y[t+h-7] observed; all models and baselines are scored on exactly this mask | 2026-11-13 | yes |
| Outliers | No treatment; keep CPCB values as reported | 2026-11-13 | yes |
| Config structure | Single `configs/experiments.yaml` with per-stage sections | 2026-11-13 | yes |
| Initial commit | Day 1 state committed as approved (31a4836) before pre-Day-2 changes | 2026-11-14 | yes ("Day 1 is approved") |
| RQ1 A/B dataset | Separate same-day table, one row per (city, target_date), core five; never random-split the long table | 2026-11-14 | yes |
| RQ1 A pool | Random 80/20 stratified by city over target dates 2015–2017 ∪ 2019 (same pool as B), 2018 excluded | 2026-11-14 | yes |
| RQ1 A/B inputs | Same-day AQI + PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene + calendar; no AQI_Bucket; no fill | 2026-11-14 | yes |
| B vs C scoring | Common (city, target_date) set in 2019 (B row ∧ C eval_mask at h); full sets also reported | 2026-11-14 | yes |
| Hyperparameters (superseded) | ~~Predeclared fixed LightGBM params~~ | 2026-11-14 | yes, then replaced |
| Hyperparameters | Expanding-window CV (2015→2016, 2015–16→2017), small predeclared grid, validation MAE, refit 2015–2017, one selected set reused across all experiments; 2018/2019 never used | 2026-11-14 | yes (user instruction; grid + selection scope proposed, pending freeze) |
| LightGBM base params | subsample 0.8 with subsample_freq 1 (bagging active), bagging_seed 42, deterministic + force_col_wise | 2026-11-14 | yes |
| RQ2 feature sets | Primary `causal_common` (no city ID); `causal_city_aware` = pooled-only ablation; unseen city → NaN | 2026-11-14 | yes |

## 8. Open decisions (ask the user before the day that needs them)
- Top-10% threshold for transfer cities: Kolkata/Mumbai have no 2015–17 data, so a "training-period" threshold is undefined in the main scheme. Needed on the metrics day.
- Whether `city_role = transfer_optional` (Visakhapatnam) runs in RQ2 at all.

## 9. Do not
- Never touch `data/raw/` (no edits, overwrites or re-saves).
- Never use AQI, AQI_Bucket or same-day co-pollutants as features in strict (protocol C) experiments.
- Never fill (impute) the target.
- Never assign splits by origin date; always use the target date.
- Never random-split, except in protocol A.
- Never compare other papers' metrics to ours in results tables.
- Never start a later Day's work until the user says so.
- Never claim ACI/CQR guarantee conditional or per-regime (season/city/extreme) coverage.
