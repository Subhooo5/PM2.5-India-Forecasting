# Design, wording rules and decisions

**Read when** you are about to change a split, a feature, a model setting, an evaluation rule or a sentence that describes the results.
Everything here is fixed: changing it needs the project owner's approval.

## 1. Study

"Leakage-free, uncertainty-aware PM2.5 forecasting for Indian cities" is a survey plus a small benchmark. One reproducible pipeline runs three controlled experiments and one stress test. No new learning algorithm is proposed.

| Experiment | Question | Module |
|---|---|---|
| Leakage audit | How optimistic do random splits and same-day inputs make PM2.5 "forecasts"? | `src/leakage_protocols.py`, `src/leakage.py` |
| Cross-city | Pooled vs per-city vs leave-one-city-out (LOCO), plus external-city transfer | `src/cross_city.py` |
| Uncertainty | Raw quantile LightGBM vs split CQR vs CQR + ACI | `src/intervals.py`, `src/uncertainty.py` |
| 2020 stress test | The same pipeline under the 2020 distribution shift | `src/stress_forecasts.py`, `src/stress.py` |

## 2. Data

- **Source:** Kaggle "Air Quality Data in India (2015-2020)" by Vopani, **version 12**, CC0, compiled from CPCB. Only `city_day.csv` is used.
- **Identity:** sha256 `0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847`; 29,531 rows; 26 cities; 2015-01-01 to 2020-07-01. `src/data_audit.verify_raw` checks all of these before any table is built.
- **`data/raw/` is immutable.** Derived data lives only in `data/processed/`.
- **Cities:**
  - Core: Delhi, Lucknow, Chennai, Hyderabad, Bengaluru. Each has 88–100% PM2.5 coverage over 2015–2017.
  - Transfer only: Kolkata and Mumbai, whose PM2.5 starts in April 2018.
  - Optional transfer: Visakhapatnam.
  - Patna and Gurugram (about 57% coverage) are excluded.
- **Outliers:** none are treated; CPCB values are kept as reported.

## 3. Splits (always by TARGET date)

| Scheme | Train | Calibration | Test |
|---|---|---|---|
| main | 2015–2017 | 2018 | 2019 |
| stress | 2015–2018 | 2019 | 2020-01-01 to 2020-07-01 (pre_lockdown < 2020-03-25 ≤ lockdown) |

- Splits are assigned by target date, never by origin date, so an h = 7 sample can never straddle a boundary.
- 2018 targets are never trained on, tuned on or scored. They only supply CQR calibration scores.
- 2018 observations may enter the causal inputs of early-2019 origins. At h = 1 / 3 / 7, the origin falls in 2018 for 5 / 15 / 35 test rows, and the lookback reaches 2018 for 70 / 80 / 100 rows. The counts are in the `leakage_protocol_c` sidecar.
- In the stress scheme the same holds with 2019 in place of 2018.

## 4. Features and missing values

- **Forecast origin t:** inputs X_t predict y_{t+h}, with direct models for h = 1, 3 and 7. Every input is known at t.
- **`causal_common` (19 columns):**
  - `pm25_origin` (y_t);
  - lags 1, 2, 3 and 7, each with an imputed flag;
  - rolling mean and std over 3, 7 and 14 days, computed on raw observed values in [t − w + 1, t], with minimum counts 2 / 4 / 7;
  - calendar features of the target date (month, day of week, sin/cos day of year).
- **`causal_city_aware`:** `causal_common` plus city. It is an ablation only, used by pooled models and refused by LOCO. Cities absent from training are encoded as NaN.
- **Never inputs to a forecast:** AQI, AQI_Bucket and the 11 co-pollutants. `models.check_feature_set` rejects them even if the YAML list is weakened, and it also rejects target columns in every feature set.
- **Missing values:**
  - Forward-fill historical inputs for at most 2 days.
  - Never fill the target or y_t, never backfill, never interpolate.
  - A sample whose y_t is missing is dropped.
- **`eval_mask`:** the target, y_t and the raw y[t + h − 7] are all observed. Every model and baseline is scored on exactly these rows.

## 5. Models and tuning

- **Baselines:**
  - Persistence: ŷ[t+h] = y[t].
  - 7-day seasonal naive: ŷ[t+h] = y[t+h−7]. This equals persistence at h = 7 by definition.
- **LightGBM base:**
  - regression objective, 500 trees, learning rate 0.05;
  - subsample 0.8 with subsample_freq 1, colsample_bytree 0.8, reg_lambda 1.0;
  - seeds 42, n_jobs 1, deterministic, force_col_wise;
  - **no early stopping**.
- **Tuning:**
  - Expanding-window CV by target year: 2015 → 2016, then 2015–2016 → 2017.
  - Grid c1–c6 over num_leaves {15, 31, 63} × min_child_samples {20, 50}.
  - Score: mean validation MAE over 2 folds × 3 horizons, on `eval_mask` rows. Ties go to the earliest candidate.
  - Hard guard: target dates ≤ 2017-12-31.
  - **Selected: c2** (num_leaves 15, min_child_samples 50; mean validation MAE 26.633). `run_tuning` re-derives the selection and refuses to continue if it differs from `lightgbm.selected`.
  - Every point model is refit on all 2015–2017 rows. The quantile models change only the objective and alpha.
- **Out of scope:** ARIMA, deep learning, SHAP, weather, AgACI, conformal PID.

## 6. Experiments

**Leakage protocols:**

| | Task | Unit | Inputs | Split |
|---|---|---|---|---|
| A | same-day estimation (invalid demo) | (city, date) | same-day AQI + 11 co-pollutants + calendar | random 80/20 per city over 2015–2017 ∪ 2019 (seeds 42–46) |
| B | same-day estimation | (city, date) | as A | train 2015–2017, test 2019 |
| C | genuine forecast, h = 1, 3, 7 | (city, origin, h) | `causal_common` | train 2015–2017, test 2019 |

- **A vs B** share the same pool, but differ in three ways:
  - assignment: interleaved random vs chronological;
  - test share: ≈ 20% vs the full 2019 year, ≈ 25–27% of the pool;
  - test period: A's test rows span 2015–2017 and 2019.
- **B vs C** are two different tasks. They are scored on the common 2019 (city, date) set: 1,825 rows per horizon.
- The random split is never applied to the multi-horizon table; `random_split_unique` raises on duplicates.

**Cross-city:**
- Pooled, per-city and LOCO models all use `causal_common`.
- LOCO training frames contain zero rows of the held-out city.
- Transfer applies the pooled core-city models to Kolkata, Mumbai and (optionally) Visakhapatnam.

**Uncertainty:**
- Quantile LightGBM at 0.05 / 0.95.
- Split CQR uses one q_hat per horizon, pooled over the core cities' 2018 `eval_mask` scores. q_hat is the k-th smallest score, k = ⌈(n+1)(1−α)⌉. Intervals are never clipped at 0 and crossed bounds are never swapped.
- ACI:
  - Update rule: alpha_{t+1} = alpha_t + γ(α − err_t), with α = 0.10, γ = 0.02 and alpha_t bounded to [0.01, 0.30].
  - One stream per city × horizon, with delayed feedback: an update is applied only once the target is observed.
  - γ ∈ {0.005, 0.01, 0.02, 0.05} is a sensitivity analysis only, never used for selection.
- Held-out city: LOCO quantile models calibrated on the source cities' scores (primary), plus a separately labelled local calibration.
- Regimes: the top-10% threshold is the per-city 90th percentile of observed 2015–2017 PM2.5.
- Seasons:
  - winter: Dec–Feb;
  - pre-monsoon: Mar–May;
  - monsoon: Jun–Sep;
  - post-monsoon: Oct–Nov.

**Calibration-ready rule:**
- A test forecast is calibration-ready only if origin_date ≥ the last calibration day (2018-12-31 main, 2019-12-31 stress). Otherwise some calibration scores were not yet observable when the forecast was issued.
- Not-ready rows (0 / 10 / 30 at h = 1 / 3 / 7) stay in the prediction files with `calibration_ready = False`.
- Every interval metric uses rows that are test AND `eval_mask` AND `calibration_ready`.
- ACI streams start at the first ready forecast, with alpha_1 = 0.10.

**2020 stress test:**
- Train 2015–2018, calibrate on 2019, test 2020-01-01 to 2020-07-01 (84 pre-lockdown days and 99 lockdown days).
- Each city × horizon has one continuous ACI stream, never reset at 2020-03-25.

## 7. Wording rules

- **B → C:** never call the B → C gap the isolated effect of removing co-pollutants; B and C are different tasks.
- **A vs B:** never say split type is the sole A-vs-B difference.
- **LOCO and transfer:** call them "held-out-city generalisation with local history" and "external-city transfer with local history". Never call them zero-shot: inference uses the city's own past PM2.5.
- **Coverage claims:**
  - CQR gives marginal coverage only, and only under exchangeability. Autocorrelated daily data and the 2019 → 2020 shift violate exchangeability.
  - ACI gives long-run marginal coverage only, approximately, because alpha_t is bounded.
  - Never claim per-season, per-city, per-regime or extreme-event guarantees; report them empirically.
- **Regime strata:** they are defined on the realised outcome, so low coverage there is a finding, not a bug.
- **Calibration rows:** they are in-sample for CQR/ACI and are never scored.
- **2020:** it is a distribution-shift stress test, not a causal study of the lockdown. The periods differ in season as well as lockdown status. It is a partial year.
- **Other papers:** never paste their numbers into results tables; literature comparison belongs in Related Work / Discussion.

## 8. Decisions log (all confirmed by the project owner)

| Decision | Choice |
|---|---|
| Layout and config | One repo, single `configs/experiments.yaml`, venv `.venv` (Python 3.11), macOS, local only |
| Provenance | Every table, prediction file and model has a `.meta.json` sidecar (dataset sha256, Kaggle version, config hash, versions, git commit) |
| Raw data protection | `chmod a-w` on `data/raw/*`; never rewritten |
| Processed layout | One long parquet with one row per (city, origin_date, horizon) for all 8 cities, with `city_role`; plus a same-day table with one row per (city, date) for protocols A/B |
| Missing origin y_t | Drop the sample |
| Lag inputs | ffill ≤ 2 days, then NaN for LightGBM; per-lag imputed flags |
| Rolling stats | Raw observed values only; minimum counts 2/4/7 |
| Seasonal-naive input | Raw y[t+h−7], never filled |
| Common evaluation sample | `eval_mask` (target, y_t and seasonal reference observed) |
| A/B inputs | Same-day AQI + 11 co-pollutants + calendar; AQI_Bucket never used; NaNs left to LightGBM |
| A pool | 2015–2017 ∪ 2019, 2018 excluded; stratified by city; seeds 42–46 vary only the split |
| B vs C scoring | Common (city, date) set in 2019; full sets also reported |
| Hyperparameters | Expanding-window CV, predeclared grid, validation MAE, one selected set (c2) reused everywhere |
| Feature sets | `causal_common` primary; `causal_city_aware` pooled-only ablation |
| Skill | 1 − metric / persistence metric on identical rows |
| Model files | Every model saved as a LightGBM text file plus sidecar; saved quantile models are reused only if all metadata match exactly (hashes, dataset, params, features, horizon, range, n_train, cities, held-out city) |
| ACI | γ 0.02 fixed; bounds [0.01, 0.30]; per city × horizon; delayed feedback; issue after applying updates matured at the same origin |
| CQR | One pooled q_hat per horizon |
| Held-out-city uncertainty | Source-calibrated (primary) and local-calibrated, always separately labelled |
| Calibration-ready rule | See section 6 |
| Regime threshold | pandas `quantile(0.90)` per city on 2015–2017 observed PM2.5; top10 = y > threshold |
| Stress periods | By target date; 2018 = training and 2019 = calibration in the stress scheme only |
| Stress reference windows | Same month-day windows in 2019 and 2015–2018 (descriptive only; Feb 29 has no 2019 value) |
| Figures | matplotlib, 200 dpi, three categorical slots with hatching as a secondary encoding; built from saved outputs only |
| Display-only constants | Rolling-coverage window 30 and low-n flag n < 30 are code constants, not model parameters |
| Git | The project owner commits manually; `outputs/predictions/*` is git-ignored and needs `git add -f` |

## 9. Open decisions

- Top-10% threshold for the transfer cities: Kolkata and Mumbai have no 2015–2017 data, so a training-period threshold is undefined. Needed before any regime breakdown of transfer cities.
- Sensitivity run with ffill limit 3 (planned, not run).
- Optional `causal_city_aware` pooled ablation (defined, not run).
