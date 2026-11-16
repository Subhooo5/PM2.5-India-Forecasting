# Outputs

**Read when** you need to find, read or regenerate a result file.
Every CSV and model has a `<name>.meta.json` sidecar. It records:
- the dataset sha256 and Kaggle version;
- `config_hash` and the package versions;
- the git commit;
- a one-line note;
- where relevant, q_hat, checks, regime thresholds and data-use counts.

`outputs/result_manifest.csv` lists the bytes and sha256 of every output, processed and config file after the last full run.

| Group | Files | Written by (`python run.py <stage>`) | How to read |
|---|---|---|---|
| Processed data | `data/processed/forecast_dataset.parquet` | preprocess | One row per (city, origin_date, horizon): causal features, `y_target`, `snaive_ref`, `eval_mask`, `split_main`, `split_stress`, `stress_period` |
| | `data/processed/leakage_same_day_dataset.parquet` | preprocess | One row per (city, date): same-day inputs, `split_A`, `split_B` (protocols A/B only) |
| Data audit | `pm25_completeness_by_city_{year,period}.csv`, `pm25_gap_stats.csv` | audit | PM2.5 coverage against a full daily calendar; gap-length bins |
| Sample audit | `usable_samples_by_city_split_horizon.csv`, `feature_missingness_by_city_horizon_split.csv`, `leakage_same_day_split_counts.csv` | features | Rows and `eval_mask` rows per split; feature NaN / imputed shares; A/B split shares |
| Tuning | `models/tuning_cv_results.csv`, `models/selected_params.json` | tuning | 36 rows (6 candidates × 2 folds × 3 h); the ranking and the selected c2 |
| Leakage | `predictions/leakage_protocol_{a,b,c}.csv`, `tables/baseline_results.csv`, `tables/leakage_protocol_a_seeds.csv`, `tables/leakage_comparison.csv`, `models/point_h{1,3,7}.txt` | leakage | C is long format (method ∈ lightgbm / persistence / seasonal_naive). `mask_type` = `common_eval_mask` means identical rows. The comparison `notes` column states the sign convention; diagnostics are in its sidecar |
| Cross-city | `tables/cross_city_{pooled_vs_per_city,loco,transfer}.csv`, `predictions/cross_city_predictions.csv`, `models/cross_city/{per_city,loco}_<city>_h<h>.txt` | cross_city | `gap_vs_*_pct` = 100·(MAE_LOCO / MAE_ref − 1); `experiment` ∈ pooled / per_city / loco / transfer |
| Uncertainty | `tables/uncertainty_{overall,by_horizon,by_city,by_season,by_pollution_regime,held_out_city,aci_gamma_sensitivity,raw_and_cqr}.csv`, `predictions/uncertainty_predictions.csv`, `predictions/uncertainty_held_out_predictions.csv`, `models/uncertainty/` | uncertainty | Scored rows are test & `eval_mask` & `calibration_ready`; methods within a group share identical n. Calibration rows appear in the prediction files but are never scored. `low_n` flags n < 30 |
| Stress test | `tables/stress_2020_{sample_counts,point_results,uncertainty_results,pm25_distribution,period_shift,aci_alpha_summary}.csv`, `predictions/stress_2020_{point,uncertainty}_predictions.csv`, `models/stress_2020/` | stress | Periods are by target date (`all_2020`, `pre_lockdown`, `lockdown`); horizon = "all" marks pooled rows. The diagnostics only read saved files |
| Figures | `figures/leakage_*.png` (2), `cross_city_*.png` (2), `uncertainty_*.png` (6), `stress_2020_*.png` (6) | figures | Drawn only from the saved tables and predictions, at 200 dpi |
| Manifest | `outputs/result_manifest.csv` | `python run.py all` | path, bytes, sha256 |

## Regeneration

- **Everything:** `python run.py all` rebuilds every file from `data/raw/city_day.csv` in about 40 s on a laptop.
- **One stage:** `python run.py uncertainty` reruns a single stage.
- **Model reuse:** the uncertainty and stress stages reload saved quantile models only if every metadata field matches; otherwise they stop with an error.
- **Determinism:** models and CSVs are byte-identical across runs. Parquet files and sidecars differ only in their timestamps.
- **Git:** `outputs/predictions/*` is git-ignored, so add those files with `git add -f`.
