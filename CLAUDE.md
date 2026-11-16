# CLAUDE.md: PM2.5 India forecasting

## Hard constraints (ask the project owner before breaking any of these)
- **No git writes**: no add, commit, stash, reset, checkout, restore, rebase, merge, push, mv or rm. Read-only git only. The owner commits manually; `outputs/predictions/*` is git-ignored and needs `git add -f`.
- **Never touch `data/raw/`.** `city_day.csv` is immutable (sha256 `0d84b21c…1847`, Kaggle v12).
- **Ask before changing any of these:**
  - any value in `configs/experiments.yaml`;
  - any saved model, prediction, table or q_hat;
  - any design rule in `docs/design.md`.
- **No retraining or re-tuning** unless asked. `run.py tuning` must reproduce c2; the uncertainty and stress stages reload saved quantile models and stop on any metadata mismatch.
- **Splits by TARGET date only.** 2018 (2019 in the stress scheme) targets are calibration-only: never trained on, tuned on or scored.
- **`eval_mask` rows only for scoring.** Every model and baseline is scored on identical rows, and interval metrics also require `calibration_ready`, i.e. origin_date ≥ the last calibration day.
- **Forbidden forecast inputs:** AQI, AQI_Bucket and the co-pollutants are never inputs to a forecast. Targets are never filled; inputs are forward-filled for at most 2 days.
- **Random splits only in protocol A,** and only on the unique (city, date) table.
- **Wording follows `docs/design.md` §7:**
  - LOCO and transfer are never "zero-shot";
  - no causal language for 2020;
  - no conditional or per-regime coverage guarantees;
  - B vs C are two different tasks.
- **No comments in code:** no descriptive comments, banners or TODOs. At most a one-line docstring where a function's purpose isn't obvious.
- **Python 3.11** in `.venv`, local macOS only. Verification is executable (pytest), never narrative.

## What and why
A survey plus a small benchmark: "Leakage-free, uncertainty-aware PM2.5 forecasting for Indian cities". One reproducible pipeline runs:
- **Leakage audit:** protocols A, B and C;
- **Cross-city:** pooled vs per-city vs leave-one-city-out, plus external transfer;
- **Uncertainty:** quantile LightGBM vs CQR vs CQR + ACI;
- **2020 distribution-shift stress test.**

Core cities: Delhi, Lucknow, Chennai, Hyderabad, Bengaluru. Transfer cities: Kolkata, Mumbai, and optionally Visakhapatnam. No new algorithm is proposed.

## Commands
```bash
source .venv/bin/activate
python run.py all            # every stage in order, then outputs/result_manifest.csv (~40 s)
python run.py <stage> ...    # audit preprocess features tuning leakage cross_city uncertainty stress figures
pytest -q -W always          # 345 tests (~30 s), 0 warnings expected
shasum -a 256 data/raw/city_day.csv
```
- **Stage order:** each stage reads what earlier stages wrote, so rerun downstream stages after changing an upstream one.
- **`figures`:** reads saved outputs only.

## Layout
```
run.py                    STAGES dict -> module.run / run_* functions
configs/experiments.yaml  all paths, splits, features, lightgbm (base, grid, selected), tuning,
                          leakage, cross_city, uncertainty, stress_test
src/data_audit.py         load_config, hashes (config / uncertainty / stress), write_csv + .meta.json
                          sidecars, output_dirs, verify_raw, completeness audit
src/preprocessing.py      city roles, daily series, ffill_inputs
src/features.py           causal features, build_dataset, same-day table, sample tables, run_preprocess
src/splits.py             target-date labels, random_split_unique, tuning guard + expanding folds
src/models.py             baselines, lightgbm_params / quantile_params, check_feature_set guards,
                          fit_lgbm / predict_lgbm / save_model / load_model
src/metrics.py            point + interval metrics, safe ratios
src/conformal.py          conformity scores, conformal_quantile, cqr_interval, aci_stream
src/experiments.py        readers that refuse calibration rows, score_methods, input_exposure, tuning
src/leakage_protocols.py  protocols C, B, A; regenerate_protocol_a
src/leakage.py            A-vs-B / B-vs-C comparison, diagnostics, figures
src/cross_city.py         pooled / per-city / LOCO / transfer
src/intervals.py          quantile_models (metadata-checked reuse), predict_intervals (alignment checks),
                          calibration-ready rule, CQR, ACI columns, grouped_metrics
src/uncertainty.py        strata, held-out cities, tables, figures
src/stress_forecasts.py   2020 frames + audit, run_point, run_uncertainty
src/stress.py             read-only diagnostics, figures, write_manifest
src/figures.py            matplotlib helpers (200 dpi)
tests/test_<module>.py    one file per module; conftest has session fixtures (cfg, raw, ds, dirs, unc_run)
docs/                     design.md, results.md, outputs.md
```

## Verification loop
1. **Code change:** run `pytest -q -W always`. The suite reruns every experiment into tmp dirs and compares the result byte for byte with `outputs/`, with spies that fail if a saved model is refit.
2. **Change that should leave results unchanged:**
   - run `python run.py all` in a scratch copy of the repo (outputs and processed data excluded);
   - compare tables and predictions with the current `outputs/` (counts exact, floats 1e-9);
   - stop and report any difference.
3. **Figure change:** view the PNG.
4. **After a full run:** `outputs/result_manifest.csv` records the bytes and sha256 of every output.

## Pointers
- [docs/design.md](docs/design.md): frozen design, wording rules, decisions log, open decisions.
- [docs/results.md](docs/results.md): neutral headline numbers from the final tables.
- [docs/outputs.md](docs/outputs.md): every output file group and how to read it.
- [README.md](README.md): public overview and quick start.
