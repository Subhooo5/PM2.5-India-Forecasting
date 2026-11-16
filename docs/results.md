# Results (neutral summary)

**Read when** you need a number for the paper or want to check that a rerun still matches.
Every value below is copied from `outputs/tables/`. Unless stated otherwise:
- the test year is 2019;
- the cities are the five core cities;
- error units are µg/m³;
- scoring uses `eval_mask` rows.

Interpretation belongs in the paper, under the wording rules in [design.md](design.md#7-wording-rules).

## Leakage audit

**Protocol C (genuine forecast) vs baselines** (`baseline_results.csv`; identical rows, n = 1,825 per h):

| h | LightGBM MAE / RMSE / R² | Persistence MAE / RMSE / R² | Skill MAE / RMSE |
|---|---|---|---|
| 1 | 15.308 / 26.392 / 0.798 | 15.161 / 28.179 / 0.770 | −0.010 / 0.063 |
| 3 | 22.762 / 38.334 / 0.574 | 23.307 / 41.517 / 0.501 | 0.023 / 0.077 |
| 7 | 24.611 / 40.736 / 0.519 | 26.894 / 47.944 / 0.334 | 0.085 / 0.150 |

Seasonal naive has MAE 26.894 at every h, by definition.

**Same-day estimation:**
- **B:** MAE 10.980, RMSE 15.861, R² 0.927 (n = 1,825).
- **A, all test rows:** MAE 12.783 ± 0.329 and RMSE 24.118 ± 2.321 over seeds 42–46 (n = 1,369).
- **A, 2019-dated test rows only:** MAE 8.625 ± 0.272.

**Relative inflation** (`leakage_comparison.csv`) is 100·(1 − error_left / error_right), where left is the optimistic protocol:

| Comparison | MAE % | RMSE % |
|---|---|---|
| A vs B, all A rows vs B (5 seeds) | −16.4 ± 3.0 | −52.1 ± 14.6 |
| A vs B, A 2019 rows vs B | +21.5 ± 2.5 | +18.1 ± 3.4 |
| A vs B, matched 2019 city-dates (n = 356.8 ± 16.6) | +22.2 ± 2.7 | +16.0 ± 3.8 |
| B vs C, h = 1 / 3 / 7 (two different tasks) | +28.3 / +51.8 / +55.4 | +39.9 / +58.6 / +61.1 |

Diagnostics are in the `leakage_comparison` sidecar:
- A's errors concentrate in its 2015–2017 rows.
- Chennai has the lowest C h = 1 R² (0.241) and the most 2019 spikes above mean + 3 sd.

## Cross-city

Overall MAE (`cross_city_pooled_vs_per_city.csv`, `cross_city_loco.csv`):

| | h = 1 | h = 3 | h = 7 |
|---|---|---|---|
| Pooled | 15.308 | 22.762 | 24.611 |
| Per-city | 17.069 | 25.566 | 27.136 |
| LOCO | 15.829 | 23.493 | 25.005 |
| LOCO vs pooled | +3.4% | +3.2% | +1.6% |
| LOCO vs per-city | −7.3% | −8.1% | −7.9% |

- **Largest LOCO gap vs pooled:** Bengaluru, at +12.0% / +37.9% / +22.3%.
- **Hyderabad:** its LOCO gap is negative at h = 3 and 7 (−7.3% / −9.5%), and its per-city model is much worse than pooled.

External-city transfer, pooled models with local history (`cross_city_transfer.csv`), MAE at h = 1 / 3 / 7:
- **Kolkata:** 14.686 / 22.739 / 23.226 (n = 365).
- **Mumbai:** 7.076 / 11.349 / 13.137 (n = 365).
- **Visakhapatnam (optional):** 12.237 / 18.183 / 20.866 (n = 337 / 336 / 341).

## Uncertainty (nominal 0.90)

These use calibration-ready rows, n = 1,825 / 1,815 / 1,795 (5,435 overall). Sources: `uncertainty_overall.csv` and `uncertainty_by_horizon.csv`.

| Method | PICP all | PICP h = 1 / 3 / 7 | MPIW all | Winkler all |
|---|---|---|---|---|
| Raw quantile | 0.806 | 0.843 / 0.786 / 0.788 | 63.14 | 119.97 |
| CQR | 0.877 | 0.892 / 0.871 / 0.870 | 71.35 | 114.92 |
| CQR + ACI | 0.888 | 0.899 / 0.885 / 0.880 | 78.24 | 110.46 |

**CQR and ACI settings:**
- CQR q_hat at h = 1 / 3 / 7: 2.456 / 4.668 / 5.199 (n_cal 1,798 / 1,799 / 1,799).
- ACI mean alpha_issue: 0.097 / 0.092 / 0.095. The gamma-sensitivity PICP range at h = 7 is 0.867–0.884.

**Strata** (empirical only, no guarantee), CQR + ACI PICP at h = 1 / 3 / 7:
- Top-10% days (per-city 2015–2017 p90; n = 102 / 97 / 91): 0.814 / 0.649 / 0.451.
- Normal days: 0.904 / 0.899 / 0.903.
- Post-monsoon: 0.872 / 0.823 / 0.770.

**Held-out city** (`uncertainty_held_out_city.csv`), mean PICP over cells:
- raw LOCO 0.788;
- CQR source-calibrated 0.876;
- CQR local-calibrated 0.878;
- ACI source-calibrated 0.888.

## 2020 stress test (distribution shift, not causal)

Point MAE, LightGBM vs persistence at h = 1 / 3 / 7 (`stress_2020_point_results.csv`):

| Period (n per h) | LightGBM | Persistence |
|---|---|---|
| All 2020 (915) | 11.40 / 16.50 / 18.38 | 11.20 / 17.55 / 18.93 |
| Pre-lockdown (420) | 15.25 / 22.93 / 25.70 | 15.17 / 25.44 / 26.67 |
| Lockdown (495) | 8.14 / 11.05 / 12.17 | 7.83 / 10.85 / 12.35 |

Intervals (`stress_2020_uncertainty_results.csv`; q_hat from 2019 = 3.076 / 6.249 / 7.966), PICP · MPIW:

| Period (n) | Raw quantile | CQR | CQR + ACI |
|---|---|---|---|
| All 2020 (2,705) | 0.796 · 48.41 | 0.919 · 59.89 | 0.908 · 61.92 |
| Pre-lockdown (1,220) | 0.794 · 64.14 | 0.895 · 75.55 | 0.907 · 81.07 |
| Lockdown (1,485) | 0.797 · 35.49 | 0.938 · 47.02 | 0.908 · 46.19 |

Mean observed PM2.5 is lower in the lockdown window than in the pre-lockdown window in every core city (`stress_2020_pm25_distribution.csv`). The two windows also differ in season.
