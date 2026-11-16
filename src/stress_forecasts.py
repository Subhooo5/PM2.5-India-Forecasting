from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src import data_audit as da
from src import intervals as iv
from src.experiments import POINT_METHODS, input_exposure, score_methods
from src.features import usable_samples_table
from src.models import baseline_predictions, fit_lgbm, lightgbm_params, predict_lgbm, save_model
from src.splits import label_by_target_date, split_ranges

PROTOCOL = "stress_2020"
POINT_MODEL_FILE = "stress_2020/point_h{h}.txt"
QUANTILE_STEM = "stress_2020/quantile"
FILES = {"counts": "stress_2020_sample_counts.csv", "point_pred": "stress_2020_point_predictions.csv",
         "point_results": "stress_2020_point_results.csv", "unc_pred": "stress_2020_uncertainty_predictions.csv",
         "unc_results": "stress_2020_uncertainty_results.csv", "distribution": "stress_2020_pm25_distribution.csv",
         "shift": "stress_2020_period_shift.csv", "alpha": "stress_2020_aci_alpha_summary.csv"}
USABLE_SAMPLES = "usable_samples_by_city_split_horizon.csv"
PERIODS = ("all_2020", "pre_lockdown", "lockdown")
MIN_N = 30
NOTES = "2020 distribution-shift stress test (not causal; partial year; periods by target date); see docs/design.md."


def lockdown(cfg: dict) -> pd.Timestamp:
    return pd.Timestamp(cfg["splits"]["stress"]["lockdown_start"])


def stress_frames(cfg: dict) -> dict[str, pd.DataFrame]:
    st, path = cfg["stress_test"], da.resolve(cfg, "forecast_dataset")
    sha, ranges, lock = da.sha256_file(path), split_ranges(cfg, "stress"), lockdown(cfg)
    cities, hs = cfg["cities"][st["cities"]], st["horizons"]
    out = {}
    for s, (lo, hi) in ranges.items():
        df = pd.read_parquet(path, filters=[("target_date", ">=", lo), ("target_date", "<=", hi)])
        df = df[df["city"].astype(str).isin(cities) & df["horizon"].isin(hs)]
        period = np.where(df["target_date"] < lock, "pre_lockdown", "lockdown") if s == "test" else np.full(len(df), "")
        for bad, msg in (
                (~df["target_date"].between(lo, hi), f"target dates outside {lo.date()}..{hi.date()}"),
                (df[st["split_column"]].astype(str) != s, f"{st['split_column']} label != {s!r}"),
                (label_by_target_date(df["target_date"], cfg, "stress") != s, "recomputed label disagrees"),
                (df["y_target"].isna(), "unobserved (never imputed) targets"),
                (df.duplicated(["city", "origin_date", "horizon"]), "duplicated (city, origin_date, horizon) keys"),
                (df["target_date"] != df["origin_date"] + pd.to_timedelta(df["horizon"].astype(int), unit="D"),
                 "target_date != origin_date + horizon days"),
                (df[st["period_column"]].astype(str) != period, "stress_period disagrees with target date")):
            if bad.any():
                raise ValueError(f"stress {s} frame: {int(bad.sum())} rows with {msg}")
        missing = {(c, h) for c in cities for h in hs} - set(zip(df["city"].astype(str), df["horizon"].astype(int)))
        if missing:
            raise ValueError(f"stress {s} frame: missing (city, horizon) {sorted(missing)}")
        out[s] = df
    days = {"pre_lockdown": (lock - ranges["test"][0]).days, "lockdown": (ranges["test"][1] - lock).days + 1}
    if days != st["expected_test_days"] or da.sha256_file(path) != sha:
        raise ValueError(f"test period lengths {days} != {st['expected_test_days']} or parquet changed")
    return out


def audit(cfg: dict, data: dict, tables_dir: Path) -> tuple[pd.DataFrame, dict]:
    st, lock = cfg["stress_test"], lockdown(cfg)
    allf = pd.concat(data.values())
    keys = ["city", "horizon", "split", "stress_period"]
    saved = pd.read_csv(tables_dir / USABLE_SAMPLES, keep_default_na=False).query("scheme == 'stress'")
    saved = saved[saved["city"].isin(cfg["cities"][st["cities"]])]
    a, b = (d.astype({"city": str, "horizon": int, "split": str})[keys + ["n_samples", "n_eval_mask"]]
            .sort_values(keys).reset_index(drop=True) for d in (usable_samples_table(allf).query("scheme == 'stress'"), saved))
    if not a.equals(b):
        raise AssertionError("stress sample counts disagree with the usable-samples table")
    days = allf.groupby(["city", "horizon", st["split_column"], st["period_column"]], observed=True)["target_date"].nunique()
    t = a.rename(columns={"n_samples": "n_rows"}).assign(
        n_target_days=[days[k] for k in a[keys].itertuples(index=False)], low_n=lambda d: d["n_eval_mask"] < MIN_N)
    if (t.loc[t["split"] == "test", "n_eval_mask"] == 0).any():
        raise ValueError("a core city has zero test eval_mask rows in a period or horizon")
    test = data["test"]
    by_origin = {int(cast(int, h)): int(((g["stress_period"] == "lockdown") & (g["origin_date"] < lock)).sum())
                 for h, g in test.groupby("horizon")}
    return t, {"data_use_2019": input_exposure(test, cfg, "stress"), "lockdown_label_origin_before_lockdown": by_origin}


def run_point(cfg: dict, dirs: dict | None = None) -> dict:
    dirs, st = dirs or da.output_dirs(cfg), cfg["stress_test"]
    data = stress_frames(cfg)
    counts, data_use = audit(cfg, data, dirs["tables"])
    shash = da.stress_config_hash(cfg)
    da.write_csv(counts, dirs["tables"] / FILES["counts"], cfg, "src.stress_forecasts.audit", NOTES, stress_config_hash=shash,
                 **data_use)
    params, train_range, parts = lightgbm_params(cfg), split_ranges(cfg, "stress")["train"], []
    for h in st["horizons"]:
        model, meta = fit_lgbm(data["train"], st["feature_set"], h, params, train_range, cfg)
        save_model(model, {**meta, "stress_config_hash": shash}, dirs["models"] / POINT_MODEL_FILE.format(h=h))
        test_h = data["test"][data["test"]["horizon"] == h]
        lgb = predict_lgbm(model, test_h).assign(method="lightgbm")
        base = baseline_predictions(test_h)
        keys = lgb[["city", "origin_date", "target_date", "horizon", "y_true"]]
        for f in (lgb, keys.assign(y_pred=base["yhat_persistence"], method="persistence"),
                  keys.assign(y_pred=base["yhat_seasonal_naive"], method="seasonal_naive")):
            parts.append(f.assign(row_id=test_h.index, eval_mask=test_h["eval_mask"].to_numpy(),
                                  stress_period=test_h["stress_period"].astype(str).to_numpy()))
    pred = pd.concat(parts, ignore_index=True).assign(protocol=PROTOCOL)
    pred["method"] = pd.Categorical(pred["method"], categories=list(POINT_METHODS))
    pred = pred.sort_values(["horizon", "method", "row_id"], kind="stable").reset_index(drop=True)
    res = pd.concat([score_methods(pred if p == "all_2020" else pred[pred["stress_period"] == p])
                     .query("mask_type == 'common_eval_mask'").assign(period=p) for p in PERIODS], ignore_index=True)
    res["low_n"] = res["n"] < MIN_N
    res = res[["period", "method", "scope", "city", "horizon", "n", "mae", "rmse", "r2", "skill_mae", "skill_rmse",
               "note", "mask_type", "protocol", "low_n"]]
    cols = ["city", "origin_date", "target_date", "horizon", "y_true", "y_pred", "method", "protocol", "eval_mask",
            "stress_period"]
    da.write_csv(pred[cols], dirs["predictions"] / FILES["point_pred"], cfg, "src.stress_forecasts.run_point", NOTES,
                 stress_config_hash=shash, **data_use)
    da.write_csv(res, dirs["tables"] / FILES["point_results"], cfg, "src.stress_forecasts.run_point", NOTES,
                 stress_config_hash=shash)
    return {"frames": data, "counts": counts, "data_use": data_use, "pred": pred, "results": res}


def interval_checks(pred: pd.DataFrame, horizons) -> dict:
    test = pred[pred["split"] == "test"]
    ready = test[test["calibration_ready"]]
    return {"row_alignment_identical": True,
            "by_split": {m: iv.interval_checks(ready if m == "cqr_aci" else pred, lo, hi)
                         for m, (lo, hi) in iv.METHODS.items()},
            "test_by_period": {m: iv.interval_checks(ready if m == "cqr_aci" else test, lo, hi,
                                                     ["stress_period", "horizon"]) for m, (lo, hi) in iv.METHODS.items()},
            "n_test_rows_not_in_eval_mask": {int(cast(int, h)): int((~g["eval_mask"]).sum())
                                             for h, g in test.groupby("horizon")},
            "n_test_rows_not_calibration_ready": iv.not_ready(pred, horizons)}


def period_results(periods: dict, cfg: dict) -> pd.DataFrame:
    res = pd.concat([iv.grouped_metrics(sub, iv.METHODS, keys, cfg).assign(period=p, scope=scope)
                     for p, sub in periods.items()
                     for scope, keys in (("overall", []), ("horizon", ["horizon"]), ("city_horizon", ["city", "horizon"]))],
                    ignore_index=True)
    res = res.assign(city=res["city"].fillna("all"), horizon=res["horizon"].astype("Int64").astype(object).fillna("all"))
    return res[["period", "scope", "city", "horizon", "method", "n", "nominal", "picp", "coverage_gap", "mpiw", "winkler",
                "low_n"]]


def run_uncertainty(cfg: dict, dirs: dict | None = None) -> dict:
    dirs, st, u = dirs or da.output_dirs(cfg), cfg["stress_test"], cfg["uncertainty"]
    for k in ("horizons", "cities", "feature_set"):
        if st[k] != u[k]:
            raise ValueError(f"stress_test.{k} must equal uncertainty.{k}")
    data, ranges, lock, core = stress_frames(cfg), split_ranges(cfg, "stress"), lockdown(cfg), cfg["cities"][st["cities"]]
    shash = da.stress_config_hash(cfg)
    models = iv.quantile_models(cfg, data["train"], dirs["models"], QUANTILE_STEM,
                                 da.sha256_file(da.resolve(cfg, "raw_city_day")), ranges["train"], stress_config_hash=shash)
    pred = iv.predict_intervals(models, data, cfg)
    period = pd.concat([data[s].loc[data[s]["horizon"] == h, "stress_period"].astype(str)
                        for s in ("calibration", "test") for h in sorted(st["horizons"])]).to_numpy()
    if not (period == np.where(pred["split"] != "test", "", np.where(pred["target_date"] < lock, "pre_lockdown",
                                                                      "lockdown"))).all():
        raise ValueError("stress_period disagrees with the target-date rule")
    pred.insert(5, "stress_period", period)
    pred = iv.mark_calibration_ready(pred, ranges["calibration"][1])
    scores = iv.calibration_scores(pred, cfg, core, ranges["calibration"])
    q_hat = iv.q_hat_by_horizon(scores, cfg)
    pred = iv.with_cqr(pred, q_hat, "cqr")
    pred = pred.join(iv.aci_columns(pred, scores, cfg, u["aci"]["gamma"]))
    test = pred[pred["split"] == "test"]
    by_period = {p: test[test["stress_period"] == p] for p in PERIODS[1:]}
    checks = interval_checks(pred, st["horizons"])
    summary = {"by_horizon": iv.aci_summary(pred, cfg),
               "by_period": {p: iv.aci_summary(g, cfg) for p, g in by_period.items()}}
    res = period_results({"all_2020": test, **by_period}, cfg)
    meta = {"stress_config_hash": shash, "uncertainty_config_hash": da.uncertainty_config_hash(cfg), "q_hat": q_hat}
    da.write_csv(pred, dirs["predictions"] / FILES["unc_pred"], cfg, "src.stress_forecasts.run_uncertainty", NOTES, **meta,
                 checks=checks, aci_summary=summary)
    da.write_csv(res, dirs["tables"] / FILES["unc_results"], cfg, "src.stress_forecasts.run_uncertainty", NOTES, **meta)
    return {"frames": data, "pred": pred, "scores": scores, "q_hat": q_hat, "results": res, "checks": checks,
            "aci_summary": summary}
