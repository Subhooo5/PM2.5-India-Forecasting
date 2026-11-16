from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src import data_audit as da
from src.experiments import (POINT_METHODS, assert_frames_identical, assert_no_calibration, forecast_frames,
                             input_exposure, read_predictions, read_without_calibration, score_methods)
from src.metrics import point_metrics
from src.models import baseline_predictions, fit_lgbm, lightgbm_params, predict_lgbm, save_model
from src.splits import LeakageError, main_ranges, random_split_unique

POINT_MODEL_FILE = "point_h{h}.txt"
FILES = {"C": "leakage_protocol_c.csv", "B": "leakage_protocol_b.csv", "A": "leakage_protocol_a.csv",
         "A_seeds": "leakage_protocol_a_seeds.csv", "baselines": "baseline_results.csv",
         "comparison": "leakage_comparison.csv"}
SEEDS = (42, 43, 44, 45, 46)
DATA_USE_2018 = ("2018 TARGET rows are never trained on or scored (calibration-reserved); 2018 observations "
                 "may enter causal input features (lags/rolling) of early-2019 forecast origins.")


def run_protocol_c(cfg: dict, dirs: dict | None = None) -> dict:
    dirs = dirs or da.output_dirs(cfg)
    train, test = forecast_frames(cfg)
    pc, params, (train_range, _, _) = cfg["leakage"]["protocol_C"], lightgbm_params(cfg), main_ranges(cfg)
    parts = []
    for h in pc["horizons"]:
        model, meta = fit_lgbm(train, pc["feature_set"], h, params, train_range, cfg)
        save_model(model, meta, dirs["models"] / POINT_MODEL_FILE.format(h=h))
        test_h = test.loc[test["horizon"] == h]
        lgb = predict_lgbm(model, test_h).assign(method="lightgbm")
        base = baseline_predictions(test_h)
        keys = lgb[["city", "origin_date", "target_date", "horizon", "y_true"]]
        for f in (lgb, keys.assign(y_pred=base["yhat_persistence"], method="persistence"),
                  keys.assign(y_pred=base["yhat_seasonal_naive"], method="seasonal_naive")):
            parts.append(f.assign(row_id=test_h.index, eval_mask=test_h["eval_mask"].to_numpy()))
    pred = pd.concat(parts, ignore_index=True).assign(protocol="C")
    pred["method"] = pd.Categorical(pred["method"], categories=list(POINT_METHODS))
    pred = pred.sort_values(["horizon", "method", "row_id"], kind="stable").reset_index(drop=True)
    results = score_methods(pred)
    cols = ["city", "origin_date", "target_date", "horizon", "y_true", "y_pred", "method", "protocol", "eval_mask"]
    da.write_csv(pred[cols], dirs["predictions"] / FILES["C"], cfg, "src.leakage_protocols.run_protocol_c",
                 "Protocol C: causal forecasting, train 2015-2017, test 2019.",
                 data_use_2018={"statement": DATA_USE_2018, "per_horizon": input_exposure(test, cfg, "main")})
    da.write_csv(results, dirs["tables"] / FILES["baselines"], cfg, "src.leakage_protocols.run_protocol_c",
                 "common_eval_mask = identical rows per horizon; own_full = each method's own rows.")
    return {"train": train, "test": test, "pred": pred, "results": results}


def same_day_frame(cfg: dict) -> pd.DataFrame:
    sd = read_without_calibration(da.resolve(cfg, "same_day_dataset"), cfg)
    if sd.duplicated(["city", "target_date"]).any():
        raise AssertionError("same-day dataset must be unique on (city, target_date)")
    if sd["y_target"].isna().any():
        raise ValueError("same-day rows must have an observed target")
    if set(sd["city"].astype(str)) != set(cfg["cities"]["core"]):
        raise ValueError("same-day dataset must contain exactly the core cities")
    return sd


def metrics_overall_and_city(pred: pd.DataFrame, **extra) -> pd.DataFrame:
    rows = [{"scope": "overall", "city": "all", **point_metrics(pred["y_true"], pred["y_pred"])}]
    rows += [{"scope": "city", "city": str(c), **point_metrics(g["y_true"], g["y_pred"])}
             for c, g in pred.groupby("city", observed=True, sort=True)]
    out = pd.DataFrame(rows)
    for k, v in extra.items():
        out.insert(0, k, v)
    return out


def _same_day_pred(model, rows: pd.DataFrame, protocol: str) -> pd.DataFrame:
    out = predict_lgbm(model, rows)[["city", "target_date", "y_true", "y_pred"]].copy()
    return out.assign(protocol=protocol, task="same_day_estimation", horizon=0, split="test")


def run_protocol_b(cfg: dict, dirs: dict | None = None) -> dict:
    dirs = dirs or da.output_dirs(cfg)
    sd = same_day_frame(cfg)
    (tr_lo, tr_hi), _, (te_lo, te_hi) = main_ranges(cfg)
    train = sd.loc[(sd["split_B"] == "train") & sd["target_date"].between(tr_lo, tr_hi)]
    test = sd.loc[(sd["split_B"] == "test") & sd["target_date"].between(te_lo, te_hi)]
    if len(train) + len(test) != int(sd["split_B"].isin(["train", "test"]).sum()):
        raise AssertionError("split_B labels disagree with the date ranges")
    fs, params = cfg["leakage"]["protocol_B"]["feature_set"], lightgbm_params(cfg)
    model, meta = fit_lgbm(train, fs, None, params, (tr_lo, tr_hi), cfg)
    pred = _same_day_pred(model, test, "B")
    metrics = metrics_overall_and_city(pred, protocol="B", task="same_day_estimation")
    da.write_csv(pred, dirs["predictions"] / FILES["B"], cfg, "src.leakage_protocols.run_protocol_b",
                 "Protocol B: same-day estimation (not forecasting), train 2015-2017, test 2019.",
                 task="same_day_estimation", feature_set=fs, features=meta["features"], n_train=len(train),
                 n_test=len(test), metrics=metrics.to_dict("records"))
    return {"train": train, "test": test, "pred": pred, "metrics": metrics, "model": model, "meta": meta,
            "params": params}


def protocol_a_split(pool: pd.DataFrame, cfg: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pa = cfg["leakage"]["protocol_A"]
    labels = random_split_unique(pool[["city", "target_date"]], pa["test_frac"], seed, pa["stratify_by"])
    train, test = pool.loc[labels == "train"], pool.loc[labels == "test"]
    keys = [pd.MultiIndex.from_frame(d[["city", "target_date"]].astype({"city": str})) for d in (train, test, pool)]
    if not keys[0].intersection(keys[1]).empty:
        raise LeakageError("a (city, date) appears in both A partitions")
    if len(train) + len(test) != len(pool) or not keys[0].union(keys[1]).equals(keys[2].sort_values()):
        raise AssertionError("A train union test must equal the pool exactly")
    return train, test


def split_composition(a_train, a_test, b_train, b_test) -> dict:
    per_year = lambda t: {int(cast(int, k)): int(v) for k, v in t["target_date"].dt.year.value_counts().sort_index().items()}
    size = lambda t: t.groupby("city", observed=True).size()
    return {"A": {"n_train": len(a_train), "n_test": len(a_test),
                  "test_share": round(len(a_test) / (len(a_train) + len(a_test)), 4),
                  "test_rows_per_year": per_year(a_test),
                  "test_share_per_city": {str(c): round(float(v), 4)
                                          for c, v in (size(a_test) / (size(a_train) + size(a_test))).items()}},
            "B": {"n_train": len(b_train), "n_test": len(b_test),
                  "test_share": round(len(b_test) / (len(b_train) + len(b_test)), 4),
                  "test_rows_per_year": per_year(b_test)},
            "note": "A's test days are interleaved with its training days; B's test is the whole of 2019."}


def protocol_a_pool(sd: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pa = cfg["leakage"]["protocol_A"]
    pool = sd.loc[sd["split_B"].isin(["train", "test"])]
    if not pool["target_date"].dt.year.isin(set(pa["pool_target_years"])).all():
        raise LeakageError("protocol A pool contains a year outside the pool years")
    assert_no_calibration(pool["target_date"], cfg, "protocol A pool")
    if pa["feature_set"] != cfg["leakage"]["protocol_B"]["feature_set"]:
        raise AssertionError("A and B must use the same feature set")
    return pool


def seed_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    summary = [{"seed": stat, "subset": subset, "n_train": np.nan, "n_test": np.nan,
                **g[["n", "mae", "rmse", "r2"]].agg(fn).to_dict()}
               for subset, g in per_seed.groupby("subset", sort=False) for stat, fn in (("mean", "mean"), ("sd", "std"))]
    return pd.concat([per_seed, pd.DataFrame(summary)], ignore_index=True)


def run_protocol_a(cfg: dict, dirs: dict | None = None, seeds=SEEDS) -> dict:
    dirs, pa = dirs or da.output_dirs(cfg), cfg["leakage"]["protocol_A"]
    sd = same_day_frame(cfg)
    pool = protocol_a_pool(sd, cfg)
    params, ((tr_lo, _), _, (te_lo, te_hi)) = lightgbm_params(cfg), main_ranges(cfg)
    seed_rows, primary, seed_preds = [], None, {}
    for seed in seeds:
        train, test = protocol_a_split(pool, cfg, seed)
        model, meta = fit_lgbm(train, pa["feature_set"], None, params, (tr_lo, te_hi), cfg)
        pred = _same_day_pred(model, test, "A").assign(seed=seed)
        seed_preds[seed] = pred
        for subset, rows in (("all_test", pred), ("test_2019_only", pred[pred["target_date"].between(te_lo, te_hi)])):
            seed_rows.append({"seed": str(seed), "subset": subset, "n_train": len(train), "n_test": len(test),
                              **point_metrics(rows["y_true"], rows["y_pred"])})
        if seed == 42:
            primary = {"train": train, "test": test, "pred": pred, "meta": meta}
    if primary is None:
        raise ValueError("seed 42 must be among the seeds")
    seeds_table = seed_table(pd.DataFrame(seed_rows))
    comp = split_composition(primary["train"], primary["test"], sd.loc[sd["split_B"] == "train"],
                             sd.loc[sd["split_B"] == "test"])
    da.write_csv(primary["pred"], dirs["predictions"] / FILES["A"], cfg, "src.leakage_protocols.run_protocol_a",
                 "Protocol A: same-day estimation, random 80/20 split of unique city-dates (seed 42).",
                 task="same_day_estimation", feature_set=pa["feature_set"], features=primary["meta"]["features"],
                 train_target_years=list(pa["pool_target_years"]), split_composition=comp)
    da.write_csv(seeds_table, dirs["tables"] / FILES["A_seeds"], cfg, "src.leakage_protocols.run_protocol_a",
                 "Seeds vary only the A split; sd = sample standard deviation.", seeds=list(seeds))
    return {**primary, "pool": pool, "seeds_table": seeds_table, "composition": comp, "params": params,
            "seed_preds": seed_preds}


def load_saved(cfg: dict, dirs: dict | None = None) -> dict:
    d = dirs or da.output_dirs(cfg)
    return {**{k: read_predictions(d["predictions"] / FILES[k], cfg) for k in ("A", "B", "C")},
            "A_seeds": pd.read_csv(d["tables"] / FILES["A_seeds"]),
            "A_meta": json.loads((d["predictions"] / FILES["A"].replace(".csv", ".meta.json")).read_text())}


def regenerate_protocol_a(cfg: dict, saved: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        dirs = {k: Path(tmp) / k for k in ("models", "predictions", "tables")}
        res = run_protocol_a(cfg, dirs)
    assert_frames_identical(res["seeds_table"], saved["A_seeds"], ["seed", "subset"],
                            ["seed", "subset", "n_train", "n_test", "mae", "rmse", "r2", "n"],
                            "regenerated Protocol A seed table")
    a42 = res["seed_preds"][42]
    if list(a42.columns) != list(saved["A"].columns):
        raise AssertionError("Protocol A seed-42 columns differ")
    assert_frames_identical(a42, saved["A"], ["city", "target_date"], list(a42.columns),
                            "regenerated Protocol A seed-42 predictions")
    return res["seed_preds"]
