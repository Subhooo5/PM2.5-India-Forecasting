from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src import data_audit as da
from src import figures
from src.experiments import assert_frames_identical, forecast_frames, read_predictions, read_without_calibration
from src.leakage_protocols import FILES as LEAKAGE_FILES
from src.leakage_protocols import POINT_MODEL_FILE
from src.metrics import point_metrics, relative_change_pct, skill_score
from src.models import fit_lgbm, lightgbm_params, load_model, predict_lgbm, save_model
from src.splits import LeakageError, main_ranges

FILES = {"pooled": "cross_city_pooled_vs_per_city.csv", "predictions": "cross_city_predictions.csv",
         "loco": "cross_city_loco.csv", "transfer": "cross_city_transfer.csv"}
MODELS_SUBDIR = "cross_city"
PRED_COLUMNS = ["city", "origin_date", "target_date", "horizon", "y_true", "y_pred", "experiment", "train_cities",
                "held_out_city", "eval_mask"]
METRIC_COLUMNS = ["experiment", "scope", "city", "horizon", "n", "mae", "rmse", "r2", "skill_mae", "skill_rmse",
                  "train_cities"]
LOCO_COLUMNS = ["held_out_city", "scope", "horizon", "n", "mae", "rmse", "r2", "skill_mae", "skill_rmse",
                "mae_pooled", "mae_per_city", "gap_vs_pooled_pct", "gap_vs_per_city_pct", "train_cities"]
TRANSFER_COLUMNS = ["city", "city_role", "optional", "label", "horizon", "n", "mae", "rmse", "r2", "skill_mae",
                    "skill_rmse", "train_cities"]
LABEL_TRANSFER = "external-city transfer with local history (no target-city observations in training)"
EXPERIMENT_ORDER = {"pooled": 0, "per_city": 1, "loco": 2, "transfer": 3}


def prediction_rows(test_h: pd.DataFrame, pred: pd.DataFrame, experiment: str, train_cities,
                    held_out: str = "") -> pd.DataFrame:
    out = pred[["city", "origin_date", "target_date", "horizon", "y_true", "y_pred"]].copy()
    return out.assign(city=out["city"].astype(str), experiment=experiment, train_cities="|".join(sorted(train_cities)),
                      held_out_city=held_out, eval_mask=test_h["eval_mask"].to_numpy(),
                      y_persistence=test_h["pm25_origin"].to_numpy())


def experiment_metrics(pred: pd.DataFrame, experiment: str, scopes=("overall", "city")) -> pd.DataFrame:
    rows, ev = [], pred.loc[pred["eval_mask"]]
    for scope in scopes:
        keys = ["horizon"] + (["city"] if scope == "city" else [])
        for gkey, g in ev.groupby(keys, sort=True):
            rec = dict(zip(keys, gkey if isinstance(gkey, tuple) else (gkey,)))
            m, r = point_metrics(g["y_true"], g["y_pred"]), point_metrics(g["y_true"], g["y_persistence"])
            rows.append({"experiment": experiment, "scope": scope, "city": rec.get("city", "all"),
                         "horizon": int(cast(int, rec["horizon"])), **m, "skill_mae": skill_score(m["mae"], r["mae"]),
                         "skill_rmse": skill_score(m["rmse"], r["rmse"]),
                         "train_cities": "|".join(sorted(g["train_cities"].unique()))})
    return pd.DataFrame(rows)[METRIC_COLUMNS]


def pooled_predictions(cfg: dict, test: pd.DataFrame, dirs: dict) -> dict:
    saved = read_predictions(dirs["predictions"] / LEAKAGE_FILES["C"], cfg)
    out = {}
    for h in cfg["leakage"]["protocol_C"]["horizons"]:
        booster, meta = load_model(dirs["models"] / POINT_MODEL_FILE.format(h=h))
        test_h = test.loc[test["horizon"] == h]
        pred = predict_lgbm(booster, test_h)
        if len(pred) != len(test_h):
            raise AssertionError(f"pooled h={h}: prediction / test length mismatch")
        assert_frames_identical(pred.assign(eval_mask=test_h["eval_mask"].to_numpy()),
                                saved[(saved["method"] == "lightgbm") & (saved["horizon"] == h)],
                                ["city", "origin_date", "target_date"],
                                ["city", "origin_date", "target_date", "horizon", "y_true", "y_pred", "eval_mask"],
                                f"pooled h={h} vs saved Protocol C")
        out[h] = (meta, test_h, pred)
    return out


def write_predictions(new: pd.DataFrame, cfg: dict, dirs: dict, replace) -> Path:
    path = dirs["predictions"] / FILES["predictions"]
    keep = pd.DataFrame(columns=PRED_COLUMNS)
    if path.exists():
        old = pd.read_csv(path, parse_dates=["origin_date", "target_date"], keep_default_na=False, na_values=[""])
        keep = old.loc[~old["experiment"].isin(replace), PRED_COLUMNS]
    out = pd.concat([f for f in (keep, new[PRED_COLUMNS]) if len(f)], ignore_index=True)
    out = out.sort_values(["experiment", "horizon", "city", "target_date"], kind="stable",
                          key=lambda s: s.map(EXPERIMENT_ORDER) if s.name == "experiment" else s).reset_index(drop=True)
    out["held_out_city"] = out["held_out_city"].fillna("")
    _, (cal_lo, cal_hi), _ = main_ranges(cfg)
    if pd.to_datetime(out["target_date"]).between(cal_lo, cal_hi).any():
        raise LeakageError("cross-city predictions contain calibration-period target rows")
    return da.write_csv(out, path, cfg, "src.cross_city.write_predictions",
                        "Experiments: pooled / per_city / loco / transfer; 2019 targets.")


def run_pooled_per_city(cfg: dict, dirs: dict | None = None) -> dict:
    dirs = dirs or da.output_dirs(cfg)
    train, test = forecast_frames(cfg)
    fs, params, (train_range, _, _) = cfg["cross_city"]["feature_set"], lightgbm_params(cfg), main_ranges(cfg)
    parts = [prediction_rows(test_h, pred, "pooled", meta["train_cities"])
             for meta, test_h, pred in pooled_predictions(cfg, test, dirs).values()]
    for city in cfg["cities"]["core"]:
        for h in cfg["leakage"]["protocol_C"]["horizons"]:
            model, meta = fit_lgbm(train.loc[train["city"].astype(str) == city], fs, h, params, train_range, cfg)
            if meta["train_cities"] != [city]:
                raise LeakageError(f"per-city model for {city} saw {meta['train_cities']}")
            save_model(model, {**meta, "experiment": "per_city", "held_out_city": None},
                       dirs["models"] / MODELS_SUBDIR / f"per_city_{city}_h{h}.txt")
            test_ch = test.loc[(test["horizon"] == h) & (test["city"].astype(str) == city)]
            parts.append(prediction_rows(test_ch, predict_lgbm(model, test_ch), "per_city", [city]))
    pred = pd.concat(parts, ignore_index=True)
    m_pooled = experiment_metrics(pred[pred["experiment"] == "pooled"], "pooled")
    m_per = experiment_metrics(pred[pred["experiment"] == "per_city"], "per_city")
    diff = m_pooled.merge(m_per, on=["scope", "city", "horizon"], suffixes=("_p", "_c"))
    if not (diff["n_p"] == diff["n_c"]).all():
        raise AssertionError("pooled and per-city scored on different rows")
    d = pd.DataFrame({"experiment": "pooled_minus_per_city", "scope": diff["scope"], "city": diff["city"],
                      "horizon": diff["horizon"], "n": diff["n_p"],
                      **{k: diff[f"{k}_p"] - diff[f"{k}_c"] for k in ("mae", "rmse", "r2", "skill_mae", "skill_rmse")},
                      "train_cities": "difference"})
    table = pd.concat([m_pooled, m_per, d], ignore_index=True)
    da.write_csv(table, dirs["tables"] / FILES["pooled"], cfg, "src.cross_city.run_pooled_per_city",
                 "Common eval_mask rows, 2019 test; skill vs persistence on identical rows.")
    write_predictions(pred, cfg, dirs, replace=("pooled", "per_city"))
    return {"pred": pred, "table": table, "train": train, "test": test}


def transfer_roles(cfg: dict) -> dict[str, str]:
    return {**{c: "transfer" for c in cfg["cities"]["transfer"]},
            **{c: "transfer_optional" for c in cfg["cities"]["transfer_optional"]}}


def loco_predictions(cfg: dict, train: pd.DataFrame, test: pd.DataFrame, dirs: dict) -> pd.DataFrame:
    fs, params, (train_range, _, _) = cfg["cross_city"]["feature_set"], lightgbm_params(cfg), main_ranges(cfg)
    core, row_key, parts = list(cfg["cities"]["core"]), ["city", "origin_date", "horizon"], []
    for held in core:
        tr = train.loc[train["city"].astype(str).isin(core) & (train["city"].astype(str) != held)]
        key_held = pd.MultiIndex.from_frame(train.loc[train["city"].astype(str) == held, row_key].astype({"city": str}))
        if len(key_held) == 0:
            raise AssertionError(f"no training-period rows found for {held}")
        if not pd.MultiIndex.from_frame(tr[row_key].astype({"city": str})).intersection(key_held).empty:
            raise LeakageError(f"held-out city {held} present in its LOCO training frame")
        if set(tr["city"].astype(str)) != set(core) - {held} or set(tr["city"].astype(str)) & set(transfer_roles(cfg)):
            raise LeakageError(f"LOCO training cities for {held} != core minus {held}")
        for h in cfg["leakage"]["protocol_C"]["horizons"]:
            model, meta = fit_lgbm(tr, fs, h, params, train_range, cfg)
            save_model(model, {**meta, "experiment": "loco", "held_out_city": held},
                       dirs["models"] / MODELS_SUBDIR / f"loco_{held}_h{h}.txt")
            test_ch = test.loc[(test["horizon"] == h) & (test["city"].astype(str) == held)]
            parts.append(prediction_rows(test_ch, predict_lgbm(model, test_ch), "loco", meta["train_cities"], held))
    return pd.concat(parts, ignore_index=True)


def transfer_predictions(cfg: dict, dirs: dict) -> pd.DataFrame:
    roles, parts = transfer_roles(cfg), []
    df = read_without_calibration(da.resolve(cfg, "forecast_dataset"), cfg)
    _, _, (te_lo, te_hi) = main_ranges(cfg)
    tf = df.loc[df["city"].astype(str).isin(list(roles)) & df["target_date"].between(te_lo, te_hi)]
    if tf["y_target"].isna().any():
        raise ValueError("transfer rows must have observed targets")
    for h in cfg["leakage"]["protocol_C"]["horizons"]:
        booster, meta = load_model(dirs["models"] / POINT_MODEL_FILE.format(h=h))
        if set(meta["train_cities"]) != set(cfg["cities"]["core"]):
            raise LeakageError("pooled model was not trained on exactly the core cities")
        tf_h = tf.loc[tf["horizon"] == h]
        parts.append(prediction_rows(tf_h, predict_lgbm(booster, tf_h), "transfer", meta["train_cities"]))
    return pd.concat(parts, ignore_index=True)


def run_loco_transfer(cfg: dict, dirs: dict | None = None) -> dict:
    dirs = dirs or da.output_dirs(cfg)
    train, test = forecast_frames(cfg)
    loco_pred, t_pred, roles = loco_predictions(cfg, train, test, dirs), transfer_predictions(cfg, dirs), transfer_roles(cfg)
    ref = pd.read_csv(dirs["tables"] / FILES["pooled"])
    ref = ref[ref["experiment"].isin(["pooled", "per_city"])].pivot_table(
        index=["scope", "city", "horizon"], columns="experiment", values="mae").reset_index()
    loco = (experiment_metrics(loco_pred, "loco").merge(ref, on=["scope", "city", "horizon"], how="left")
            .rename(columns={"city": "held_out_city", "pooled": "mae_pooled", "per_city": "mae_per_city"}))
    loco["gap_vs_pooled_pct"] = relative_change_pct(loco["mae"], loco["mae_pooled"])
    loco["gap_vs_per_city_pct"] = relative_change_pct(loco["mae"], loco["mae_per_city"])
    m_t = experiment_metrics(t_pred, "transfer", scopes=("city",))
    transfer = m_t.assign(city_role=m_t["city"].map(roles), optional=m_t["city"].map(roles).eq("transfer_optional"))
    transfer["label"] = np.where(transfer["optional"], "OPTIONAL " + LABEL_TRANSFER, LABEL_TRANSFER)
    da.write_csv(loco[LOCO_COLUMNS], dirs["tables"] / FILES["loco"], cfg, "src.cross_city.run_loco_transfer",
                 "Held-out-city generalisation with local history; gap_vs_*_pct = 100*(MAE_loco/MAE_ref - 1).")
    da.write_csv(transfer[TRANSFER_COLUMNS], dirs["tables"] / FILES["transfer"], cfg,
                 "src.cross_city.run_loco_transfer", "Pooled core-city models applied to cities never trained on.")
    write_predictions(pd.concat([loco_pred, t_pred], ignore_index=True), cfg, dirs, replace=("loco", "transfer"))
    return {"loco": loco[LOCO_COLUMNS], "transfer": transfer[TRANSFER_COLUMNS], "loco_pred": loco_pred,
            "transfer_pred": t_pred, "train": train, "test": test}


def run(cfg: dict) -> dict:
    return {"pooled": run_pooled_per_city(cfg), "loco": run_loco_transfer(cfg)}


def make_figures(cfg: dict, dirs: dict | None = None) -> list[Path]:
    dirs = dirs or da.output_dirs(cfg)
    core, hs = list(cfg["cities"]["core"]), cfg["leakage"]["protocol_C"]["horizons"]
    loco, table = (pd.read_csv(dirs["tables"] / FILES[k]) for k in ("loco", "pooled"))
    C = read_predictions(dirs["predictions"] / LEAKAGE_FILES["C"], cfg)
    pers = C[(C["method"] == "persistence") & C["eval_mask"]]
    lc, city_rows = loco[loco["scope"] == "city"], table[table["scope"] == "city"]
    grid = lc.pivot(index="held_out_city", columns="horizon", values="gap_vs_pooled_pct").reindex(core)
    mae = {key: {h: city_rows[(city_rows["experiment"] == key) & (city_rows["horizon"] == h)]
                 .set_index("city").reindex(core)["mae"].tolist() for h in hs} for key in ("pooled", "per_city")}
    mae["loco"] = {h: lc[lc["horizon"] == h].set_index("held_out_city").reindex(core)["mae"].tolist() for h in hs}
    mae_of = lambda g: point_metrics(g["y_true"], g["y_pred"])["mae"]
    pers_mae = {h: [mae_of(pers[(pers["horizon"] == h) & (pers["city"] == c)]) for c in core] for h in hs}
    return [figures.heatmaps({"Leave-one-city-out: MAE change vs pooled model\n(held-out city, 2019, eval_mask rows)":
                              grid[hs].to_numpy()}, core, hs, dirs["figures"] / "cross_city_heatmap.png", "",
                             "100·(MAE_LOCO / MAE_pooled − 1)  [%]", fmt="{:+.1f}%"),
            figures.model_comparison(core, hs, mae, pers_mae, dirs["figures"] / "cross_city_model_comparison.png")]
