from __future__ import annotations

import json
from typing import cast

import lightgbm
import numpy as np
import pandas as pd

from src import data_audit as da
from src.features import feature_set
from src.metrics import point_metrics, skill_score
from src.models import STRUCTURAL_KEYS, candidate_params, merge_params
from src.splits import LeakageError, assert_training_only, expanding_window_folds, main_ranges, split_ranges

TUNING_CV_FILE = "tuning_cv_results.csv"
SELECTED_FILE = "selected_params.json"
CV_COLUMNS = ["candidate_id", "num_leaves", "min_child_samples", "fold", "horizon", "n_train", "n_val",
              "mae", "rmse", "r2"]
POINT_METHODS = ("lightgbm", "persistence", "seasonal_naive")
RESULT_COLUMNS = ["method", "protocol", "scope", "city", "horizon", "n", "mae", "rmse", "r2", "skill_mae",
                  "skill_rmse", "mask_type", "note"]


def assert_no_calibration(dates: pd.Series, cfg: dict, what: str) -> None:
    _, (lo, hi), _ = main_ranges(cfg)
    if pd.to_datetime(dates).between(lo, hi).any():
        raise LeakageError(f"{what}: calibration-reserved TARGET rows present")


def read_without_calibration(path, cfg: dict) -> pd.DataFrame:
    (tr_lo, tr_hi), _, (te_lo, te_hi) = main_ranges(cfg)
    df = pd.read_parquet(path, filters=[[("target_date", ">=", tr_lo), ("target_date", "<=", tr_hi)],
                                        [("target_date", ">=", te_lo), ("target_date", "<=", te_hi)]])
    assert_no_calibration(df["target_date"], cfg, "loaded frame")
    return df


def forecast_frames(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_without_calibration(da.resolve(cfg, "forecast_dataset"), cfg)
    core = df["city"].astype(str).isin(cfg["cities"]["core"])
    (tr_lo, tr_hi), _, (te_lo, te_hi) = main_ranges(cfg)
    train = df.loc[core & df["target_date"].between(tr_lo, tr_hi)]
    test = df.loc[core & df["target_date"].between(te_lo, te_hi)]
    if not ((train["split_main"] == "train").all() and (test["split_main"] == "test").all()):
        raise LeakageError("split labels disagree with target-date ranges")
    if train["y_target"].isna().any() or test["y_target"].isna().any():
        raise ValueError("forecasting rows must have observed targets")
    return train, test


def read_predictions(path, cfg: dict) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, parse_dates=[c for c in ("origin_date", "target_date") if c in head])
    assert_no_calibration(df["target_date"], cfg, path.name)
    return df


def score_methods(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pers = pred.loc[pred["method"] == "persistence"]
    for method, m in pred.groupby("method", sort=False):
        for mask_type in ("common_eval_mask", "own_full"):
            scored = m.loc[m["eval_mask"] if mask_type == "common_eval_mask" else m["y_pred"].notna()]
            for scope in ("overall", "city"):
                keys = ["horizon"] + (["city"] if scope == "city" else [])
                for gkey, g in scored.groupby(keys, observed=True, sort=True):
                    rec = dict(zip(keys, gkey if isinstance(gkey, tuple) else (gkey,)))
                    ref = pers.loc[pers["row_id"].isin(g["row_id"])]
                    met, ref_met = point_metrics(g["y_true"], g["y_pred"]), point_metrics(ref["y_true"], ref["y_pred"])
                    if ref_met["n"] != met["n"]:
                        raise AssertionError("persistence reference rows differ from scored rows")
                    rows.append({"method": method, "protocol": m["protocol"].iloc[0], "scope": scope,
                                 "city": rec.get("city", "all"), "horizon": rec.get("horizon"), **met,
                                 "skill_mae": skill_score(met["mae"], ref_met["mae"]),
                                 "skill_rmse": skill_score(met["rmse"], ref_met["rmse"]), "mask_type": mask_type})
    out = pd.DataFrame(rows)
    out["note"] = np.where((out["method"] == "seasonal_naive") & (out["horizon"] == 7),
                           "h=7: seasonal naive y[t+7-7]=y[t] equals persistence (expected)", "")
    return out[RESULT_COLUMNS]


def input_exposure(test: pd.DataFrame, cfg: dict, scheme: str) -> list[dict]:
    """Test rows whose origin, or whose longest causal lookback, reaches into the calibration year."""
    cal_lo, cal_hi = split_ranges(cfg, scheme)["calibration"]
    f, year = cfg["features"], cal_lo.year
    lookback = max(max(f["rolling_windows"]) - 1, max(f["lags"]) + int(cfg["preprocessing"]["ffill_limit"]))
    out = []
    for h, g in test.groupby("horizon", sort=True):
        origin = pd.to_datetime(g["origin_date"])
        reaches = (origin - pd.Timedelta(days=lookback) <= cal_hi) & (origin >= cal_lo)
        out.append({"horizon": int(cast(int, h)), "n_test_rows": int(len(g)),
                    f"n_origin_in_{year}": int(origin.between(cal_lo, cal_hi).sum()),
                    f"n_lookback_reaches_{year}": int(reaches.sum()), "lookback_days": int(lookback)})
    return out


def _comparable(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[c] = s.astype("datetime64[ns]")
        elif pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(bool)
        elif pd.api.types.is_integer_dtype(s):
            out[c] = s.astype("int64")
        elif pd.api.types.is_float_dtype(s):
            out[c] = s.astype("float64")
        else:
            out[c] = s.astype(str).astype(object)
    return pd.DataFrame(out, index=df.index)


def assert_frames_identical(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], columns: list[str],
                            what: str) -> None:
    if len(left) != len(right):
        raise AssertionError(f"{what}: length mismatch {len(left)} != {len(right)}")
    for side, df in (("left", left), ("right", right)):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise AssertionError(f"{what}: {side} side missing columns {missing}")
    a, b = (_comparable(d[columns]).sort_values(keys, kind="stable").reset_index(drop=True) for d in (left, right))
    for c in columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if not (np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f" else bool((x == y).all())):
            raise AssertionError(f"{what}: column {c!r} differs")


def load_tuning_frame(cfg: dict) -> pd.DataFrame:
    max_date = pd.Timestamp(cfg["tuning"]["max_allowed_target_date"])
    frame = pd.read_parquet(da.resolve(cfg, "forecast_dataset"), filters=[("target_date", "<=", max_date)])
    cities = list(cfg["cities"][cfg["tuning"]["cities"]])
    frame = frame.loc[frame["city"].astype(str).isin(cities)
                      & frame["horizon"].isin(cfg["tuning"]["horizons"])].reset_index(drop=True)
    if frame["target_date"].max() > max_date:
        raise LeakageError("tuning frame contains target dates after the hard guard")
    if set(frame["city"].astype(str)) != set(cities):
        raise ValueError("tuning frame does not contain exactly the core cities")
    assert_training_only(frame["target_date"], cfg)
    return frame


def tune_frame(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    assert_training_only(frame["target_date"], cfg)
    if frame["y_target"].isna().any():
        raise ValueError("training/validation rows must have an observed target (never imputed)")
    t = cfg["tuning"]
    feats = feature_set(cfg, t["feature_set"])
    records = []
    for fold_no, (tri, vai) in enumerate(expanding_window_folds(frame, cfg), start=1):
        for cand in candidate_params(cfg):
            for h in t["horizons"]:
                tr = frame.loc[tri][lambda d: d["horizon"] == h]
                va = frame.loc[vai][lambda d: (d["horizon"] == h) & d[t["eval_rows"]]]
                assert_training_only(tr["target_date"], cfg)
                assert_training_only(va["target_date"], cfg)
                model = lightgbm.LGBMRegressor(**merge_params(cfg, cand))
                model.fit(cast(pd.DataFrame, tr[feats]), tr["y_target"])
                m = point_metrics(va["y_target"], model.predict(cast(pd.DataFrame, va[feats])))
                records.append({"candidate_id": cand["candidate_id"], **{k: cand[k] for k in STRUCTURAL_KEYS},
                                "fold": fold_no, "horizon": h, "n_train": len(tr), "n_val": m["n"],
                                "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"]})
    return pd.DataFrame(records, columns=CV_COLUMNS)


def select_candidate(cv: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    cands = candidate_params(cfg)
    order = [c["candidate_id"] for c in cands]
    counts = cv.groupby("candidate_id").size().reindex(order)
    if counts.isna().any() or (counts != len(cfg["tuning"]["folds"]) * len(cfg["tuning"]["horizons"])).any():
        raise ValueError("every candidate needs exactly one score per fold x horizon")
    mean_mae = cv.groupby("candidate_id")["mae"].mean().reindex(order)
    per_h = cv.pivot_table(index="candidate_id", columns="horizon", values="mae", aggfunc="mean")
    best_id = next(cid for cid in order if mean_mae[cid] == mean_mae.min())
    ranking = pd.DataFrame(cands)
    ranking["mean_val_mae"] = mean_mae.to_numpy()
    ranking = ranking.join(per_h.reindex(order).add_prefix("mean_mae_h").reset_index(drop=True))
    ranking["grid_order"] = range(1, len(order) + 1)
    ranking["rank"] = ranking.sort_values(["mean_val_mae", "grid_order"]).index.argsort() + 1
    ranking["selected"] = ranking["candidate_id"] == best_id
    return next(c for c in cands if c["candidate_id"] == best_id), ranking


def run_tuning(cfg: dict) -> dict:
    frame = load_tuning_frame(cfg)
    cv = tune_frame(frame, cfg)
    selected, ranking = select_candidate(cv, cfg)
    if selected != {**cfg["lightgbm"]["selected"]}:
        raise ValueError(f"tuning selected {selected}, but the config fixes {cfg['lightgbm']['selected']}")
    models_dir = da.output_dirs(cfg)["models"]
    cv_path = da.write_csv(cv, models_dir / TUNING_CV_FILE, cfg, "src.experiments.run_tuning",
                           "Expanding-window CV inside 2015-2017 only; eval_mask rows.")
    payload = {"selected_candidate_id": selected["candidate_id"],
               "selected_structural_params": {k: selected[k] for k in STRUCTURAL_KEYS},
               "full_params": merge_params(cfg, selected), "ranking": json.loads(ranking.to_json(orient="records")),
               "folds": cfg["tuning"]["folds"], "horizons": cfg["tuning"]["horizons"],
               "feature_set": cfg["tuning"]["feature_set"], "features": feature_set(cfg, cfg["tuning"]["feature_set"]),
               "cities": list(cfg["cities"][cfg["tuning"]["cities"]]),
               "tuning_frame": {"n_rows": int(len(frame)), "min_target_date": str(frame["target_date"].min().date()),
                                "max_target_date": str(frame["target_date"].max().date())},
               "processed_dataset_sha256": da.sha256_file(da.resolve(cfg, "forecast_dataset")),
               **da.metadata(cv, SELECTED_FILE, cfg, "src.experiments.run_tuning")}
    (models_dir / SELECTED_FILE).write_text(json.dumps(payload, indent=2, default=str))
    return {"frame": frame, "cv": cv, "ranking": ranking, "selected": selected, "payload": payload,
            "paths": (cv_path, models_dir / SELECTED_FILE)}
