"""Forecasting models.

Day 1, item 8: reference baselines (no fitting):
- persistence:     yhat[t+h] = y[t]        (column `pm25_origin`, always observed)
- seasonal naive:  yhat[t+h] = y[t+h-7]    (column `snaive_ref`, raw observed value only)
Both must be scored on the common evaluation sample `eval_mask` (see CLAUDE.md).

Hyperparameters: `lightgbm_params` (selection rule below; no training happens at import).
Day 2 task 3: shared pipeline `fit_lgbm` / `predict_lgbm` / `evaluate_predictions` /
`save_model` / `load_model` + feature-set guard (no experiment logic here).
TODO(RQ1, RQ2 - later day): run pooled point models per protocol / RQ.
TODO(RQ3 - later day): LightGBM quantile models (0.05 / 0.95).
"""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

from src import data_audit as da
from src.features import encode_city, feature_set
from src.metrics import point_metrics
from src.splits import LeakageError, assert_training_only


def persistence(ds: pd.DataFrame) -> pd.Series:
    """yhat[t+h] = y[t]."""
    return ds["pm25_origin"].rename("yhat_persistence")


def seasonal_naive(ds: pd.DataFrame) -> pd.Series:
    """yhat[t+h] = y[t+h-7] (raw, unfilled; NaN where that day was not observed)."""
    return ds["snaive_ref"].rename("yhat_seasonal_naive")


BASELINES = {"persistence": persistence, "seasonal_naive": seasonal_naive}


def baseline_predictions(ds: pd.DataFrame, eval_only: bool = True) -> pd.DataFrame:
    """Predictions of every baseline; restricted to the common `eval_mask` by default."""
    rows = ds.loc[ds["eval_mask"]] if eval_only else ds
    preds = pd.concat([fn(rows) for fn in BASELINES.values()], axis=1)
    keys = ["city", "origin_date", "target_date", "horizon", "y_target"]
    return pd.concat([rows[keys], preds], axis=1)


# ------------------------------------------------------------------ hyperparameter rule
# Frozen 2026-11-14 (Day 2 task 1): `lightgbm.base` + one of the 6 predeclared
# `lightgbm.grid.candidates`, chosen by expanding-window CV inside 2015-2017
# (src/experiments.py::run_tuning) and written once to `lightgbm.selected`.
# Every experiment reuses `lightgbm_params(cfg)`; RQ3 changes only objective/alpha.

def base_params(cfg: dict) -> dict:
    """`lightgbm.base` (every point model starts from these). Early stopping is never allowed."""
    lgbm = cfg["lightgbm"]
    if lgbm.get("early_stopping"):
        raise ValueError("early stopping would consult held-out data; it is disabled by design")
    params = dict(lgbm["base"])
    for key in ("bagging_seed", "random_state"):
        if params.get(key) != cfg["seed"]:
            raise ValueError(f"lightgbm.base.{key} must equal the top-level seed")
    return params


def candidate_params(cfg: dict) -> list[dict]:
    """The predeclared grid candidates in fixed order (c1..c6 = tie-break order).

    Each entry: {candidate_id, num_leaves, min_child_samples}. Verifies the explicit list is
    exactly num_leaves (outer loop) x min_child_samples (inner loop).
    """
    grid = cfg["lightgbm"]["grid"]
    expected = [{"candidate_id": f"c{i}", "num_leaves": nl, "min_child_samples": mcs}
                for i, (nl, mcs) in enumerate(
                    itertools.product(grid["num_leaves"], grid["min_child_samples"]), start=1)]
    if [dict(c) for c in grid["candidates"]] != expected:
        raise ValueError("lightgbm.grid.candidates must enumerate num_leaves (outer) x "
                         "min_child_samples (inner) as c1..cN")
    return [dict(c) for c in grid["candidates"]]


STRUCTURAL_KEYS = ("num_leaves", "min_child_samples")


def merge_params(cfg: dict, structural: dict) -> dict:
    """Full parameter dict = lightgbm.base + structural params (candidate_id dropped)."""
    return {**base_params(cfg), **{k: structural[k] for k in STRUCTURAL_KEYS}}


def lightgbm_params(cfg: dict) -> dict:
    """THE point-model parameters for every experiment: base + `lightgbm.selected`. No data."""
    selected = cfg["lightgbm"].get("selected")
    if not selected:
        raise ValueError("lightgbm.selected is not set: run the task 2 tuning first")
    return merge_params(cfg, selected)


def quantile_params(cfg: dict, alpha: float) -> dict:
    """RQ3 quantile models reuse the selected parameters; only objective/alpha change."""
    if alpha not in cfg["lightgbm"]["quantile_alphas"]:
        raise ValueError(f"alpha {alpha} not predeclared")
    return {**lightgbm_params(cfg), "objective": "quantile", "alpha": alpha}


def final_training_rows(ds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Refit data for the main scheme: every split_main == train row (2015-2017), guarded."""
    rows = ds.loc[ds["split_main"] == "train"]
    assert_training_only(rows["target_date"], cfg)
    return rows


# ------------------------------------------------------------------ shared LightGBM pipeline
# Day 2 task 3. Reusable functions only, no experiment logic. Direct forecasting: one model
# per horizon. Missing features stay NaN (LightGBM native); the target is never imputed.

KEY_COLUMNS = ["city", "origin_date", "target_date", "horizon"]
EARLY_STOPPING_KEYS = {"early_stopping_round", "early_stopping_rounds", "early_stopping",
                       "n_iter_no_change"}
SAME_DAY_SET = "same_day_estimation"


def check_feature_set(cfg: dict, name: str) -> list[str]:
    """Resolve a named feature set and enforce the leakage guards.

    - AQI_Bucket: never allowed in any set.
    - AQI / same-day co-pollutants: only in `same_day_estimation` (RQ1 A/B).
    - City ID: only in city-aware sets.
    - `same_day_estimation` may not contain PM2.5 history (pm25_*).
    """
    feats = feature_set(cfg, name)
    forbidden = set(cfg["features"]["forbidden"])
    if "AQI_Bucket" in feats:
        raise LeakageError(f"AQI_Bucket is never a feature (set {name!r})")
    if name != SAME_DAY_SET and forbidden & set(feats):
        raise LeakageError(f"forbidden same-day columns {sorted(forbidden & set(feats))} in {name!r}")
    if "city" in feats and "city_aware" not in name:
        raise LeakageError(f"city ID is only allowed in city-aware sets, not {name!r}")
    if name == SAME_DAY_SET and any(f.startswith("pm25_") for f in feats):
        raise LeakageError("same_day_estimation must not contain PM2.5 history")
    return feats


def validate_feature_frame(X: pd.DataFrame, features: list[str]) -> None:
    """Raise unless X has exactly `features`, in exactly that order."""
    cols = list(X.columns)
    if cols == list(features):
        return
    missing = [f for f in features if f not in cols]
    extra = [c for c in cols if c not in features]
    raise ValueError(f"feature columns mismatch: missing={missing} extra={extra} "
                     f"reordered={not missing and not extra}")


def _feature_matrix(rows: pd.DataFrame, features: list[str], train_cities) -> pd.DataFrame:
    missing = [f for f in features if f not in rows.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    X = rows.loc[:, features].copy()
    if "city" in features:                              # unseen cities -> NaN (explicit)
        X["city"] = encode_city(rows, list(train_cities))
    validate_feature_frame(X, features)
    return X


def fit_lgbm(df: pd.DataFrame, feature_set_name: str, horizon: int | None, params: dict,
             train_date_range, config: dict, *, stage: str = "final"):
    """Fit one direct LightGBM model. Returns (model, metadata); metadata is also on model.pm25_meta_.

    - `df` rows are filtered to `horizon` (None = table without horizons, e.g. same-day).
    - Every remaining row's TARGET date must lie in `train_date_range`, else LeakageError.
    - stage == 'tuning': additionally no target date may exceed tuning.max_allowed_target_date.
    - Training targets must be observed (never imputed); missing features stay NaN.
    - No eval set and no early stopping, ever.
    """
    feats = check_feature_set(config, feature_set_name)
    bad_keys = EARLY_STOPPING_KEYS & set(params)
    if bad_keys:
        raise ValueError(f"early stopping is disabled by design: {sorted(bad_keys)}")
    if horizon is None:
        if "horizon" in df.columns:
            raise ValueError("horizon=None but df has a horizon column: pick one horizon")
        rows = df
    else:
        rows = df.loc[df["horizon"] == horizon]
    lo, hi = (pd.Timestamp(x) for x in train_date_range)
    outside = ~rows["target_date"].between(lo, hi)
    if outside.any():
        raise LeakageError(f"{int(outside.sum())} training rows have target dates outside "
                           f"{lo.date()}..{hi.date()}")
    if stage == "tuning":
        assert_training_only(rows["target_date"], config)
        if hi > pd.Timestamp(config["tuning"]["max_allowed_target_date"]):
            raise LeakageError("tuning train_date_range extends past the hard guard")
    if rows.empty:
        raise ValueError("no training rows")
    if rows["y_target"].isna().any():
        raise ValueError("training rows must have an observed target; targets are never imputed")
    train_cities = sorted(rows["city"].astype(str).unique())
    X = _feature_matrix(rows, feats, train_cities)
    model = lightgbm.LGBMRegressor(**params)
    model.fit(X, rows["y_target"])
    if model.booster_.feature_name() != feats:
        raise AssertionError("LightGBM stored a different feature order")
    meta = {
        "feature_set": feature_set_name,
        "features": feats,
        "horizon": horizon,
        "params": dict(params),
        "train_date_range": [str(lo.date()), str(hi.date())],
        "n_train": int(len(rows)),
        "train_cities": train_cities,
        "stage": stage,
        "seed": config["seed"],
        "dataset_sha256": da.sha256_file(da.resolve(config, "raw_city_day")),
        "kaggle_version": config["data"]["kaggle_version"],
        "lightgbm_version": lightgbm.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    setattr(model, "pm25_meta_", meta)
    return model, meta


def _booster(model) -> lightgbm.Booster:
    return model.booster_ if isinstance(model, lightgbm.LGBMModel) else model


def predict_lgbm(model, df: pd.DataFrame) -> pd.DataFrame:
    """Predict every row of df (none dropped), keeping df's index and row order.

    Returns city, origin_date, target_date, horizon, y_true, y_pred. Raises if a feature is
    missing, if the model's stored feature list/order disagrees with the booster, or if df
    contains rows of another horizon than the model's.
    """
    meta = getattr(model, "pm25_meta_", None)
    if meta is None:
        raise ValueError("model has no pm25 metadata: use fit_lgbm / load_model")
    feats = meta["features"]
    if _booster(model).feature_name() != feats:
        raise ValueError("model feature list/order does not match its metadata")
    if meta["horizon"] is not None and (df["horizon"] != meta["horizon"]).any():
        raise ValueError(f"rows of another horizon passed to the h={meta['horizon']} model")
    X = _feature_matrix(df, feats, meta["train_cities"])
    y_pred = _booster(model).predict(X)
    out = pd.DataFrame(index=df.index)
    for k in KEY_COLUMNS:
        out[k] = df[k] if k in df.columns else pd.NA
    out["y_true"] = df["y_target"]
    out["y_pred"] = np.asarray(y_pred, dtype=float)
    return out


def evaluate_predictions(pred_df: pd.DataFrame, mask=None) -> dict:
    """{mae, rmse, r2, n} on rows selected by `mask` (aligned boolean), observed targets only."""
    if mask is None:
        rows = pred_df
    else:
        mask = pd.Series(mask, index=pred_df.index) if not isinstance(mask, pd.Series) else mask
        if not mask.index.equals(pred_df.index):
            raise ValueError("mask index must match pred_df index")
        rows = pred_df.loc[mask.astype(bool)]
    return point_metrics(rows["y_true"], rows["y_pred"])


def save_model(model, metadata: dict, path) -> Path:
    """LightGBM text model at `path` + metadata JSON at <stem>.meta.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _booster(model).save_model(str(path))
    da.write_sidecar(metadata, path)
    return path


def load_model(path):
    """Returns (booster, metadata); metadata attached as booster.pm25_meta_ for predict_lgbm."""
    path = Path(path)
    booster = lightgbm.Booster(model_file=str(path))
    meta = json.loads(path.with_name(f"{path.stem}.meta.json").read_text())
    setattr(booster, "pm25_meta_", meta)
    return booster, meta
