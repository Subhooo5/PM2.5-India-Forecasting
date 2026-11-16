from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

from src import data_audit as da
from src.features import SAME_DAY_SET, encode_city, feature_set
from src.splits import LeakageError

KEY_COLUMNS = ["city", "origin_date", "target_date", "horizon"]
STRUCTURAL_KEYS = ("num_leaves", "min_child_samples")
EARLY_STOPPING_KEYS = {"early_stopping_round", "early_stopping_rounds", "early_stopping", "n_iter_no_change"}
SAME_DAY_COPOLLUTANTS = ("PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene")
ALWAYS_FORBIDDEN_IN_CAUSAL = ("AQI", "AQI_Bucket") + SAME_DAY_COPOLLUTANTS
TARGET_COLUMNS = ("y_target", "y_true")


def persistence(ds: pd.DataFrame) -> pd.Series:
    return ds["pm25_origin"].rename("yhat_persistence")


def seasonal_naive(ds: pd.DataFrame) -> pd.Series:
    return ds["snaive_ref"].rename("yhat_seasonal_naive")


def baseline_predictions(ds: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([ds[["city", "origin_date", "target_date", "horizon", "y_target"]],
                      persistence(ds), seasonal_naive(ds)], axis=1)


def base_params(cfg: dict) -> dict:
    lgbm = cfg["lightgbm"]
    if lgbm.get("early_stopping"):
        raise ValueError("early stopping would consult held-out data; it is disabled by design")
    params = dict(lgbm["base"])
    for key in ("bagging_seed", "random_state"):
        if params.get(key) != cfg["seed"]:
            raise ValueError(f"lightgbm.base.{key} must equal the top-level seed")
    return params


def candidate_params(cfg: dict) -> list[dict]:
    grid = cfg["lightgbm"]["grid"]
    expected = [{"candidate_id": f"c{i}", "num_leaves": nl, "min_child_samples": mcs}
                for i, (nl, mcs) in enumerate(itertools.product(grid["num_leaves"], grid["min_child_samples"]), 1)]
    if [dict(c) for c in grid["candidates"]] != expected:
        raise ValueError("lightgbm.grid.candidates must enumerate num_leaves x min_child_samples as c1..cN")
    return expected


def merge_params(cfg: dict, structural: dict) -> dict:
    return {**base_params(cfg), **{k: structural[k] for k in STRUCTURAL_KEYS}}


def lightgbm_params(cfg: dict) -> dict:
    if not cfg["lightgbm"].get("selected"):
        raise ValueError("lightgbm.selected is not set: run the tuning stage first")
    return merge_params(cfg, cfg["lightgbm"]["selected"])


def quantile_params(cfg: dict, alpha: float) -> dict:
    if alpha not in cfg["lightgbm"]["quantile_alphas"]:
        raise ValueError(f"alpha {alpha} not predeclared")
    return {**lightgbm_params(cfg), "objective": "quantile", "alpha": alpha}


def target_columns(cfg: dict) -> set[str]:
    return set(TARGET_COLUMNS) | {cfg["data"]["target"]}


def check_feature_set(cfg: dict, name: str) -> list[str]:
    feats = feature_set(cfg, name)
    leaked = target_columns(cfg) & set(feats)
    if leaked:
        raise LeakageError(f"target column(s) {sorted(leaked)} are never features (set {name!r})")
    if "AQI_Bucket" in feats:
        raise LeakageError(f"AQI_Bucket is never a feature (set {name!r})")
    forbidden = (set(cfg["features"]["forbidden"]) | set(ALWAYS_FORBIDDEN_IN_CAUSAL)) & set(feats)
    if name != SAME_DAY_SET and forbidden:
        raise LeakageError(f"forbidden same-day columns {sorted(forbidden)} in {name!r}")
    if "city" in feats and "city_aware" not in name:
        raise LeakageError(f"city ID is only allowed in city-aware sets, not {name!r}")
    if name == SAME_DAY_SET and any(f.startswith("pm25_") for f in feats):
        raise LeakageError("same_day_estimation must not contain PM2.5 history")
    return feats


def feature_matrix(rows: pd.DataFrame, features: list[str], train_cities) -> pd.DataFrame:
    missing = [f for f in features if f not in rows.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    X = rows.loc[:, features].copy()
    if "city" in features:
        X["city"] = encode_city(rows, list(train_cities))
    return X


def fit_lgbm(df: pd.DataFrame, feature_set_name: str, horizon: int | None, params: dict,
             train_date_range, cfg: dict):
    feats = check_feature_set(cfg, feature_set_name)
    if EARLY_STOPPING_KEYS & set(params):
        raise ValueError(f"early stopping is disabled by design: {sorted(EARLY_STOPPING_KEYS & set(params))}")
    if horizon is None and "horizon" in df.columns:
        raise ValueError("horizon=None but df has a horizon column: pick one horizon")
    rows = df if horizon is None else df.loc[df["horizon"] == horizon]
    lo, hi = (pd.Timestamp(x) for x in train_date_range)
    outside = ~rows["target_date"].between(lo, hi)
    if outside.any():
        raise LeakageError(f"{int(outside.sum())} training rows have target dates outside {lo.date()}..{hi.date()}")
    if rows.empty:
        raise ValueError("no training rows")
    if rows["y_target"].isna().any():
        raise ValueError("training rows must have an observed target; targets are never imputed")
    train_cities = sorted(rows["city"].astype(str).unique())
    model = lightgbm.LGBMRegressor(**params)
    model.fit(feature_matrix(rows, feats, train_cities), rows["y_target"])
    if model.booster_.feature_name() != feats:
        raise AssertionError("LightGBM stored a different feature order")
    meta = {"feature_set": feature_set_name, "features": feats, "horizon": horizon, "params": dict(params),
            "train_date_range": [str(lo.date()), str(hi.date())], "n_train": int(len(rows)),
            "train_cities": train_cities, "seed": cfg["seed"],
            "dataset_sha256": da.sha256_file(da.resolve(cfg, "raw_city_day")),
            "kaggle_version": cfg["data"]["kaggle_version"], "config_hash": da.config_hash(cfg),
            "versions": {"lightgbm": lightgbm.__version__, "numpy": np.__version__, "pandas": pd.__version__},
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    setattr(model, "pm25_meta_", meta)
    return model, meta


def _booster(model) -> lightgbm.Booster:
    return model.booster_ if isinstance(model, lightgbm.LGBMModel) else model


def predict_lgbm(model, df: pd.DataFrame) -> pd.DataFrame:
    meta = getattr(model, "pm25_meta_", None)
    if meta is None:
        raise ValueError("model has no metadata: use fit_lgbm / load_model")
    if _booster(model).feature_name() != meta["features"]:
        raise ValueError("model feature list/order does not match its metadata")
    if meta["horizon"] is not None and (df["horizon"] != meta["horizon"]).any():
        raise ValueError(f"rows of another horizon passed to the h={meta['horizon']} model")
    y_pred = _booster(model).predict(feature_matrix(df, meta["features"], meta["train_cities"]))
    out = pd.DataFrame(index=df.index)
    for k in KEY_COLUMNS:
        out[k] = df[k] if k in df.columns else pd.NA
    out["y_true"] = df["y_target"]
    out["y_pred"] = np.asarray(y_pred, dtype=float)
    return out


def save_model(model, meta: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _booster(model).save_model(str(path))
    da.write_sidecar(meta, path)
    return path


def load_model(path):
    path = Path(path)
    booster = lightgbm.Booster(model_file=str(path))
    meta = json.loads(path.with_name(f"{path.stem}.meta.json").read_text())
    setattr(booster, "pm25_meta_", meta)
    return booster, meta
