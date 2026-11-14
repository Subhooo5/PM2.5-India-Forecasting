"""Forecasting models.

Day 1, item 8: reference baselines (no fitting):
- persistence:     yhat[t+h] = y[t]        (column `pm25_origin`, always observed)
- seasonal naive:  yhat[t+h] = y[t+h-7]    (column `snaive_ref`, raw observed value only)
Both must be scored on the common evaluation sample `eval_mask` (see CLAUDE.md).

Hyperparameters: `lightgbm_params` (selection rule below; no training happens at import).
TODO(RQ1, RQ2 - later day): pooled LightGBM point model, one direct model per horizon h in {1,3,7}.
TODO(RQ3 - later day): LightGBM quantile models (0.05 / 0.95).
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src import data_audit as da
from src.features import feature_set
from src.splits import assert_training_only, expanding_window_folds


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
# Revised 2026-11-14: expanding-window CV over a small predeclared grid inside 2015-2017
# (by target date), validation MAE, refit on 2015-2017. One selected set is persisted and
# reused by every experiment (RQ1 A/B/C, RQ2, RQ3 quantiles). 2018/2019 never touched.

def base_params(cfg: dict) -> dict:
    """Predeclared base parameters (every grid candidate starts from these)."""
    lgb = cfg["models"]["lightgbm"]
    if lgb.get("early_stopping"):
        raise ValueError("early stopping would consult held-out data; it is disabled by design")
    params = dict(lgb["params"])
    params["random_state"] = cfg["seed"]
    if params.get("bagging_seed") != cfg["seed"]:
        raise ValueError("models.lightgbm.params.bagging_seed must equal the top-level seed")
    return params


def candidate_params(cfg: dict) -> list[dict]:
    """All grid candidates, in predeclared (YAML) order; ties resolve to the earliest."""
    grid = cfg["tuning"]["grid"]
    keys = list(grid)
    return [{**base_params(cfg), **dict(zip(keys, values))}
            for values in itertools.product(*(grid[k] for k in keys))]


def _grid_hash(cfg: dict) -> str:
    blob = {"grid": cfg["tuning"]["grid"], "base": base_params(cfg),
            "folds": cfg["tuning"]["expanding_cv_folds"], "selection": cfg["tuning"]["selection"]}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()


FitPredict = Callable[[dict, pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray]


def lightgbm_fit_predict(params: dict, X_tr: pd.DataFrame, y_tr: pd.Series,
                         X_va: pd.DataFrame) -> np.ndarray:
    """Default fit/predict used by selection (point model, L2 objective)."""
    import lightgbm as lgb
    model = lgb.LGBMRegressor(objective="regression", **params)
    model.fit(X_tr, y_tr)
    return np.asarray(model.predict(X_va))


def selection_rows(ds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Rows the grid may be scored on: selection cities/horizons, split_main == train only."""
    sel = cfg["tuning"]["selection"]
    if sel["dataset"] != "forecast":
        raise ValueError("selection is defined on the protocol C forecasting table")
    cities = cfg["cities"][sel["cities"]]
    rows = ds.loc[ds["city"].astype(str).isin(cities) & (ds["split_main"] == "train")
                  & ds["horizon"].isin(sel["horizons"])]
    assert_training_only(rows["target_date"], cfg)
    return rows


def select_hyperparameters(ds: pd.DataFrame, cfg: dict,
                           fit_predict: FitPredict | None = None) -> tuple[dict, pd.DataFrame]:
    """Expanding-window CV over the predeclared grid. Returns (best params, CV table).

    Every fit and every prediction is guarded by `assert_training_only`. Score =
    validation MAE averaged over folds x horizons; ties -> earliest candidate.
    """
    fit_predict = fit_predict or lightgbm_fit_predict
    feats = feature_set(cfg, cfg["tuning"]["selection"]["feature_set"])
    rows = selection_rows(ds, cfg)
    folds = expanding_window_folds(rows, cfg)
    cands = candidate_params(cfg)
    grid_keys = list(cfg["tuning"]["grid"])
    records = []
    for ci, params in enumerate(cands):
        for h in cfg["tuning"]["selection"]["horizons"]:
            in_h = rows.index[rows["horizon"] == h]
            for fi, (tri, vai) in enumerate(folds):
                tr, va = rows.loc[tri.intersection(in_h)], rows.loc[vai.intersection(in_h)]
                assert_training_only(tr["target_date"], cfg)
                assert_training_only(va["target_date"], cfg)
                pred = np.asarray(fit_predict(params, tr[feats], tr["y_target"], va[feats]), dtype=float)
                mae = float(np.mean(np.abs(pred - va["y_target"].to_numpy())))
                records.append({"candidate": ci, **{k: params[k] for k in grid_keys},
                                "horizon": h, "fold": fi, "n_train": len(tr), "n_valid": len(va),
                                "valid_mae": mae})
    table = pd.DataFrame(records)
    score = table.groupby("candidate")["valid_mae"].mean()
    best = int(score.idxmin())                         # first minimum -> earliest candidate
    table["mean_valid_mae"] = table["candidate"].map(score)
    table["selected"] = table["candidate"] == best
    return dict(cands[best]), table


def save_selected_params(params: dict, cv_table: pd.DataFrame, cfg: dict, path=None):
    """Persist the selected set (+ provenance) so every experiment reuses the same values."""
    path = da.ROOT / cfg["tuning"]["selected_params_file"] if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = da.build_metadata(cv_table, path.name, cfg, "src.models.select_hyperparameters",
                             notes="Expanding-window CV inside 2015-2017 only.")
    payload = {"params": params, "grid_hash": _grid_hash(cfg), "provenance": meta}
    path.write_text(json.dumps(payload, indent=2, default=str))
    da.write_table(cv_table, path.stem + "_cv", cfg, "src.models.select_hyperparameters")
    return path


def lightgbm_params(cfg: dict, path=None) -> dict:
    """THE LightGBM parameters for every experiment. Takes no data.

    mode 'expanding_cv': the persisted selection (must match the current grid/base/folds
    and dataset hash, else it is stale). mode 'fixed': the base parameters.
    """
    mode = cfg["tuning"]["mode"]
    if mode == "fixed":
        return base_params(cfg)
    if mode != "expanding_cv":
        raise ValueError(f"unknown tuning.mode {mode!r}")
    path = da.ROOT / cfg["tuning"]["selected_params_file"] if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run hyperparameter selection (2015-2017 CV) first")
    payload = json.loads(path.read_text())
    if payload["grid_hash"] != _grid_hash(cfg):
        raise ValueError("selected parameters are stale: grid/base/folds/selection changed")
    if payload["provenance"]["dataset_sha256"] != cfg["data"]["sha256"]:
        raise ValueError("selected parameters were computed on a different dataset")
    params = dict(payload["params"])
    return params


def quantile_params(cfg: dict, alpha: float, path=None) -> dict:
    """RQ3 quantile models reuse the selected parameters; only objective/alpha change."""
    if alpha not in cfg["models"]["lightgbm"]["quantile_alphas"]:
        raise ValueError(f"alpha {alpha} not predeclared")
    return {**lightgbm_params(cfg, path), "objective": "quantile", "alpha": alpha}


def final_training_rows(ds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Refit data for the main scheme: every split_main == train row (2015-2017), guarded."""
    rows = ds.loc[ds["split_main"] == "train"]
    assert_training_only(rows["target_date"], cfg)
    return rows
