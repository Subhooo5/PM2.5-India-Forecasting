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

import itertools

import pandas as pd

from src.splits import assert_training_only


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
