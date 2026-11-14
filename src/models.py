"""Forecasting models.

Day 1, item 8: reference baselines (no fitting):
- persistence:     yhat[t+h] = y[t]        (column `pm25_origin`, always observed)
- seasonal naive:  yhat[t+h] = y[t+h-7]    (column `snaive_ref`, raw observed value only)
Both must be scored on the common evaluation sample `eval_mask` (see CLAUDE.md).

TODO(RQ1, RQ2 - later day): pooled LightGBM point model, one direct model per horizon h in {1,3,7}.
TODO(RQ3 - later day): LightGBM quantile models (0.05 / 0.95).
"""
from __future__ import annotations

import pandas as pd


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
