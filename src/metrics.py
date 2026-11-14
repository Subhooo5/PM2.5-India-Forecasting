"""Evaluation metrics (point metrics: Day 2; interval metrics and breakdowns: later days).

Definitions (config `metrics`):
  MAE  = mean |y - yhat|
  RMSE = sqrt(mean (y - yhat)^2)
  R2   = 1 - SSE/SST, SSE = sum (y - yhat)^2, SST = sum (y - mean(y))^2
Only observed targets are scored. Explicit edge cases:
  n == 0         -> mae = rmse = r2 = NaN, n = 0
  SST == 0       -> r2 = NaN (constant target: R2 undefined; sklearn would return 0 or 1)
  NaN prediction -> ValueError (a model must predict every observed-target row)

TODO(RQ1-RQ3 - later day): skill vs persistence; PICP, MPIW, Winkler score; breakdowns by
horizon, Indian season, top-10% days (per-city TRAINING-period threshold), city, held-out city.
"""
from __future__ import annotations

import numpy as np


def _clean(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Keep observed targets only; reject missing predictions on them."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    observed = ~np.isnan(yt)
    yt, yp = yt[observed], yp[observed]
    if np.isnan(yp).any():
        raise ValueError("prediction missing for an observed target")
    return yt, yp


def mae(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp))) if len(yt) else float("nan")


def rmse(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2))) if len(yt) else float("nan")


def r2(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    sst = float(np.sum((yt - yt.mean()) ** 2))
    if sst == 0.0:
        return float("nan")
    return float(1.0 - np.sum((yt - yp) ** 2) / sst)


def point_metrics(y_true, y_pred) -> dict:
    """{mae, rmse, r2, n} on observed targets."""
    yt, _ = _clean(y_true, y_pred)
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred),
            "r2": r2(y_true, y_pred), "n": int(len(yt))}
