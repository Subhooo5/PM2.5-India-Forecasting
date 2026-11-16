from __future__ import annotations

import numpy as np


def _clean(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if np.isinf(yt).any():
        raise ValueError("infinite target value (targets must be finite or NaN=unobserved)")
    observed = ~np.isnan(yt)
    yt, yp = yt[observed], yp[observed]
    if not np.isfinite(yp).all():
        raise ValueError("prediction missing or non-finite for an observed target")
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
    yt, _ = _clean(y_true, y_pred)
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred),
            "r2": r2(y_true, y_pred), "n": int(len(yt))}
def _safe_ratio(num, den) -> np.ndarray:
    n, d = np.broadcast_arrays(np.asarray(num, dtype=float), np.asarray(den, dtype=float))
    out = np.full(n.shape, np.nan)
    ok = np.isfinite(d) & (d != 0)
    np.divide(n, d, out=out, where=ok)
    return out


def _shape_like(result: np.ndarray, *inputs):
    return float(result) if all(np.ndim(x) == 0 for x in inputs) else result


def skill_score(metric, reference):
    return _shape_like(1.0 - _safe_ratio(metric, reference), metric, reference)


def inflation_pct(error_left, error_right):
    return _shape_like(100.0 * (1.0 - _safe_ratio(error_left, error_right)),
                       error_left, error_right)


def relative_change_pct(value, reference):
    return _shape_like(100.0 * (_safe_ratio(value, reference) - 1.0), value, reference)
def interval_metrics(y_true, lower, upper, alpha: float) -> dict:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    y, lo = _clean(y_true, lower)
    _, hi = _clean(y_true, upper)
    if len(y) == 0:
        return {"picp": float("nan"), "mpiw": float("nan"), "winkler": float("nan"), "n": 0}
    width = hi - lo
    w = np.where(y < lo, width + (2 / alpha) * (lo - y),
                 np.where(y > hi, width + (2 / alpha) * (y - hi), width))
    return {"picp": float(np.mean((y >= lo) & (y <= hi))), "mpiw": float(np.mean(width)),
            "winkler": float(np.mean(w)), "n": int(len(y))}
