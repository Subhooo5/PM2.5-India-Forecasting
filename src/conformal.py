from __future__ import annotations

import math
from fractions import Fraction

import numpy as np


def conformity_scores(y, lo, hi) -> np.ndarray:
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError("y, lo, hi must have the same shape")
    if not (np.isfinite(y).all() and np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError("calibration arrays must be finite (use observed eval_mask rows)")
    return np.maximum(lo - y, y - hi)


def conformal_quantile(scores, alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=float))
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if s.ndim != 1 or s.size == 0 or not np.isfinite(s).all():
        raise ValueError("scores must be a non-empty, one-dimensional, finite array")
    k = math.ceil((len(s) + 1) * (1 - Fraction(alpha).limit_denominator(10**6)))
    return float(s[k - 1]) if k <= len(s) else float("inf")


def cqr_interval(lo, hi, q_hat):
    lo, hi, q = (np.asarray(a, dtype=float) for a in (lo, hi, q_hat))
    return lo - q, hi + q


def aci_stream(origin, target, y, lo, hi, cal_scores, alpha: float, gamma: float, bounds) -> dict:
    origin, target = (np.asarray(a, dtype="datetime64[ns]") for a in (origin, target))
    y, lo, hi, cal = (np.asarray(a, dtype=float) for a in (y, lo, hi, cal_scores))
    b_lo, b_hi = bounds
    for ok, msg in (
            (lambda: origin.ndim == target.ndim == y.ndim == lo.ndim == hi.ndim == 1, "arrays must be 1-D"),
            (lambda: len({len(a) for a in (origin, target, y, lo, hi)}) == 1, "arrays differ in length"),
            (lambda: not np.isinf(y).any(), "y contains +/-inf (NaN = unobserved is allowed)"),
            (lambda: np.isfinite(lo).all() and np.isfinite(hi).all(), "lo / hi must be finite"),
            (lambda: not (np.isnat(origin).any() or np.isnat(target).any()), "NaT in origin/target"),
            (lambda: (np.diff(origin) > np.timedelta64(0)).all(), "origins not strictly increasing"),
            (lambda: (np.diff(target) > np.timedelta64(0)).all(), "targets not strictly increasing"),
            (lambda: (target > origin).all(), "a target is not after its origin"),
            (lambda: np.isfinite(gamma) and gamma > 0, "gamma must be finite and > 0"),
            (lambda: 0 < b_lo < b_hi < 1 and b_lo <= alpha <= b_hi, "need 0 < lo <= alpha <= hi < 1, lo < hi"),
            (lambda: cal.ndim == 1 and np.isfinite(cal).all(), "cal_scores must be finite and 1-D"),
            (lambda: len(cal) >= 1 / b_lo - 1, "cal_scores too short for the lower bound (q_hat = inf)")):
        if not ok():
            raise ValueError(f"aci_stream: {msg}")
    n, a, pending = len(y), alpha, []
    out = {k: np.full(n, np.nan) for k in ("alpha_issue", "lo", "hi", "miss")}
    out["update_origin"] = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    out["alpha_after"] = np.full(n, np.nan)
    for i in range(n):
        due = sorted((j for j in pending if target[j] <= origin[i]), key=lambda j: (target[j], j))
        pending = [j for j in pending if target[j] > origin[i]]
        for j in due:
            if not np.isnan(y[j]):
                a = min(max(a + gamma * (alpha - float(out["miss"][j])), bounds[0]), bounds[1])
                out["update_origin"][j], out["alpha_after"][j] = origin[i], a
        q = conformal_quantile(cal, a)
        out["alpha_issue"][i], out["lo"][i], out["hi"][i] = a, lo[i] - q, hi[i] + q
        if not np.isnan(y[i]):
            out["miss"][i] = float(not out["lo"][i] <= y[i] <= out["hi"][i])
        pending.append(i)
    return out
