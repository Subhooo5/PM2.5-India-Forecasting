"""Conformal prediction intervals.

TODO(RQ3 - later day): CQR calibrated on 2018 (source-city and held-out-city calibration are distinct experiments).
TODO(RQ3 - later day): CQR+ACI, alpha=0.10, bounded alpha_t, per city x horizon, delayed feedback via a
pending-prediction queue (state updated only when the target is observed, h days later).
ACI gives long-run marginal coverage only; never claim conditional / per-regime guarantees.
"""
