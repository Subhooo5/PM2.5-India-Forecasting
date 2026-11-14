"""Chronological splits assigned by TARGET date (Day 1, item 7; used by RQ1-RQ3).

Main:   train 2015-2017 / calibration 2018 / test 2019.
Stress: train 2015-2018 / calibration 2019 / test 2020-01-01..2020-07-01,
        reported pre-lockdown (< 2020-03-25) vs lockdown separately.

Labels are derived ONLY from `target_date` (never `origin_date`), so an h=7 sample whose
origin is 2017-12-30 belongs to calibration because its target is 2018-01-06.
Samples whose target lies outside every range get the label "outside".

TODO(RQ1 - later day): protocol A random 80/20 split (intentionally invalid demonstration).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_ORDER = ("train", "calibration", "test")
OUTSIDE = "outside"


def split_ranges(cfg: dict, scheme: str) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Inclusive (lo, hi) target-date range per split; checks order and non-overlap."""
    sc = cfg["splits"][scheme]
    ranges = {k: (pd.Timestamp(sc[k][0]), pd.Timestamp(sc[k][1])) for k in SPLIT_ORDER}
    prev_hi = None
    for k in SPLIT_ORDER:
        lo, hi = ranges[k]
        if lo > hi or (prev_hi is not None and lo <= prev_hi):
            raise ValueError(f"split ranges for {scheme!r} overlap or are out of order at {k}")
        prev_hi = hi
    return ranges


def label_by_target_date(target_date: pd.Series, cfg: dict, scheme: str) -> pd.Series:
    """Split label for each sample, computed from its target date only."""
    td = pd.to_datetime(target_date)
    labels = np.full(len(td), OUTSIDE, dtype=object)
    for name, (lo, hi) in split_ranges(cfg, scheme).items():
        labels[((td >= lo) & (td <= hi)).to_numpy()] = name
    return pd.Series(labels, index=target_date.index, name=f"split_{scheme}")


def assign_splits(ds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add split_main, split_stress and stress_period columns (all from target_date)."""
    out = ds.copy()
    out["split_main"] = label_by_target_date(out["target_date"], cfg, "main")
    out["split_stress"] = label_by_target_date(out["target_date"], cfg, "stress")
    lock = pd.Timestamp(cfg["splits"]["stress"]["lockdown_start"])
    period = np.where(out["target_date"] < lock, "pre_lockdown", "lockdown")
    out["stress_period"] = np.where(out["split_stress"] == "test", period, "")
    for c in ("split_main", "split_stress", "stress_period"):
        out[c] = out[c].astype("category")
    return out
