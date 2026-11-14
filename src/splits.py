"""Chronological splits assigned by TARGET date (Day 1, item 7; used by RQ1-RQ3).

Main:   train 2015-2017 / calibration 2018 / test 2019.
Stress: train 2015-2018 / calibration 2019 / test 2020-01-01..2020-07-01,
        reported pre-lockdown (< 2020-03-25) vs lockdown separately.

Labels are derived ONLY from `target_date` (never `origin_date`), so an h=7 sample whose
origin is 2017-12-30 belongs to calibration because its target is 2018-01-06.
Samples whose target lies outside every range get the label "outside".

Also here (pre-Day-2, no training):
- RQ1 splits on the same-day table: A = random 80/20 over UNIQUE (city, target_date),
  stratified by city, pool 2015-2017 + 2019; B = chronological. 2018 reserved.
- `rq1_bc_common_index`: common 2019 (city, target_date) set for scoring B vs C.
- `assert_training_only` / `expanding_window_folds`: the hyperparameter-selection rule.
- `loco_partition`: RQ2 leave-one-city-out with zero held-out-city rows in training.
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


# ------------------------------------------------------------------ RQ1 (A / B)
RESERVED = "reserved_calibration"


class LeakageError(AssertionError):
    """Raised when data outside the permitted period/partition would be used."""


def random_split_unique(units: pd.DataFrame, test_frac: float, seed: int,
                        stratify_by: str = "city",
                        key: tuple[str, str] = ("city", "target_date")) -> pd.Series:
    """Random train/test labels at the UNIQUE (city, target_date) level, stratified by city.

    Refuses input with duplicate keys (e.g. the three-horizon long table), because the same
    city-date target would otherwise land in both train and test. Deterministic for a seed
    and independent of input row order.
    """
    if units.duplicated(list(key)).any():
        raise ValueError("random split requires unique (city, target_date) rows; "
                         "do not random-split the multi-horizon table")
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    labels = pd.Series("train", index=units.index, dtype=object, name="split_random")
    ordered = units.sort_values([stratify_by, key[1]])
    for _, grp in ordered.groupby(stratify_by, observed=True, sort=True):
        n_test = int(round(test_frac * len(grp)))
        pick = rng.permutation(len(grp))[:n_test]
        labels.loc[grp.index[pick]] = "test"
    return labels


def assign_rq1_splits(sd: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add split_A (random) and split_B (chronological) to the same-day table."""
    out = sd.copy()
    years = out["target_date"].dt.year
    pool = years.isin(cfg["rq1"]["protocol_A"]["pool_target_years"])
    cal_lo, cal_hi = split_ranges(cfg, "main")["calibration"]
    reserved = out["target_date"].between(cal_lo, cal_hi)
    if (pool & reserved).any():
        raise LeakageError("protocol A pool overlaps the calibration year")

    a = pd.Series(OUTSIDE, index=out.index, dtype=object)
    a[reserved] = RESERVED
    pa = cfg["rq1"]["protocol_A"]
    a[pool] = random_split_unique(out.loc[pool], pa["test_frac"], cfg["seed"], pa["stratify_by"])
    b = label_by_target_date(out["target_date"], cfg, "main").replace({"calibration": RESERVED})
    out["split_A"] = a.astype("category")
    out["split_B"] = b.astype("category")
    return out


def rq1_bc_common_index(same_day: pd.DataFrame, forecast: pd.DataFrame, h: int) -> pd.MultiIndex:
    """(city, target_date) pairs on which B and C are scored together for horizon h:
    B test rows (2019) AND C test rows with eval_mask at horizon h."""
    b = same_day.loc[same_day["split_B"] == "test", ["city", "target_date"]]
    c = forecast.loc[(forecast["split_main"] == "test") & forecast["eval_mask"]
                     & (forecast["horizon"] == h), ["city", "target_date"]]
    bi = pd.MultiIndex.from_frame(b.astype({"city": str}))
    ci = pd.MultiIndex.from_frame(c.astype({"city": str}))
    return bi.intersection(ci).sort_values()


# ------------------------------------------------------------------ tuning rule
def assert_training_only(target_dates, cfg: dict) -> None:
    """Raise LeakageError if any date lies outside tuning.allowed_target_range (2015-2017).

    Every hyperparameter/model-selection step must call this on ALL data it touches.
    """
    lo, hi = (pd.Timestamp(x) for x in cfg["tuning"]["allowed_target_range"])
    cal_start = split_ranges(cfg, "main")["calibration"][0]
    if hi >= cal_start:
        raise LeakageError("tuning range must end before the calibration year")
    td = pd.to_datetime(pd.Series(target_dates))
    bad = td[(td < lo) | (td > hi)]
    if len(bad):
        raise LeakageError(f"{len(bad)} target dates outside the training-only tuning range "
                           f"{lo.date()}..{hi.date()} (e.g. {bad.iloc[0].date()})")


def expanding_window_folds(ds: pd.DataFrame, cfg: dict) -> list[tuple[pd.Index, pd.Index]]:
    """Expanding-window (train, valid) row-index pairs by target date, all inside 2015-2017."""
    folds, prev_train_end = [], None
    lo_allowed = pd.Timestamp(cfg["tuning"]["allowed_target_range"][0])
    for f in cfg["tuning"]["expanding_cv_folds"]:
        tr_lo, tr_hi = pd.Timestamp(f["train"][0]), pd.Timestamp(f["train"][1])
        va_lo, va_hi = pd.Timestamp(f["valid"][0]), pd.Timestamp(f["valid"][1])
        if tr_lo != lo_allowed:
            raise LeakageError("expanding window must start at the beginning of the training range")
        if not tr_hi < va_lo:
            raise LeakageError("validation must come strictly after training")
        if prev_train_end is not None and tr_hi <= prev_train_end:
            raise LeakageError("training window must expand fold by fold")
        assert_training_only([tr_lo, tr_hi, va_lo, va_hi], cfg)
        prev_train_end = tr_hi
        tri = ds.index[ds["target_date"].between(tr_lo, tr_hi)]
        vai = ds.index[ds["target_date"].between(va_lo, va_hi)]
        assert_training_only(ds.loc[tri.union(vai), "target_date"], cfg)
        folds.append((tri, vai))
    return folds


# ------------------------------------------------------------------ RQ2 LOCO
def loco_partition(ds: pd.DataFrame, held_out: str, cfg: dict,
                   feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-city-out on the forecasting table (main scheme).

    train = other core cities, split_main == 'train'; test = held-out city, split_main ==
    'test' & eval_mask. The feature set must be city-ID-free (zero-shot).
    """
    if "city" in feature_cols:
        raise ValueError("LOCO uses the city-ID-free feature set; city-aware is a pooled-only ablation")
    core = list(cfg["cities"]["core"])
    if held_out not in core:
        raise ValueError(f"{held_out!r} is not a core city")
    city = ds["city"].astype(str)
    train = ds.loc[city.isin(core) & (city != held_out) & (ds["split_main"] == "train")]
    test = ds.loc[(city == held_out) & (ds["split_main"] == "test") & ds["eval_mask"]]
    if (train["city"].astype(str) == held_out).any():
        raise LeakageError("held-out city leaked into LOCO training rows")
    return train, test
