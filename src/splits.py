from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_ORDER = ("train", "calibration", "test")
OUTSIDE = "outside"
RESERVED = "reserved_calibration"


class LeakageError(AssertionError):
    pass


def split_ranges(cfg: dict, scheme: str) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    sc = cfg["splits"][scheme]
    ranges = {k: (pd.Timestamp(sc[k][0]), pd.Timestamp(sc[k][1])) for k in SPLIT_ORDER}
    prev_hi = None
    for k in SPLIT_ORDER:
        lo, hi = ranges[k]
        if lo > hi or (prev_hi is not None and lo <= prev_hi):
            raise ValueError(f"split ranges for {scheme!r} overlap or are out of order at {k}")
        prev_hi = hi
    return ranges


def main_ranges(cfg: dict):
    r = split_ranges(cfg, "main")
    return r["train"], r["calibration"], r["test"]


def label_by_target_date(target_date: pd.Series, cfg: dict, scheme: str) -> pd.Series:
    td = pd.to_datetime(target_date)
    labels = np.full(len(td), OUTSIDE, dtype=object)
    for name, (lo, hi) in split_ranges(cfg, scheme).items():
        labels[((td >= lo) & (td <= hi)).to_numpy()] = name
    return pd.Series(labels, index=target_date.index, name=f"split_{scheme}")


def assign_splits(ds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = ds.copy()
    out["split_main"] = label_by_target_date(out["target_date"], cfg, "main")
    out["split_stress"] = label_by_target_date(out["target_date"], cfg, "stress")
    lock = pd.Timestamp(cfg["splits"]["stress"]["lockdown_start"])
    period = np.where(out["target_date"] < lock, "pre_lockdown", "lockdown")
    out["stress_period"] = np.where(out["split_stress"] == "test", period, "")
    for c in ("split_main", "split_stress", "stress_period"):
        out[c] = out[c].astype("category")
    return out


def random_split_unique(units: pd.DataFrame, test_frac: float, seed: int, stratify_by: str = "city",
                        key: tuple[str, str] = ("city", "target_date")) -> pd.Series:
    if units.duplicated(list(key)).any():
        raise ValueError("random split requires unique (city, target_date) rows; "
                         "do not random-split the multi-horizon table")
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    labels = pd.Series("train", index=units.index, dtype=object, name="split_random")
    for _, grp in units.sort_values([stratify_by, key[1]]).groupby(stratify_by, observed=True, sort=True):
        labels.loc[grp.index[rng.permutation(len(grp))[:int(round(test_frac * len(grp)))]]] = "test"
    return labels


def assign_same_day_splits(sd: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = sd.copy()
    pa = cfg["leakage"]["protocol_A"]
    pool = out["target_date"].dt.year.isin(pa["pool_target_years"])
    cal_lo, cal_hi = split_ranges(cfg, "main")["calibration"]
    reserved = out["target_date"].between(cal_lo, cal_hi)
    if (pool & reserved).any():
        raise LeakageError("protocol A pool overlaps the calibration year")
    a = pd.Series(OUTSIDE, index=out.index, dtype=object)
    a[reserved] = RESERVED
    a[pool] = random_split_unique(out.loc[pool], pa["test_frac"], cfg["seed"], pa["stratify_by"])
    out["split_A"] = a.astype("category")
    out["split_B"] = (label_by_target_date(out["target_date"], cfg, "main")
                      .replace({"calibration": RESERVED}).astype("category"))
    return out


def year_span(spec) -> tuple[pd.Timestamp, pd.Timestamp]:
    parts = str(spec).split("-")
    if len(parts) not in (1, 2) or not all(p.isdigit() and len(p) == 4 for p in parts) or parts[-1] < parts[0]:
        raise ValueError(f"bad year span {spec!r}")
    return pd.Timestamp(f"{parts[0]}-01-01"), pd.Timestamp(f"{parts[-1]}-12-31")


def assert_training_only(target_dates, cfg: dict) -> None:
    lo = pd.Timestamp(cfg["data"]["date_min"])
    hi = pd.Timestamp(cfg["tuning"]["max_allowed_target_date"])
    if hi >= split_ranges(cfg, "main")["calibration"][0]:
        raise LeakageError("tuning.max_allowed_target_date must be before the calibration year")
    td = pd.to_datetime(pd.Series(target_dates))
    bad = td[(td < lo) | (td > hi)]
    if len(bad):
        raise LeakageError(f"{len(bad)} target dates outside the training-only tuning range "
                           f"{lo.date()}..{hi.date()} (e.g. {bad.iloc[0].date()})")


def expanding_window_folds(ds: pd.DataFrame, cfg: dict) -> list[tuple[pd.Index, pd.Index]]:
    folds, prev_train_end = [], None
    for f in cfg["tuning"]["folds"]:
        tr_lo, tr_hi = year_span(f["train_targets"])
        va_lo, va_hi = year_span(f["validate_targets"])
        if tr_lo != pd.Timestamp(cfg["data"]["date_min"]):
            raise LeakageError("expanding window must start at the beginning of the data")
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
