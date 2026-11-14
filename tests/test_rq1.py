"""RQ1 regression tests: same-day estimation table (A/B), unique city-date random split,
no target overlap between random train/test, chronological B, and the B-vs-C common set."""
import numpy as np
import pandas as pd
import pytest

from src import features as F
from src.splits import (OUTSIDE, RESERVED, assign_rq1_splits, random_split_unique,
                        rq1_bc_common_index, split_ranges)

KEY = ["city", "target_date"]


# ------------------------------------------------------------ same-day table
def test_one_row_per_city_target_date(same_day, raw, cfg):
    assert not same_day.duplicated(KEY).any()
    core = cfg["cities"]["core"]
    n_obs = int(raw.loc[raw["City"].isin(core), "PM2.5"].notna().sum())
    assert len(same_day) == n_obs                       # every observed core city-date, once
    assert set(same_day["city"].astype(str)) == set(core)


def test_same_day_columns(same_day, cfg):
    assert list(same_day.columns) == ["city", "target_date", *F.feature_set(cfg, "same_day"),
                                      "y_target", "split_A", "split_B"]
    assert "AQI_Bucket" not in same_day.columns
    assert not any(c.startswith("pm25_") for c in same_day.columns)   # no PM2.5 history
    assert same_day["y_target"].notna().all()


def test_same_day_values_match_raw(same_day, raw):
    m = same_day.assign(city=same_day["city"].astype(str)).merge(
        raw.rename(columns={"City": "city", "Date": "target_date"}), on=KEY, suffixes=("", "_raw"))
    assert len(m) == len(same_day)
    np.testing.assert_array_equal(m["y_target"].to_numpy(), m["PM2.5"].to_numpy())
    for c in ["AQI", "PM10", "NO", "CO", "Xylene"]:
        np.testing.assert_array_equal(m[c].to_numpy(), m[f"{c}_raw"].to_numpy())  # unfilled


# ------------------------------------------------------------ protocol A
def test_random_split_refuses_multi_horizon_table(ds):
    """Regression: the 3-horizon table repeats city-date targets, so it must be rejected."""
    core = ds[ds["city_role"] == "core"]
    assert core.duplicated(KEY).any()
    with pytest.raises(ValueError, match="unique"):
        random_split_unique(core, 0.2, seed=0)


def test_no_city_date_overlap_between_random_train_and_test(same_day):
    tr = pd.MultiIndex.from_frame(same_day.loc[same_day["split_A"] == "train", KEY])
    te = pd.MultiIndex.from_frame(same_day.loc[same_day["split_A"] == "test", KEY])
    assert len(tr) > 0 and len(te) > 0
    assert tr.intersection(te).empty
    assert tr.is_unique and te.is_unique


def test_random_split_is_80_20_per_city(same_day, cfg):
    frac = cfg["rq1"]["protocol_A"]["test_frac"]
    pool = same_day[same_day["split_A"].isin(["train", "test"])]
    for city, g in pool.groupby("city", observed=True):
        assert (g["split_A"] == "test").sum() == int(round(frac * len(g))), city


def test_A_pool_equals_B_train_plus_test_and_excludes_2018(same_day, cfg):
    a_pool = same_day["split_A"].isin(["train", "test"])
    b_pool = same_day["split_B"].isin(["train", "test"])
    assert (a_pool == b_pool).all()
    cal_lo, cal_hi = split_ranges(cfg, "main")["calibration"]
    in_2018 = same_day["target_date"].between(cal_lo, cal_hi)
    assert (same_day.loc[in_2018, "split_A"] == RESERVED).all()
    assert (same_day.loc[in_2018, "split_B"] == RESERVED).all()
    assert (same_day.loc[same_day["target_date"].dt.year == 2020, "split_A"] == OUTSIDE).all()


def test_random_split_deterministic_and_order_independent(same_day, cfg):
    base = same_day[["city", "target_date", "y_target"]]
    shuffled = base.sample(frac=1.0, random_state=123)
    a = assign_rq1_splits(base, cfg)["split_A"]
    b = assign_rq1_splits(shuffled, cfg)["split_A"].reindex(a.index)
    pd.testing.assert_series_equal(a, b)
    other = assign_rq1_splits(base, {**cfg, "seed": cfg["seed"] + 1})["split_A"]
    assert not a.equals(other)                          # the seed actually matters


# ------------------------------------------------------------ protocol B
def test_B_is_chronological(same_day, cfg):
    r = split_ranges(cfg, "main")
    tr = same_day.loc[same_day["split_B"] == "train", "target_date"]
    te = same_day.loc[same_day["split_B"] == "test", "target_date"]
    assert tr.between(*r["train"]).all() and te.between(*r["test"]).all()
    assert tr.max() < r["calibration"][0] and te.min() > r["calibration"][1]


# ------------------------------------------------------------ B vs C common set
@pytest.mark.parametrize("h", [1, 3, 7])
def test_bc_common_index(same_day, ds, h):
    idx = rq1_bc_common_index(same_day, ds, h)
    assert len(idx) > 0 and idx.is_unique
    dates = pd.DatetimeIndex(idx.get_level_values("target_date"))
    assert (dates.year == 2019).all()
    b = pd.MultiIndex.from_frame(same_day.loc[same_day["split_B"] == "test", KEY].astype({"city": str}))
    c_rows = ds[(ds["horizon"] == h) & (ds["split_main"] == "test") & ds["eval_mask"]]
    c = pd.MultiIndex.from_frame(c_rows[KEY].astype({"city": str}))
    assert idx.isin(b).all() and idx.isin(c).all()
    assert set(idx.get_level_values("city")) <= set(same_day["city"].astype(str))
