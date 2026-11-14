"""Day 1 item 7: target-date-based chronological splits.

Required checks covered here:
 (b) every sample's target date lies inside its split's date range
 (c) no train sample has a target date >= the calibration start
"""
import pandas as pd
import pytest

from src.splits import OUTSIDE, label_by_target_date, split_ranges


@pytest.mark.parametrize("scheme", ["main", "stress"])
def test_target_dates_inside_split_ranges(ds, cfg, scheme):
    col = f"split_{scheme}"
    ranges = split_ranges(cfg, scheme)
    for name, (lo, hi) in ranges.items():
        td = ds.loc[ds[col] == name, "target_date"]
        assert len(td) > 0
        assert td.between(lo, hi).all(), f"{scheme}/{name} has target dates outside {lo}..{hi}"
    inside_any = pd.Series(False, index=ds.index)
    for lo, hi in ranges.values():
        inside_any |= ds["target_date"].between(lo, hi)
    assert (ds.loc[~inside_any, col] == OUTSIDE).all()
    assert (ds.loc[inside_any, col] != OUTSIDE).all()


@pytest.mark.parametrize("scheme", ["main", "stress"])
def test_no_train_target_at_or_after_calibration_start(ds, cfg, scheme):
    cal_start = split_ranges(cfg, scheme)["calibration"][0]
    test_start = split_ranges(cfg, scheme)["test"][0]
    col = f"split_{scheme}"
    assert (ds.loc[ds[col] == "train", "target_date"] < cal_start).all()
    assert (ds.loc[ds[col] == "calibration", "target_date"] < test_start).all()


def test_split_uses_target_not_origin_date(cfg):
    """h=7 sample with origin 2017-12-30 has target 2018-01-06 -> calibration, not train."""
    origin = pd.Series(pd.to_datetime(["2017-12-30", "2018-12-28", "2017-12-31"]))
    h = pd.Series([7, 7, 1])
    target = origin + pd.to_timedelta(h, unit="D")
    assert list(label_by_target_date(target, cfg, "main")) == ["calibration", "test", "calibration"]
    assert list(label_by_target_date(origin, cfg, "main")) == ["train", "calibration", "train"]


def test_boundary_straddling_samples_exist_and_are_labelled_by_target(ds):
    """Real data: some calibration samples have origins in 2017 (h > 1 near New Year)."""
    cal = ds[ds["split_main"] == "calibration"]
    straddle = cal[cal["origin_date"] < pd.Timestamp("2018-01-01")]
    assert len(straddle) > 0
    assert (straddle["target_date"] >= pd.Timestamp("2018-01-01")).all()


def test_stress_period_labels(ds, cfg):
    lock = pd.Timestamp(cfg["splits"]["stress"]["lockdown_start"])
    test = ds[ds["split_stress"] == "test"]
    assert (test.loc[test["stress_period"] == "pre_lockdown", "target_date"] < lock).all()
    assert (test.loc[test["stress_period"] == "lockdown", "target_date"] >= lock).all()
    assert (ds.loc[ds["split_stress"] != "test", "stress_period"] == "").all()
    assert test["target_date"].max() <= pd.Timestamp("2020-07-01")


def test_split_ranges_reject_overlap(cfg):
    bad = {**cfg, "splits": {"main": {"train": ["2015-01-01", "2018-01-05"],
                                      "calibration": ["2018-01-01", "2018-12-31"],
                                      "test": ["2019-01-01", "2019-12-31"]}}}
    with pytest.raises(ValueError):
        split_ranges(bad, "main")
