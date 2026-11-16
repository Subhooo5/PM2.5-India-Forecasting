import copy

import numpy as np
import pandas as pd
import pytest

from src.splits import (OUTSIDE, LeakageError, assert_training_only, expanding_window_folds, label_by_target_date,
                        random_split_unique, split_ranges)


@pytest.mark.parametrize("scheme", ["main", "stress"])
def test_target_dates_inside_split_ranges(ds, cfg, scheme):
    col, ranges = f"split_{scheme}", split_ranges(cfg, scheme)
    inside_any = pd.Series(False, index=ds.index)
    for name, (lo, hi) in ranges.items():
        td = ds.loc[ds[col] == name, "target_date"]
        assert len(td) > 0 and td.between(lo, hi).all()
        inside_any |= ds["target_date"].between(lo, hi)
    assert (ds.loc[~inside_any, col] == OUTSIDE).all() and (ds.loc[inside_any, col] != OUTSIDE).all()
    assert (ds.loc[ds[col] == "train", "target_date"] < ranges["calibration"][0]).all()


def test_split_uses_target_not_origin_date(cfg, ds):
    origin = pd.Series(pd.to_datetime(["2017-12-30", "2018-12-28", "2017-12-31"]))
    target = origin + pd.to_timedelta(pd.Series([7, 7, 1]), unit="D")
    assert list(label_by_target_date(target, cfg, "main")) == ["calibration", "test", "calibration"]
    assert list(label_by_target_date(origin, cfg, "main")) == ["train", "calibration", "train"]
    cal = ds[ds["split_main"] == "calibration"]
    assert (cal["origin_date"] < pd.Timestamp("2018-01-01")).any()


def test_stress_period_labels(ds, cfg):
    lock = pd.Timestamp(cfg["splits"]["stress"]["lockdown_start"])
    test = ds[ds["split_stress"] == "test"]
    assert (test.loc[test["stress_period"] == "pre_lockdown", "target_date"] < lock).all()
    assert (test.loc[test["stress_period"] == "lockdown", "target_date"] >= lock).all()
    assert (ds.loc[ds["split_stress"] != "test", "stress_period"] == "").all()


def test_split_ranges_reject_overlap(cfg):
    bad = {**cfg, "splits": {"main": {"train": ["2015-01-01", "2018-01-05"], "calibration": ["2018-01-01", "2018-12-31"],
                                      "test": ["2019-01-01", "2019-12-31"]}}}
    with pytest.raises(ValueError):
        split_ranges(bad, "main")


def test_random_split_refuses_multi_horizon_table(ds):
    with pytest.raises(ValueError, match="unique"):
        random_split_unique(ds[["city", "target_date"]], 0.2, 42)


def test_random_split_is_stratified_deterministic_and_order_independent(same_day):
    units = same_day[["city", "target_date"]]
    a = random_split_unique(units, 0.2, 42)
    shuffled = units.sample(frac=1, random_state=1)
    pd.testing.assert_series_equal(a, random_split_unique(shuffled, 0.2, 42).reindex(units.index))
    assert not a.equals(random_split_unique(units, 0.2, 43))
    share = (a == "test").groupby(units["city"], observed=True).mean()
    assert share.between(0.199, 0.201).all()


def test_same_day_splits(same_day, cfg):
    years = same_day["target_date"].dt.year
    assert set(same_day.loc[years == 2018, "split_A"]) == {"reserved_calibration"}
    assert set(same_day.loc[years == 2018, "split_B"]) == {"reserved_calibration"}
    pool_a = same_day["split_A"].isin(["train", "test"])
    pool_b = same_day["split_B"].isin(["train", "test"])
    assert (pool_a == pool_b).all() and (pool_a == years.isin([2015, 2016, 2017, 2019])).all()
    assert set(years[same_day["split_B"] == "test"]) == {2019}
    assert set(years[same_day["split_B"] == "train"]) == {2015, 2016, 2017}


def test_training_only_guard(cfg):
    assert_training_only(pd.to_datetime(["2015-01-01", "2017-12-31"]), cfg)
    with pytest.raises(LeakageError):
        assert_training_only(pd.to_datetime(["2018-01-01"]), cfg)
    bad = copy.deepcopy(cfg)
    bad["tuning"]["max_allowed_target_date"] = "2018-06-30"
    with pytest.raises(LeakageError):
        assert_training_only(pd.to_datetime(["2015-01-01"]), bad)


def test_expanding_folds(ds, cfg):
    core = ds[ds["city"].isin(cfg["cities"]["core"]) & (ds["target_date"] <= "2017-12-31")]
    (tr1, va1), (tr2, va2) = expanding_window_folds(core, cfg)
    years = lambda idx: set(core.loc[idx, "target_date"].dt.year)
    assert (years(tr1), years(va1), years(tr2), years(va2)) == ({2015}, {2016}, {2015, 2016}, {2017})
    assert core.loc[tr2, "target_date"].max() < core.loc[va2, "target_date"].min()


@pytest.mark.parametrize("folds", [[{"train_targets": "2016", "validate_targets": "2017"}],
                                   [{"train_targets": "2015-2016", "validate_targets": "2016"}],
                                   [{"train_targets": "2015-2016", "validate_targets": "2017"},
                                    {"train_targets": "2015", "validate_targets": "2016"}],
                                   [{"train_targets": "2015-2017", "validate_targets": "2018"}]])
def test_expanding_folds_reject_invalid(ds, cfg, folds):
    bad = copy.deepcopy(cfg)
    bad["tuning"]["folds"] = folds
    with pytest.raises(LeakageError):
        expanding_window_folds(ds.iloc[:10], bad)


def test_split_label_counts_match_ranges(ds, cfg):
    for scheme in ("main", "stress"):
        recomputed = label_by_target_date(ds["target_date"], cfg, scheme)
        np.testing.assert_array_equal(recomputed.astype(str), ds[f"split_{scheme}"].astype(str))
