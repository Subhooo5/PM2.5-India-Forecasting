"""Hyperparameter-selection rule (frozen Day 2 task 1): training-only guard and the two
expanding-window folds by TARGET year. Selection itself is tested in tests/test_tuning.py."""
import inspect

import pandas as pd
import pytest

from src import models as M
from src.splits import (LeakageError, assert_training_only, expanding_window_folds,
                        split_ranges, year_span)


# ------------------------------------------------------------ guard
@pytest.mark.parametrize("bad_date", ["2018-01-01", "2018-06-30", "2019-03-01", "2020-01-02",
                                      "2014-12-31"])
def test_guard_rejects_calibration_test_and_out_of_range(cfg, bad_date):
    with pytest.raises(LeakageError):
        assert_training_only(["2016-05-01", bad_date], cfg)


def test_guard_accepts_training_years(cfg):
    assert_training_only(pd.date_range("2015-01-01", "2017-12-31"), cfg)


def test_guard_rejects_config_that_reaches_calibration(cfg):
    bad = {**cfg, "tuning": {**cfg["tuning"], "max_allowed_target_date": "2018-03-31"}}
    with pytest.raises(LeakageError):
        assert_training_only(["2016-01-01"], bad)


# ------------------------------------------------------------ folds
def test_configured_folds_are_the_two_declared_expanding_folds(cfg):
    t = cfg["tuning"]
    assert t["folds"] == [{"train_targets": "2015", "validate_targets": "2016"},
                          {"train_targets": "2015-2016", "validate_targets": "2017"}]
    assert t["assign_by"] == "target_date"
    assert t["max_allowed_target_date"] == "2017-12-31"
    assert t["horizons"] == [1, 3, 7] and t["feature_set"] == "causal_common"
    assert t["cities"] == "core" and t["eval_rows"] == "eval_mask"
    assert t["selection_metric"] == "mean_val_mae" and t["tie_break"] == "earliest_in_grid_order"


def test_year_span():
    assert year_span("2015") == (pd.Timestamp("2015-01-01"), pd.Timestamp("2015-12-31"))
    assert year_span("2015-2016") == (pd.Timestamp("2015-01-01"), pd.Timestamp("2016-12-31"))
    for bad in ("2016-2015", "15", "2015-2016-2017", "abcd"):
        with pytest.raises(ValueError):
            year_span(bad)


def test_expanding_folds_stay_in_training_years(ds, cfg):
    folds = expanding_window_folds(ds, cfg)
    assert len(folds) == 2
    cal_start = split_ranges(cfg, "main")["calibration"][0]
    prev_train = 0
    for tri, vai in folds:
        tr, va = ds.loc[tri, "target_date"], ds.loc[vai, "target_date"]
        assert len(tr) and len(va)
        assert tr.max() < va.min()                       # validation strictly later
        assert va.max() < cal_start and tr.max() < cal_start
        assert len(tri) > prev_train                     # expanding
        prev_train = len(tri)


@pytest.mark.parametrize("folds", [
    [{"train_targets": "2015-2016", "validate_targets": "2018"}],          # into calibration
    [{"train_targets": "2015-2016", "validate_targets": "2016"}],          # overlap
    [{"train_targets": "2016", "validate_targets": "2017"}],               # not expanding from start
    [{"train_targets": "2015-2016", "validate_targets": "2017"},
     {"train_targets": "2015", "validate_targets": "2016"}],               # shrinking
])
def test_bad_fold_definitions_rejected(ds, cfg, folds):
    with pytest.raises(LeakageError):
        expanding_window_folds(ds, {**cfg, "tuning": {**cfg["tuning"], "folds": folds}})


# ------------------------------------------------------------ refit + reuse
def test_refit_rows_are_exactly_2015_2017(ds, cfg):
    rows = M.final_training_rows(ds, cfg)
    assert rows["target_date"].min() >= pd.Timestamp("2015-01-01")
    assert rows["target_date"].max() <= pd.Timestamp("2017-12-31")
    assert len(rows) == int((ds["split_main"] == "train").sum())


def _with_selected(cfg, cand):
    return {**cfg, "lightgbm": {**cfg["lightgbm"], "selected": cand}}


def test_selected_params_are_reused_everywhere(cfg):
    cand = M.candidate_params(cfg)[3]
    c = _with_selected(cfg, cand)
    p = M.lightgbm_params(c)
    assert p == M.lightgbm_params(c) == M.merge_params(cfg, cand)
    assert p["num_leaves"] == cand["num_leaves"] and "candidate_id" not in p
    for alpha in cfg["lightgbm"]["quantile_alphas"]:
        q = M.quantile_params(c, alpha)
        assert {k: v for k, v in q.items() if k not in ("objective", "alpha")} == \
            {k: v for k, v in p.items() if k != "objective"}
        assert q["objective"] == "quantile" and q["alpha"] == alpha
    with pytest.raises(ValueError):
        M.quantile_params(c, 0.5)


def test_lightgbm_params_requires_selection(cfg):
    with pytest.raises(ValueError, match="selected"):
        M.lightgbm_params(_with_selected(cfg, None))


def test_lightgbm_params_takes_no_data():
    assert list(inspect.signature(M.lightgbm_params).parameters) == ["cfg"]
