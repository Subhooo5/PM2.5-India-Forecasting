"""Hyperparameter-selection rule (revised 2026-11-14): expanding-window CV over a predeclared
grid inside 2015-2017 only, validation MAE, one selected set reused everywhere."""
import inspect

import numpy as np
import pandas as pd
import pytest

from src import models as M
from src.splits import LeakageError, assert_training_only, expanding_window_folds, split_ranges


# ------------------------------------------------------------ guard + folds
@pytest.mark.parametrize("bad_date", ["2018-01-01", "2018-06-30", "2019-03-01", "2020-01-02",
                                      "2014-12-31"])
def test_guard_rejects_calibration_test_and_out_of_range(cfg, bad_date):
    with pytest.raises(LeakageError):
        assert_training_only(["2016-05-01", bad_date], cfg)


def test_guard_accepts_training_years(cfg):
    assert_training_only(pd.date_range("2015-01-01", "2017-12-31"), cfg)


def test_guard_rejects_config_that_reaches_calibration(cfg):
    bad = {**cfg, "tuning": {**cfg["tuning"], "allowed_target_range": ["2015-01-01", "2018-03-31"]}}
    with pytest.raises(LeakageError):
        assert_training_only(["2016-01-01"], bad)


def test_configured_folds_are_the_two_declared_expanding_folds(cfg):
    assert cfg["tuning"]["mode"] == "expanding_cv"
    assert cfg["tuning"]["expanding_cv_folds"] == [
        {"train": ["2015-01-01", "2015-12-31"], "valid": ["2016-01-01", "2016-12-31"]},
        {"train": ["2015-01-01", "2016-12-31"], "valid": ["2017-01-01", "2017-12-31"]}]
    assert cfg["tuning"]["metric"] == "mae"


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
    [{"train": ["2015-01-01", "2016-12-31"], "valid": ["2017-01-01", "2018-06-30"]}],  # into cal
    [{"train": ["2015-01-01", "2016-12-31"], "valid": ["2016-06-01", "2016-12-31"]}],  # overlap
    [{"train": ["2015-06-01", "2015-12-31"], "valid": ["2016-01-01", "2016-12-31"]}],  # not expanding
    [{"train": ["2015-01-01", "2016-12-31"], "valid": ["2017-01-01", "2017-12-31"]},
     {"train": ["2015-01-01", "2015-12-31"], "valid": ["2016-01-01", "2016-12-31"]}],  # shrinking
])
def test_bad_fold_definitions_rejected(ds, cfg, folds):
    with pytest.raises(LeakageError):
        expanding_window_folds(ds, {**cfg, "tuning": {**cfg["tuning"], "expanding_cv_folds": folds}})


# ------------------------------------------------------------ selection (dummy model, no training)
class SpyFitPredict:
    """Stand-in for LightGBM: records every row it sees; error = num_leaves exactly."""

    def __init__(self, ds):
        self.y = ds["y_target"]
        self.seen_index = pd.Index([])
        self.calls = 0

    def __call__(self, params, X_tr, y_tr, X_va):
        self.calls += 1
        self.seen_index = self.seen_index.union(X_tr.index).union(X_va.index)
        return self.y.loc[X_va.index].to_numpy() + params["num_leaves"]


def test_selection_only_ever_sees_2015_2017_training_rows(ds, cfg):
    spy = SpyFitPredict(ds)
    M.select_hyperparameters(ds, cfg, fit_predict=spy)
    seen = ds.loc[spy.seen_index]
    assert len(seen) > 0
    assert seen["target_date"].max() < pd.Timestamp("2018-01-01")      # no calibration/test
    assert (seen["split_main"] == "train").all()
    assert set(seen["city"].astype(str)) <= set(cfg["cities"]["core"])
    n_cand = len(M.candidate_params(cfg))
    assert spy.calls == n_cand * len(cfg["tuning"]["selection"]["horizons"]) * 2


def test_selection_picks_min_mean_validation_mae(ds, cfg):
    best, table = M.select_hyperparameters(ds, cfg, fit_predict=SpyFitPredict(ds))
    assert best["num_leaves"] == min(cfg["tuning"]["grid"]["num_leaves"])
    # tie on min_child_samples -> earliest candidate in predeclared order
    assert best["min_child_samples"] == cfg["tuning"]["grid"]["min_child_samples"][0]
    assert np.allclose(table["valid_mae"], table["num_leaves"])
    sel = table[table["selected"]]
    assert sel["mean_valid_mae"].iloc[0] == table["mean_valid_mae"].min()
    assert set(table["fold"]) == {0, 1} and set(table["horizon"]) == {1, 3, 7}


def test_selection_ignores_rows_outside_training_even_if_present(ds, cfg):
    """Feeding the full table (incl. 2018-2020) must give the identical result."""
    only_train = ds[ds["split_main"] == "train"]
    a, ta = M.select_hyperparameters(ds, cfg, fit_predict=SpyFitPredict(ds))
    b, tb = M.select_hyperparameters(only_train, cfg, fit_predict=SpyFitPredict(ds))
    assert a == b
    pd.testing.assert_frame_equal(ta, tb)


def test_refit_rows_are_exactly_2015_2017(ds, cfg):
    rows = M.final_training_rows(ds, cfg)
    assert rows["target_date"].min() >= pd.Timestamp("2015-01-01")
    assert rows["target_date"].max() <= pd.Timestamp("2017-12-31")
    assert len(rows) == int((ds["split_main"] == "train").sum())


# ------------------------------------------------------------ reuse of the selected set
def test_selected_params_persist_and_are_reused(ds, cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(M.da, "write_table", lambda *a, **k: None)   # keep outputs/ untouched
    best, table = M.select_hyperparameters(ds, cfg, fit_predict=SpyFitPredict(ds))
    path = M.save_selected_params(best, table, cfg, tmp_path / "sel.json")
    p1, p2 = M.lightgbm_params(cfg, path), M.lightgbm_params(cfg, path)
    assert p1 == p2 == best
    for alpha in cfg["models"]["lightgbm"]["quantile_alphas"]:
        q = M.quantile_params(cfg, alpha, path)
        assert {k: v for k, v in q.items() if k not in ("objective", "alpha")} == best
        assert q["objective"] == "quantile" and q["alpha"] == alpha


def test_stale_or_missing_selection_is_rejected(ds, cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(M.da, "write_table", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError):
        M.lightgbm_params(cfg, tmp_path / "missing.json")
    best, table = M.select_hyperparameters(ds, cfg, fit_predict=SpyFitPredict(ds))
    path = M.save_selected_params(best, table, cfg, tmp_path / "sel.json")
    changed = {**cfg, "tuning": {**cfg["tuning"], "grid": {"num_leaves": [7], "min_child_samples": [20]}}}
    with pytest.raises(ValueError, match="stale"):
        M.lightgbm_params(changed, path)


def test_lightgbm_params_takes_no_data():
    assert list(inspect.signature(M.lightgbm_params).parameters) == ["cfg", "path"]
