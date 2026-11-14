"""Day 2 task 2: expanding-window hyperparameter selection (run_tuning).

Machinery tests use a FAST copy of the config (n_estimators=20, outputs in tmp) so they never
overwrite or re-run the official tuning; the official artifacts are checked separately below.
"""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src import data_audit as da
from src import experiments as E
from src.models import candidate_params
from src.splits import LeakageError, expanding_window_folds, year_span

GUARD = pd.Timestamp("2017-12-31")


@pytest.fixture(scope="module")
def fast_cfg(cfg, tmp_path_factory):
    c = copy.deepcopy(cfg)
    c["lightgbm"]["base"]["n_estimators"] = 20
    out = tmp_path_factory.mktemp("tuning")
    c["tuning"]["cv_results_file"] = str(out / "tuning_cv_results.csv")
    c["tuning"]["selected_params_file"] = str(out / "selected_params.json")
    return c


@pytest.fixture(scope="module")
def clean_run(fast_cfg):
    return E.run_tuning(fast_cfg)


# ------------------------------------------------------------ (a) refuses 2018+ rows
def test_tune_frame_raises_on_any_2018_row(fast_cfg, clean_run):
    frame = clean_run["frame"]
    leak = frame.iloc[[0]].assign(target_date=pd.Timestamp("2018-01-01"))
    with pytest.raises(LeakageError):
        E.tune_frame(pd.concat([frame, leak], ignore_index=True), fast_cfg)


def test_tuning_frame_never_contains_2018_or_later(clean_run, cfg):
    f = clean_run["frame"]
    assert f["target_date"].max() <= GUARD
    assert set(f["city"].astype(str)) == set(cfg["cities"]["core"])
    assert set(f["horizon"]) == {1, 3, 7}
    assert clean_run["payload"]["tuning_frame"]["max_target_date"] <= "2017-12-31"


# ------------------------------------------------------------ (b) poisoned data
@pytest.fixture(scope="module")
def poisoned_path(cfg, tmp_path_factory):
    real = pd.read_parquet(da.ROOT / cfg["paths"]["processed_dataset"])
    src = real[(real["city"] == "Delhi") & (real["target_date"].dt.year == 2017)].head(400).copy()
    fakes = []
    for year, label in ((2018, "calibration"), (2019, "test")):
        f = src.copy()
        f["target_date"] = f["target_date"] + pd.DateOffset(years=year - 2017)
        f["origin_date"] = f["origin_date"] + pd.DateOffset(years=year - 2017)
        num = f.select_dtypes("float").columns
        f[num] = 1e9                                   # absurd PM2.5 and features
        f["split_main"] = pd.Categorical(["train"] * len(f), categories=real["split_main"].cat.categories)
        fakes.append(f)                                 # mislabelled as train on purpose
    poisoned = pd.concat([real, *fakes], ignore_index=True)
    assert (poisoned["target_date"] >= "2018-01-01").sum() > len(real[real["target_date"] >= "2018-01-01"])
    path = tmp_path_factory.mktemp("poison") / "forecast_dataset.parquet"
    poisoned.to_parquet(path, index=False)
    return path


def test_poisoned_rows_do_not_change_tuning(fast_cfg, clean_run, poisoned_path):
    res = E.run_tuning(fast_cfg, data_path=poisoned_path)
    assert res["frame"]["target_date"].max() <= GUARD
    assert not (res["frame"].select_dtypes("float") >= 1e9).any().any()
    pd.testing.assert_frame_equal(res["cv"], clean_run["cv"])
    assert res["selected"] == clean_run["selected"]


# ------------------------------------------------------------ (c) (d) folds
def test_fold_dates(clean_run, fast_cfg):
    frame = clean_run["frame"]
    folds = expanding_window_folds(frame, fast_cfg)
    for (tri, vai), spec in zip(folds, fast_cfg["tuning"]["folds"]):
        va_lo, va_hi = year_span(spec["validate_targets"])
        tr_dates, va_dates = frame.loc[tri, "target_date"], frame.loc[vai, "target_date"]
        assert (tr_dates < va_lo).all()                          # strictly before validation
        assert va_dates.between(va_lo, va_hi).all()              # inside the validation year
        assert len(tri) > 0 and len(vai) > 0


def test_no_row_in_both_train_and_validation(clean_run, fast_cfg):
    frame = clean_run["frame"]
    for tri, vai in expanding_window_folds(frame, fast_cfg):
        assert tri.intersection(vai).empty
        tr_keys = pd.MultiIndex.from_frame(frame.loc[tri, ["city", "origin_date", "horizon"]])
        va_keys = pd.MultiIndex.from_frame(frame.loc[vai, ["city", "origin_date", "horizon"]])
        assert tr_keys.intersection(va_keys).empty


# ------------------------------------------------------------ (e) candidate list
def test_candidates_exactly_six_in_order(cfg, clean_run):
    ids = [c["candidate_id"] for c in candidate_params(cfg)]
    assert ids == ["c1", "c2", "c3", "c4", "c5", "c6"]
    assert [(c["num_leaves"], c["min_child_samples"]) for c in candidate_params(cfg)] == \
        [(15, 20), (15, 50), (31, 20), (31, 50), (63, 20), (63, 50)]
    cv = clean_run["cv"]
    assert list(cv.columns) == E.CV_COLUMNS
    assert len(cv) == 6 * 2 * 3
    assert list(clean_run["ranking"]["candidate_id"]) == ids


# ------------------------------------------------------------ (f) argmin + tie-break
def _synthetic_cv(cfg, means):
    rows = []
    for cand in candidate_params(cfg):
        for fold in (1, 2):
            for h in (1, 3, 7):
                rows.append({"candidate_id": cand["candidate_id"], "num_leaves": cand["num_leaves"],
                             "min_child_samples": cand["min_child_samples"], "fold": fold,
                             "horizon": h, "n_train": 1, "n_val": 1,
                             "mae": means[cand["candidate_id"]], "rmse": 0.0, "r2": 0.0})
    return pd.DataFrame(rows, columns=E.CV_COLUMNS)


def test_selection_is_argmin(cfg):
    means = {"c1": 9, "c2": 8, "c3": 7, "c4": 3, "c5": 5, "c6": 6}
    sel, ranking = E.select_candidate(_synthetic_cv(cfg, means), cfg)
    assert sel["candidate_id"] == "c4"
    assert ranking.loc[ranking["selected"], "candidate_id"].tolist() == ["c4"]
    assert ranking.set_index("candidate_id")["rank"].to_dict() == \
        {"c4": 1, "c5": 2, "c6": 3, "c3": 4, "c2": 5, "c1": 6}


def test_tie_break_is_earliest_in_grid_order(cfg):
    means = {"c1": 9, "c2": 4, "c3": 7, "c4": 4, "c5": 4, "c6": 6}
    sel, ranking = E.select_candidate(_synthetic_cv(cfg, means), cfg)
    assert sel["candidate_id"] == "c2"
    r = ranking.set_index("candidate_id")["rank"]
    assert r["c2"] < r["c4"] < r["c5"]


def test_selection_requires_complete_score_table(cfg):
    cv = _synthetic_cv(cfg, {f"c{i}": i for i in range(1, 7)})
    with pytest.raises(ValueError):
        E.select_candidate(cv.iloc[1:], cfg)


# ------------------------------------------------------------ (g) reproducibility
def test_tuning_is_reproducible(fast_cfg, clean_run):
    again = E.run_tuning(fast_cfg)
    pd.testing.assert_frame_equal(again["cv"], clean_run["cv"], check_exact=True)
    pd.testing.assert_frame_equal(again["ranking"], clean_run["ranking"], check_exact=True)
    assert again["selected"] == clean_run["selected"]


def test_selected_params_json_contents(clean_run, cfg):
    p = json.loads(clean_run["paths"][1].read_text())
    for key in ("selected_candidate_id", "selected_structural_params", "full_params", "ranking",
                "config_hash", "dataset_sha256", "kaggle_version", "versions", "seed", "created_utc"):
        assert key in p, key
    assert p["dataset_sha256"] == cfg["data"]["sha256"] and p["kaggle_version"] == 12
    assert set(p["versions"]) == {"lightgbm", "numpy", "pandas"}
    assert len(p["ranking"]) == 6
    assert {"mean_val_mae", "mean_mae_h1", "mean_mae_h3", "mean_mae_h7"} <= set(p["ranking"][0])
    assert p["full_params"]["subsample_freq"] == 1 and "early_stopping" not in p["full_params"]


# ------------------------------------------------------------ official run artifacts
@pytest.fixture(scope="module")
def official(cfg):
    sel = da.ROOT / cfg["tuning"]["selected_params_file"]
    cvf = da.ROOT / cfg["tuning"]["cv_results_file"]
    return json.loads(sel.read_text()), pd.read_csv(cvf)


def test_official_run_used_only_training_years(official):
    payload, cv = official
    assert payload["tuning_frame"]["max_target_date"] <= "2017-12-31"
    assert list(cv.columns) == E.CV_COLUMNS and len(cv) == 36
    assert set(cv["fold"]) == {1, 2} and set(cv["horizon"]) == {1, 3, 7}


def test_yaml_selected_matches_official_selection(cfg, official):
    payload, cv = official
    sel = cfg["lightgbm"]["selected"]
    assert sel["candidate_id"] == payload["selected_candidate_id"]
    assert {"num_leaves": sel["num_leaves"], "min_child_samples": sel["min_child_samples"]} == \
        payload["selected_structural_params"]
    # recomputing the argmin from the saved CV table gives the same candidate
    recomputed, _ = E.select_candidate(cv, cfg)
    assert recomputed["candidate_id"] == payload["selected_candidate_id"]


def test_no_frozen_setting_changed_after_tuning(cfg, official):
    """config_hash excludes lightgbm.selected, so any other post-freeze edit breaks this."""
    payload, _ = official
    assert E.config_hash(cfg) == payload["config_hash"]
    assert payload["dataset_sha256"] == cfg["data"]["sha256"]
