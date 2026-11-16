import copy
import json

import lightgbm
import numpy as np
import pandas as pd
import pytest

from src import models as M
from src.features import feature_set
from src.splits import LeakageError

TRAIN_RANGE = ("2015-01-01", "2017-12-31")
H = 1


@pytest.fixture(scope="module")
def params(cfg):
    return {**M.lightgbm_params(cfg), "n_estimators": 30}


@pytest.fixture(scope="module")
def train_rows(ds, cfg):
    return ds.loc[ds["city"].astype(str).isin(cfg["cities"]["core"]) & (ds["split_main"] == "train")]


@pytest.fixture(scope="module")
def fitted(train_rows, params, cfg):
    return M.fit_lgbm(train_rows, "causal_common", H, params, TRAIN_RANGE, cfg)


def test_params_are_the_frozen_spec(cfg):
    base = M.base_params(cfg)
    assert isinstance(base, dict) and isinstance(M.lightgbm_params(cfg), dict)
    assert {k: base[k] for k in ("objective", "n_estimators", "learning_rate", "subsample", "subsample_freq",
                                 "colsample_bytree", "reg_lambda", "bagging_seed", "random_state", "n_jobs",
                                 "deterministic", "force_col_wise")} == {
        "objective": "regression", "n_estimators": 500, "learning_rate": 0.05, "subsample": 0.8,
        "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0, "bagging_seed": 42, "random_state": 42,
        "n_jobs": 1, "deterministic": True, "force_col_wise": True}
    assert M.lightgbm_params(cfg) == {**base, "num_leaves": 15, "min_child_samples": 50}
    assert [c["candidate_id"] for c in M.candidate_params(cfg)] == ["c1", "c2", "c3", "c4", "c5", "c6"]
    q = M.quantile_params(cfg, 0.05)
    assert {k: v for k, v in q.items() if k != "alpha"} == {**M.lightgbm_params(cfg), "objective": "quantile"}
    assert (q["objective"], q["alpha"]) == ("quantile", 0.05)
    with pytest.raises(ValueError):
        M.quantile_params(cfg, 0.5)


@pytest.mark.parametrize("mutate", [
    lambda c: c["lightgbm"]["base"].update(random_state=7),
    lambda c: c["lightgbm"]["base"].update(bagging_seed=7),
    lambda c: c["lightgbm"].update(early_stopping=True),
    lambda c: c["lightgbm"]["grid"]["candidates"].reverse(),
    lambda c: c["lightgbm"].update(selected=None),
])
def test_param_guards(cfg, mutate):
    c = copy.deepcopy(cfg)
    mutate(c)
    with pytest.raises(ValueError):
        M.candidate_params(c)
        M.lightgbm_params(c)


@pytest.mark.parametrize("name", ["causal_common", "causal_city_aware", "same_day_estimation"])
@pytest.mark.parametrize("column", ["y_target", "y_true", "PM2.5"])
def test_target_columns_rejected_in_every_set(cfg, name, column):
    c = copy.deepcopy(cfg)
    c["feature_sets"][name] = feature_set(cfg, name) + [column]
    with pytest.raises(LeakageError):
        M.check_feature_set(c, name)


@pytest.mark.parametrize("column", list(M.ALWAYS_FORBIDDEN_IN_CAUSAL))
@pytest.mark.parametrize("weaken", [False, True])
def test_forbidden_columns_rejected_in_causal_sets(cfg, column, weaken):
    c = copy.deepcopy(cfg)
    c["feature_sets"]["causal_common"] = feature_set(cfg, "causal_common") + [column]
    if weaken:
        c["features"]["forbidden"] = []
    with pytest.raises(LeakageError):
        M.check_feature_set(c, "causal_common")


def test_feature_set_guards(cfg):
    assert set(cfg["features"]["forbidden"]) == set(M.ALWAYS_FORBIDDEN_IN_CAUSAL)
    assert "city" in M.check_feature_set(cfg, "causal_city_aware")
    assert set(M.SAME_DAY_COPOLLUTANTS) <= set(M.check_feature_set(cfg, "same_day_estimation"))
    for name, extra in (("causal_common", "city"), ("same_day_estimation", "pm25_lag1"),
                        ("same_day_estimation", "AQI_Bucket")):
        c = copy.deepcopy(cfg)
        c["feature_sets"][name] = feature_set(cfg, name) + [extra]
        with pytest.raises(LeakageError):
            M.check_feature_set(c, name)


def test_fit_guards(ds, train_rows, params, cfg):
    core = ds[ds["city"].astype(str).isin(cfg["cities"]["core"])]
    with pytest.raises(LeakageError, match="outside"):
        M.fit_lgbm(core, "causal_common", H, params, TRAIN_RANGE, cfg)
    with pytest.raises(ValueError, match="early stopping"):
        M.fit_lgbm(train_rows, "causal_common", H, {**params, "early_stopping_round": 5}, TRAIN_RANGE, cfg)
    with pytest.raises(ValueError, match="horizon"):
        M.fit_lgbm(train_rows, "causal_common", None, params, TRAIN_RANGE, cfg)
    with pytest.raises(ValueError, match="observed target"):
        M.fit_lgbm(train_rows.assign(y_target=np.nan), "causal_common", H, params, TRAIN_RANGE, cfg)


def test_fit_metadata_and_roundtrip(fitted, train_rows, cfg, tmp_path):
    model, meta = fitted
    assert model.booster_.feature_name() == meta["features"] == feature_set(cfg, "causal_common")
    assert meta["n_train"] == int((train_rows["horizon"] == H).sum()) and meta["horizon"] == H
    assert meta["train_cities"] == sorted(cfg["cities"]["core"]) and "config_hash" in meta
    path = M.save_model(model, meta, tmp_path / "m.txt")
    booster, loaded = M.load_model(path)
    assert loaded == json.loads(json.dumps(meta))
    rows = train_rows[train_rows["horizon"] == H]
    np.testing.assert_array_equal(M.predict_lgbm(booster, rows)["y_pred"], M.predict_lgbm(model, rows)["y_pred"])


def test_predict_guards_and_alignment(fitted, train_rows):
    model, meta = fitted
    rows = train_rows[train_rows["horizon"] == H]
    pred = M.predict_lgbm(model, rows.sample(frac=1, random_state=0))
    assert pred.index.equals(rows.sample(frac=1, random_state=0).index)
    assert (pred["y_true"] == rows.loc[pred.index, "y_target"]).all()
    with pytest.raises(ValueError, match="missing feature"):
        M.predict_lgbm(model, rows.drop(columns=["pm25_lag1"]))
    with pytest.raises(ValueError, match="another horizon"):
        M.predict_lgbm(model, train_rows[train_rows["horizon"] == 3])
    tampered = lightgbm.Booster(model_str=model.booster_.model_to_string())
    setattr(tampered, "pm25_meta_", {**meta, "features": list(reversed(meta["features"]))})
    with pytest.raises(ValueError, match="order"):
        M.predict_lgbm(tampered, rows)


def test_fit_is_deterministic_and_bagging_is_active(train_rows, params, cfg):
    rows = train_rows[train_rows["horizon"] == H]
    a = M.predict_lgbm(M.fit_lgbm(train_rows, "causal_common", H, params, TRAIN_RANGE, cfg)[0], rows)["y_pred"]
    b = M.predict_lgbm(M.fit_lgbm(train_rows, "causal_common", H, params, TRAIN_RANGE, cfg)[0], rows)["y_pred"]
    other = {**params, "bagging_seed": 7, "random_state": 7}
    c = M.predict_lgbm(M.fit_lgbm(train_rows, "causal_common", H, other, TRAIN_RANGE, cfg)[0], rows)["y_pred"]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_nan_features_are_kept(fitted, train_rows):
    rows = train_rows[(train_rows["horizon"] == H) & train_rows["pm25_lag7"].isna()]
    assert len(rows) > 0 and np.isfinite(M.predict_lgbm(fitted[0], rows)["y_pred"]).all()


def test_city_aware_fit_encodes_unseen_city_as_missing(ds, params, cfg):
    tr = ds[ds["city"].isin(["Delhi", "Chennai"]) & (ds["split_main"] == "train")]
    model, meta = M.fit_lgbm(tr, "causal_city_aware", H, params, TRAIN_RANGE, cfg)
    X = M.feature_matrix(ds[ds["city"] == "Mumbai"].head(5), meta["features"], meta["train_cities"])
    assert X["city"].isna().all() and list(X["city"].cat.categories) == ["Chennai", "Delhi"]
    assert isinstance(M.predict_lgbm(model, ds[(ds["city"] == "Mumbai") & (ds["horizon"] == H)]), pd.DataFrame)
