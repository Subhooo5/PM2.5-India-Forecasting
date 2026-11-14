"""Day 2 task 3: shared LightGBM pipeline (fit/predict/evaluate/save/load) and metrics.

Fits here use the frozen selected params with n_estimators reduced to 30 for speed, on
2015-2017 training rows only. No protocol/RQ experiment is run.
(i) return types: covered by the existing tests/test_return_types.py (re-run, not rewritten).
"""
import copy
import json

import lightgbm
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import metrics as MET
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
    core = ds["city"].astype(str).isin(cfg["cities"]["core"])
    return ds.loc[core & (ds["split_main"] == "train")]


@pytest.fixture(scope="module")
def fitted(train_rows, params, cfg):
    return M.fit_lgbm(train_rows, "causal_common", H, params, TRAIN_RANGE, cfg)


# ------------------------------------------------------------ (a) date guards
def test_fit_refuses_rows_outside_train_range(ds, params, cfg):
    core = ds[ds["city"].astype(str).isin(cfg["cities"]["core"])]
    with pytest.raises(LeakageError, match="outside"):
        M.fit_lgbm(core, "causal_common", H, params, TRAIN_RANGE, cfg)       # contains 2018+
    narrow = core[core["split_main"] == "train"]
    with pytest.raises(LeakageError):
        M.fit_lgbm(narrow, "causal_common", H, params, ("2016-01-01", "2017-12-31"), cfg)


def test_tuning_stage_refuses_any_2018_target(ds, params, cfg):
    core = ds[ds["city"].astype(str).isin(cfg["cities"]["core"])
              & (ds["target_date"] <= "2018-03-31")]
    with pytest.raises(LeakageError):
        M.fit_lgbm(core, "causal_common", H, params, ("2015-01-01", "2018-03-31"), cfg, stage="tuning")
    ok = core[core["target_date"] <= "2017-12-31"]
    M.fit_lgbm(ok, "causal_common", H, params, TRAIN_RANGE, cfg, stage="tuning")   # allowed


def test_early_stopping_params_rejected(train_rows, params, cfg):
    with pytest.raises(ValueError, match="early stopping"):
        M.fit_lgbm(train_rows, "causal_common", H, {**params, "early_stopping_round": 10},
                   TRAIN_RANGE, cfg)


# ------------------------------------------------------------ (b) feature consistency
def test_model_features_equal_yaml_feature_set(fitted, cfg):
    model, meta = fitted
    expected = feature_set(cfg, "causal_common")
    assert meta["features"] == expected
    assert model.booster_.feature_name() == expected


def test_predict_raises_on_missing_feature(fitted, train_rows):
    model, _ = fitted
    rows = train_rows[train_rows["horizon"] == H].drop(columns=["pm25_lag7"])
    with pytest.raises(ValueError, match="missing"):
        M.predict_lgbm(model, rows)


def test_feature_frame_validator_rejects_extra_and_reordered(cfg, train_rows):
    feats = feature_set(cfg, "causal_common")
    X = train_rows[feats]
    M.validate_feature_frame(X, feats)
    with pytest.raises(ValueError, match="extra"):
        M.validate_feature_frame(train_rows[feats + ["snaive_ref"]], feats)
    with pytest.raises(ValueError, match="reordered=True"):
        M.validate_feature_frame(train_rows[feats[::-1]], feats)
    with pytest.raises(ValueError, match="missing"):
        M.validate_feature_frame(train_rows[feats[1:]], feats)


def test_predict_raises_if_model_feature_order_tampered(fitted, train_rows):
    model, meta = fitted
    tampered = copy.copy(model)
    setattr(tampered, "pm25_meta_", {**meta, "features": meta["features"][::-1]})
    with pytest.raises(ValueError, match="order"):
        M.predict_lgbm(tampered, train_rows[train_rows["horizon"] == H])


def test_predict_rejects_other_horizon_rows(fitted, train_rows):
    model, _ = fitted
    with pytest.raises(ValueError, match="horizon"):
        M.predict_lgbm(model, train_rows[train_rows["horizon"] == 3])


# ------------------------------------------------------------ (c) alignment
def test_prediction_alignment(fitted, ds, cfg):
    model, _ = fitted
    rows = ds[(ds["horizon"] == H) & ds["city"].astype(str).isin(cfg["cities"]["core"])
              & (ds["split_main"] == "train")]
    shuffled = rows.sample(frac=0.5, random_state=1)                  # non-monotonic index
    pred = M.predict_lgbm(model, shuffled)
    assert type(pred) is pd.DataFrame
    assert pred.index.equals(shuffled.index) and len(pred) == len(shuffled)
    assert list(pred.columns) == ["city", "origin_date", "target_date", "horizon", "y_true", "y_pred"]
    for k in ["city", "origin_date", "target_date", "horizon"]:
        assert (pred[k].astype(str) == shuffled[k].astype(str)).all(), k
    np.testing.assert_array_equal(pred["y_true"].to_numpy(), shuffled["y_target"].to_numpy())
    assert pred["y_pred"].notna().all()
    direct = model.predict(shuffled[getattr(model, "pm25_meta_")["features"]])
    np.testing.assert_allclose(pred["y_pred"].to_numpy(), direct)


# ------------------------------------------------------------ (d) reproducibility
def test_same_data_same_params_identical_predictions(train_rows, params, cfg, fitted):
    again, _ = M.fit_lgbm(train_rows, "causal_common", H, params, TRAIN_RANGE, cfg)
    rows = train_rows[train_rows["horizon"] == H]
    np.testing.assert_array_equal(M.predict_lgbm(fitted[0], rows)["y_pred"].to_numpy(),
                                  M.predict_lgbm(again, rows)["y_pred"].to_numpy())


def test_different_seed_changes_bagged_fit(train_rows, params, cfg, fitted):
    assert params["subsample"] < 1 and params["subsample_freq"] >= 1
    other = {**params, "random_state": 7, "bagging_seed": 7}
    m2, _ = M.fit_lgbm(train_rows, "causal_common", H, other, TRAIN_RANGE, cfg)
    rows = train_rows[train_rows["horizon"] == H]
    a = M.predict_lgbm(fitted[0], rows)["y_pred"].to_numpy()
    b = M.predict_lgbm(m2, rows)["y_pred"].to_numpy()
    assert not np.array_equal(a, b)


def test_pipeline_matches_tuning_fit(ds, cfg):
    """fit_lgbm reproduces the raw LGBMRegressor fit used inside run_tuning (fold 2, h=1)."""
    p = {**M.lightgbm_params(cfg), "n_estimators": 30}
    feats = feature_set(cfg, "causal_common")
    core = ds[ds["city"].astype(str).isin(cfg["cities"]["core"]) & (ds["horizon"] == H)]
    tr = core[core["target_date"].between("2015-01-01", "2016-12-31")]
    va = core[core["target_date"].between("2017-01-01", "2017-12-31") & core["eval_mask"]]
    raw = lightgbm.LGBMRegressor(**p).fit(tr[feats], tr["y_target"]).predict(va[feats])
    m, _ = M.fit_lgbm(tr, "causal_common", H, p, ("2015-01-01", "2016-12-31"), cfg, stage="tuning")
    np.testing.assert_array_equal(M.predict_lgbm(m, va)["y_pred"].to_numpy(), raw)


# ------------------------------------------------------------ (e) NaN features / target
def test_nan_features_accepted_and_predicted(fitted, train_rows, cfg):
    feats = feature_set(cfg, "causal_common")
    rows = train_rows[train_rows["horizon"] == H]
    nan_rows = rows[rows[feats].isna().any(axis=1)]
    assert len(nan_rows) > 0                               # training data really has NaNs
    pred = M.predict_lgbm(fitted[0], nan_rows)
    assert pred["y_pred"].notna().all() and len(pred) == len(nan_rows)


def test_training_target_never_imputed(train_rows, params, cfg):
    rows = train_rows[train_rows["horizon"] == H].copy()
    rows.iloc[5, rows.columns.get_loc("y_target")] = np.nan
    with pytest.raises(ValueError, match="never imputed"):
        M.fit_lgbm(rows, "causal_common", H, params, TRAIN_RANGE, cfg)
    assert np.isnan(rows["y_target"].iloc[5])               # input untouched, not filled


# ------------------------------------------------------------ (f) forbidden-feature guards
@pytest.mark.parametrize("inject", ["AQI", "PM10", "CO", "AQI_Bucket"])
def test_forbidden_columns_rejected_in_causal_sets(cfg, train_rows, params, inject):
    bad = copy.deepcopy(cfg)
    bad["feature_sets"]["causal_common"] = bad["feature_sets"]["causal_common"] + [inject]
    with pytest.raises(LeakageError):
        M.check_feature_set(bad, "causal_common")
    with pytest.raises(LeakageError):
        M.fit_lgbm(train_rows, "causal_common", H, params, TRAIN_RANGE, bad)


def test_city_id_only_in_city_aware_sets(cfg):
    bad = copy.deepcopy(cfg)
    bad["feature_sets"]["causal_common"] = bad["feature_sets"]["causal_common"] + ["city"]
    with pytest.raises(LeakageError, match="city"):
        M.check_feature_set(bad, "causal_common")
    assert "city" in M.check_feature_set(cfg, "causal_city_aware")
    bad2 = copy.deepcopy(cfg)
    bad2["feature_sets"]["same_day_estimation"] = bad2["feature_sets"]["same_day_estimation"] + ["city"]
    with pytest.raises(LeakageError):
        M.check_feature_set(bad2, "same_day_estimation")


def test_same_day_set_allows_copollutants_but_not_history(cfg):
    feats = M.check_feature_set(cfg, "same_day_estimation")
    assert "AQI" in feats and "AQI_Bucket" not in feats
    bad = copy.deepcopy(cfg)
    bad["feature_sets"]["same_day_estimation"] = bad["feature_sets"]["same_day_estimation"] + ["pm25_lag1"]
    with pytest.raises(LeakageError, match="history"):
        M.check_feature_set(bad, "same_day_estimation")


def test_city_aware_fit_encodes_unseen_cities(ds, params, cfg):
    core = ds[ds["city"].astype(str).isin(cfg["cities"]["core"]) & (ds["split_main"] == "train")]
    model, meta = M.fit_lgbm(core, "causal_city_aware", H, params, TRAIN_RANGE, cfg)
    assert meta["train_cities"] == sorted(cfg["cities"]["core"])
    kolkata = ds[(ds["city"] == "Kolkata") & (ds["horizon"] == H)].head(20)
    pred = M.predict_lgbm(model, kolkata)                    # unseen city -> NaN code, no error
    assert pred["y_pred"].notna().all()


# ------------------------------------------------------------ (g) metadata + save/load
REQUIRED_META = {"feature_set", "features", "horizon", "params", "train_date_range", "n_train",
                 "seed", "dataset_sha256", "lightgbm_version", "created_utc"}


def test_metadata_keys_and_roundtrip(fitted, train_rows, cfg, tmp_path):
    model, meta = fitted
    assert REQUIRED_META <= set(meta)
    assert meta["dataset_sha256"] == cfg["data"]["sha256"]
    assert meta["horizon"] == H and meta["n_train"] == int((train_rows["horizon"] == H).sum())
    path = M.save_model(model, meta, tmp_path / "c_h1.txt")
    side = json.loads((tmp_path / "c_h1.meta.json").read_text())
    assert side == json.loads(json.dumps(meta, default=str))
    booster, meta2 = M.load_model(path)
    assert meta2 == side
    rows = train_rows[train_rows["horizon"] == H]
    np.testing.assert_allclose(M.predict_lgbm(booster, rows)["y_pred"].to_numpy(),
                               M.predict_lgbm(model, rows)["y_pred"].to_numpy(), rtol=0, atol=1e-9)


def test_evaluate_predictions_uses_mask(fitted, ds, cfg):
    model, _ = fitted
    rows = ds[(ds["horizon"] == H) & ds["city"].astype(str).isin(cfg["cities"]["core"])
              & (ds["split_main"] == "train")]
    pred = M.predict_lgbm(model, rows)
    out = M.evaluate_predictions(pred, rows["eval_mask"])
    assert type(out) is dict and set(out) == {"mae", "rmse", "r2", "n"}
    assert out["n"] == int(rows["eval_mask"].sum())
    with pytest.raises(ValueError):
        M.evaluate_predictions(pred, rows["eval_mask"].reset_index(drop=True))


# ------------------------------------------------------------ (h) metrics
def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.gamma(2, 40, 500), rng.gamma(2, 40, 500)
    assert MET.mae(y, p) == pytest.approx(mean_absolute_error(y, p), rel=1e-12)
    assert MET.rmse(y, p) == pytest.approx(np.sqrt(mean_squared_error(y, p)), rel=1e-12)
    assert MET.r2(y, p) == pytest.approx(r2_score(y, p), rel=1e-12)
    assert MET.point_metrics(y, p)["n"] == 500


def test_metrics_edge_cases():
    empty = MET.point_metrics([], [])
    assert empty["n"] == 0 and all(np.isnan(empty[k]) for k in ("mae", "rmse", "r2"))
    const = MET.point_metrics([5.0, 5.0, 5.0], [4.0, 5.0, 6.0])
    assert np.isnan(const["r2"]) and const["mae"] == pytest.approx(2 / 3) and const["n"] == 3
    with_nan_target = MET.point_metrics([1.0, np.nan, 3.0], [1.0, 99.0, 2.0])
    assert with_nan_target["n"] == 2 and with_nan_target["mae"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        MET.point_metrics([1.0, 2.0], [1.0, np.nan])
    with pytest.raises(ValueError):
        MET.point_metrics([1.0, 2.0], [1.0])
