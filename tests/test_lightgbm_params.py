"""LightGBM parameter checks (frozen Day 2 config): return types, grid order,
subsampling actually enabled, determinism. No model is trained here."""
import lightgbm as lgb
import pytest

from src import models as M


def _with_selected(cfg, cand):
    return {**cfg, "lightgbm": {**cfg["lightgbm"], "selected": cand}}


def _all_param_sets(cfg):
    return [M.base_params(cfg), *(M.merge_params(cfg, c) for c in M.candidate_params(cfg))]


# ---------------------------------------------------------------- return types
def test_param_functions_return_dicts_not_tuples(cfg):
    c = _with_selected(cfg, M.candidate_params(cfg)[0])
    for out in (M.base_params(cfg), M.lightgbm_params(c), M.quantile_params(c, 0.05),
                M.merge_params(cfg, M.candidate_params(cfg)[0])):
        assert type(out) is dict and not isinstance(out, tuple)
    assert type(M.candidate_params(cfg)) is list
    assert all(type(x) is dict for x in M.candidate_params(cfg))


# ---------------------------------------------------------------- frozen base + grid
def test_base_matches_frozen_spec(cfg):
    assert M.base_params(cfg) == {
        "objective": "regression", "n_estimators": 500, "learning_rate": 0.05,
        "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
        "bagging_seed": 42, "random_state": 42, "n_jobs": 1, "deterministic": True,
        "force_col_wise": True, "verbosity": -1}
    assert cfg["seed"] == 42 and cfg["lightgbm"]["early_stopping"] is False


def test_grid_has_six_candidates_in_fixed_order(cfg):
    assert M.candidate_params(cfg) == [
        {"candidate_id": "c1", "num_leaves": 15, "min_child_samples": 20},
        {"candidate_id": "c2", "num_leaves": 15, "min_child_samples": 50},
        {"candidate_id": "c3", "num_leaves": 31, "min_child_samples": 20},
        {"candidate_id": "c4", "num_leaves": 31, "min_child_samples": 50},
        {"candidate_id": "c5", "num_leaves": 63, "min_child_samples": 20},
        {"candidate_id": "c6", "num_leaves": 63, "min_child_samples": 50}]


def test_grid_order_is_enforced(cfg):
    g = cfg["lightgbm"]["grid"]
    swapped = [g["candidates"][1], g["candidates"][0], *g["candidates"][2:]]
    bad = {**cfg, "lightgbm": {**cfg["lightgbm"], "grid": {**g, "candidates": swapped}}}
    with pytest.raises(ValueError, match="enumerate"):
        M.candidate_params(bad)


# ---------------------------------------------------------------- subsampling / determinism
def test_subsampling_is_actually_enabled(cfg):
    for p in _all_param_sets(cfg):
        assert 0 < p["subsample"] < 1
        assert p["subsample_freq"] >= 1, "subsample has no effect unless subsample_freq > 0"
        assert 0 < p["colsample_bytree"] <= 1


def test_deterministic_settings_present(cfg):
    for p in _all_param_sets(cfg):
        assert p["deterministic"] is True
        assert p["force_col_wise"] is True
        assert p["bagging_seed"] == cfg["seed"] == p["random_state"]
        assert p["n_jobs"] == 1 and p["verbosity"] == -1


@pytest.mark.parametrize("key", ["bagging_seed", "random_state"])
def test_seeds_must_match_top_level_seed(cfg, key):
    lgbm = cfg["lightgbm"]
    bad = {**cfg, "lightgbm": {**lgbm, "base": {**lgbm["base"], key: 7}}}
    with pytest.raises(ValueError, match=key):
        M.base_params(bad)


def test_early_stopping_disabled(cfg):
    with pytest.raises(ValueError):
        M.base_params({**cfg, "lightgbm": {**cfg["lightgbm"], "early_stopping": True}})


def test_lightgbm_accepts_the_parameters(cfg):
    """Constructing (not fitting) the estimator: every key reaches LightGBM unchanged."""
    for p in _all_param_sets(cfg):
        got = lgb.LGBMRegressor(**p).get_params()
        for k, v in p.items():
            assert got[k] == v, k
