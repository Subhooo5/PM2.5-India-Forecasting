"""LightGBM parameter checks: return types (item 1) and subsampling / determinism (item 5).
No model is trained here; LGBMRegressor is only constructed."""
import lightgbm as lgb
import pytest

from src import models as M


@pytest.fixture
def fixed_cfg(cfg):
    return {**cfg, "tuning": {**cfg["tuning"], "mode": "fixed"}}


# ---------------------------------------------------------------- item 1: return type
def test_lightgbm_params_returns_dict_not_tuple(fixed_cfg, ds, cfg, tmp_path, monkeypatch):
    out = M.lightgbm_params(fixed_cfg)
    assert type(out) is dict and not isinstance(out, tuple)
    monkeypatch.setattr(M.da, "write_table", lambda *a, **k: None)
    best, table = M.select_hyperparameters(
        ds, cfg, fit_predict=lambda p, Xt, yt, Xv: ds.loc[Xv.index, "y_target"].to_numpy())
    path = M.save_selected_params(best, table, cfg, tmp_path / "sel.json")
    tuned = M.lightgbm_params(cfg, path)
    assert type(tuned) is dict and not isinstance(tuned, tuple)
    assert type(M.quantile_params(cfg, 0.05, path)) is dict
    assert type(M.base_params(cfg)) is dict
    assert all(type(c) is dict for c in M.candidate_params(cfg))


# ---------------------------------------------------------------- item 5: subsampling
def _all_param_sets(cfg):
    return [M.base_params(cfg), *M.candidate_params(cfg)]


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
        assert p["n_jobs"] == 1


def test_bagging_seed_must_match_seed(cfg):
    lgbm = cfg["models"]["lightgbm"]
    bad = {**cfg, "models": {"lightgbm": {**lgbm, "params": {**lgbm["params"], "bagging_seed": 7}}}}
    with pytest.raises(ValueError, match="bagging_seed"):
        M.base_params(bad)


def test_early_stopping_disabled(cfg):
    assert cfg["models"]["lightgbm"]["early_stopping"] is False
    lgbm = {**cfg["models"]["lightgbm"], "early_stopping": True}
    with pytest.raises(ValueError):
        M.base_params({**cfg, "models": {"lightgbm": lgbm}})


def test_lightgbm_accepts_the_parameters(cfg):
    """Constructing (not fitting) the estimator: every key reaches LightGBM unchanged."""
    for p in _all_param_sets(cfg):
        got = lgb.LGBMRegressor(**p).get_params()
        for k in ("subsample", "subsample_freq", "bagging_seed", "deterministic",
                  "force_col_wise", "random_state", "num_leaves", "min_child_samples"):
            assert got[k] == p[k], k
