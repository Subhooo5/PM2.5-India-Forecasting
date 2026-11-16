import json

import numpy as np
import pandas as pd
import pytest

from src import data_audit as da
from src import experiments as E
from src.splits import LeakageError


@pytest.fixture(scope="module")
def tuning_frame(cfg):
    return E.load_tuning_frame(cfg)


def test_tuning_frame_is_training_only(tuning_frame, cfg):
    assert tuning_frame["target_date"].max() <= pd.Timestamp("2017-12-31")
    assert set(tuning_frame["city"].astype(str)) == set(cfg["cities"]["core"])
    with pytest.raises(LeakageError):
        E.tune_frame(pd.concat([tuning_frame, tuning_frame.tail(1).assign(target_date=pd.Timestamp("2018-01-01"))]),
                     cfg)


def test_tuning_reproduces_saved_cv_and_selection(tuning_frame, cfg, dirs):
    cv = E.tune_frame(tuning_frame, cfg)
    saved = pd.read_csv(dirs["models"] / E.TUNING_CV_FILE)
    assert list(saved.columns) == E.CV_COLUMNS and len(saved) == 36
    pd.testing.assert_frame_equal(cv.reset_index(drop=True), saved, check_dtype=False, rtol=0, atol=1e-9)
    selected, ranking = E.select_candidate(cv, cfg)
    assert selected == {"candidate_id": "c2", "num_leaves": 15, "min_child_samples": 50} == cfg["lightgbm"]["selected"]
    payload = json.loads((dirs["models"] / E.SELECTED_FILE).read_text())
    assert payload["selected_candidate_id"] == "c2" and payload["tuning_frame"]["max_target_date"] <= "2017-12-31"
    assert ranking.loc[ranking["selected"], "rank"].item() == 1


def test_selection_tie_breaks_to_earliest_candidate(cfg):
    cv = pd.DataFrame([{"candidate_id": f"c{i}", "fold": f, "horizon": h, "mae": 1.0}
                       for i in range(1, 7) for f in (1, 2) for h in (1, 3, 7)])
    assert E.select_candidate(cv, cfg)[0]["candidate_id"] == "c1"
    with pytest.raises(ValueError):
        E.select_candidate(cv.iloc[1:], cfg)


def test_forecast_frames_exclude_calibration(frames, cfg):
    train, test = frames
    assert train["target_date"].between("2015-01-01", "2017-12-31").all()
    assert test["target_date"].between("2019-01-01", "2019-12-31").all()
    assert set(train["city"].astype(str)) == set(test["city"].astype(str)) == set(cfg["cities"]["core"])
    with pytest.raises(LeakageError):
        E.assert_no_calibration(pd.Series(pd.to_datetime(["2018-06-01"])), cfg, "x")


def _pred(y, a, b, mask):
    n = len(y)
    keys = dict(city=["X"] * n, horizon=[1] * n, protocol="C", y_true=y, eval_mask=mask, row_id=range(n))
    return pd.concat([pd.DataFrame({**keys, "method": "lightgbm", "y_pred": a}),
                      pd.DataFrame({**keys, "method": "persistence", "y_pred": b})], ignore_index=True)


def test_score_methods_hand_computed_on_identical_rows():
    t = E.score_methods(_pred([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 4.0, 9.0], [2.0, 3.0, 3.0, 4.0],
                              [True, True, True, False]))
    row = t[(t["method"] == "lightgbm") & (t["mask_type"] == "common_eval_mask") & (t["scope"] == "overall")].iloc[0]
    assert (row["n"], row["mae"]) == (3, pytest.approx(1 / 3)) and row["skill_mae"] == pytest.approx(1 - (1 / 3) / (2 / 3))
    own = t[(t["method"] == "lightgbm") & (t["mask_type"] == "own_full") & (t["scope"] == "overall")].iloc[0]
    assert own["n"] == 4 and list(t.columns) == E.RESULT_COLUMNS


def test_saved_baseline_results_use_identical_rows(dirs):
    t = pd.read_csv(dirs["tables"] / "baseline_results.csv")
    common = t[t["mask_type"] == "common_eval_mask"]
    assert (common.groupby(["scope", "city", "horizon"])["n"].nunique() == 1).all()
    assert (common.loc[common["method"] == "persistence", "skill_mae"] == 0).all()
    sn = common[common["method"] == "seasonal_naive"]
    assert sn.groupby(["scope", "city"])["mae"].nunique().eq(1).all()


def test_input_exposure_hand_computed(cfg):
    test = pd.DataFrame({"horizon": [1, 1, 1, 7], "origin_date": pd.to_datetime(
        ["2018-12-31", "2019-01-13", "2019-01-14", "2018-12-25"])})
    out = {r["horizon"]: r for r in E.input_exposure(test, cfg, "main")}
    assert out[1] == {"horizon": 1, "n_test_rows": 3, "n_origin_in_2018": 1, "n_lookback_reaches_2018": 2,
                      "lookback_days": 13}
    assert (out[7]["n_origin_in_2018"], out[7]["n_lookback_reaches_2018"]) == (1, 1)


def test_saved_input_exposure_counts(dirs):
    meta = json.loads((dirs["predictions"] / "leakage_protocol_c.meta.json").read_text())
    per_h = {r["horizon"]: r for r in meta["data_use_2018"]["per_horizon"]}
    assert [(per_h[h]["n_origin_in_2018"], per_h[h]["n_lookback_reaches_2018"]) for h in (1, 3, 7)] == [
        (5, 70), (15, 80), (35, 100)]


@pytest.mark.parametrize("tamper, match", [
    (lambda d: d.iloc[:-1], "length"),
    (lambda d: d.assign(y=d["y"] + 1e-12), "'y'"),
    (lambda d: d.assign(k=d["k"].replace({"b": "z"})), "'k'"),
    (lambda d: d.drop(columns="y"), "missing"),
])
def test_assert_frames_identical(tamper, match):
    a = pd.DataFrame({"k": ["a", "b", "c"], "y": [1.0, np.nan, 3.0]})
    E.assert_frames_identical(a, a.iloc[::-1], ["k"], ["k", "y"], "x")
    with pytest.raises(AssertionError, match=match):
        E.assert_frames_identical(a, tamper(a), ["k"], ["k", "y"], "x")


def test_read_predictions_rejects_calibration_targets(cfg, tmp_path):
    p = tmp_path / "p.csv"
    pd.DataFrame({"target_date": ["2018-03-01"], "y": [1]}).to_csv(p, index=False)
    with pytest.raises(LeakageError):
        E.read_predictions(p, cfg)


def test_sidecars_carry_provenance(dirs, cfg):
    sidecars = sorted(dirs["tables"].glob("*.meta.json")) + sorted(dirs["predictions"].glob("*.meta.json"))
    assert len(sidecars) >= 30
    for f in sidecars:
        m = json.loads(f.read_text())
        assert m["dataset_sha256"] == cfg["data"]["sha256"] and m["kaggle_version"] == 12, f.name
        assert m["config_hash"] == da.config_hash(cfg) and "versions" in m and len(m["notes"].splitlines()) <= 1
