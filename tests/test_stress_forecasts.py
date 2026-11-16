import copy
import shutil

import numpy as np
import pandas as pd
import pytest

from src import data_audit as da
from src import intervals as iv
from src import models as M
from src import stress_forecasts as F


CORE = ["Delhi", "Lucknow", "Chennai", "Hyderabad", "Bengaluru"]


TRAIN = ("2015-01-01", "2018-12-31")


@pytest.fixture(scope="module")
def runs(cfg, dirs, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stress")
    point_dirs = {k: tmp / "point" / k for k in ("models", "predictions", "tables")}
    point_dirs["tables"].mkdir(parents=True)
    shutil.copy(dirs["tables"] / F.USABLE_SAMPLES, point_dirs["tables"])
    point = F.run_point(cfg, point_dirs)
    unc_dirs = {"models": dirs["models"], "predictions": tmp / "unc", "tables": tmp / "unc"}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(iv, "fit_lgbm", lambda *a, **k: pytest.fail("a saved stress quantile model was refit"))
        unc = F.run_uncertainty(cfg, unc_dirs)
    return {"point": (point, point_dirs), "unc": (unc, unc_dirs)}


def test_point_rerun_reproduces_models_and_files(runs, dirs):
    _, d = runs["point"]
    for h in (1, 3, 7):
        f = F.POINT_MODEL_FILE.format(h=h)
        assert (d["models"] / f).read_bytes() == (dirs["models"] / f).read_bytes()
    for sub, key in (("predictions", "point_pred"), ("tables", "point_results"), ("tables", "counts")):
        assert (d[sub] / F.FILES[key]).read_bytes() == (dirs[sub] / F.FILES[key]).read_bytes(), key


def test_uncertainty_rerun_reuses_models_and_reproduces_files(runs, dirs):
    _, d = runs["unc"]
    for sub, key in (("predictions", "unc_pred"), ("tables", "unc_results")):
        assert (d[sub] / F.FILES[key]).read_bytes() == (dirs[sub] / F.FILES[key]).read_bytes(), key


def test_frames_ranges_periods_and_counts(runs):
    point = runs["point"][0]
    f = point["frames"]
    assert set(f["train"]["target_date"].dt.year) == {2015, 2016, 2017, 2018}
    assert set(f["calibration"]["target_date"].dt.year) == {2019}
    t = f["test"]
    assert t["target_date"].between("2020-01-01", "2020-07-01").all()
    assert ((t["stress_period"] == "lockdown") == (t["target_date"] >= "2020-03-25")).all()
    assert t.groupby("stress_period", observed=True)["target_date"].nunique().to_dict() == {
        "lockdown": 99, "pre_lockdown": 84}
    assert point["data_use"]["lockdown_label_origin_before_lockdown"] == {1: 5, 3: 15, 7: 35}
    assert [r["n_origin_in_2019"] for r in point["data_use"]["data_use_2019"]] == [5, 15, 35]


@pytest.mark.parametrize("tamper", ["label", "period", "unobserved", "duplicate", "target_date"])
def test_stress_frames_reject_tampered_parquet(cfg, tmp_path, tamper):
    df = pd.read_parquet(da.resolve(cfg, "forecast_dataset"))
    i = df.index[df["city"].isin(CORE) & (df["split_stress"] == "test")][0]
    for c in ("split_stress", "stress_period"):
        df[c] = df[c].astype(object)
    if tamper == "label":
        df.loc[i, "split_stress"] = "train"
    elif tamper == "period":
        df.loc[i, "stress_period"] = "lockdown" if df.at[i, "stress_period"] == "pre_lockdown" else "pre_lockdown"
    elif tamper == "unobserved":
        df.loc[i, "y_target"] = np.nan
    elif tamper == "duplicate":
        df = pd.concat([df, df.loc[[i]]], ignore_index=True)
    else:
        df.loc[[i], "origin_date"] -= pd.Timedelta(days=1)
    df.to_parquet(tmp_path / "d.parquet")
    c = copy.deepcopy(cfg)
    c["paths"]["forecast_dataset"] = str(tmp_path / "d.parquet")
    with pytest.raises(ValueError, match="stress"):
        F.stress_frames(c)


def test_audit_rejects_mismatch_with_usable_samples(cfg, runs, tmp_path):
    saved = pd.read_csv(runs["point"][1]["tables"] / F.USABLE_SAMPLES, keep_default_na=False)
    saved.loc[saved["scheme"] == "stress", "n_samples"] += 1
    saved.to_csv(tmp_path / F.USABLE_SAMPLES, index=False)
    with pytest.raises(AssertionError, match="usable-samples"):
        F.audit(cfg, runs["point"][0]["frames"], tmp_path)


@pytest.mark.parametrize("h", [1, 3, 7])
def test_models_trained_on_2015_2018_only(runs, dirs, h):
    tr = runs["point"][0]["frames"]["train"]
    for name in (F.POINT_MODEL_FILE.format(h=h), f"{F.QUANTILE_STEM}_h{h}_q05.txt"):
        meta = M.load_model(dirs["models"] / name)[1]
        assert meta["train_date_range"] == list(TRAIN) and meta["n_train"] == int((tr["horizon"] == h).sum())
        assert meta["train_cities"] == sorted(CORE) and meta["stress_config_hash"] == da.stress_config_hash(
            da.load_config())


def test_point_results_on_identical_rows(runs):
    r, p = runs["point"][0]["results"], runs["point"][0]["pred"]
    keys = ["period", "scope", "city", "horizon"]
    lg, pe = (r[r["method"] == m].set_index(keys) for m in ("lightgbm", "persistence"))
    assert (lg["n"] == pe["n"]).all() and np.allclose(lg["skill_mae"], 1 - lg["mae"] / pe["mae"], atol=1e-15)
    assert set(r["mask_type"]) == {"common_eval_mask"} and set(r["period"]) == set(F.PERIODS)
    p7 = p[p["horizon"] == 7]
    np.testing.assert_array_equal(p7[p7["method"] == "persistence"]["y_pred"],
                                  p7[p7["method"] == "seasonal_naive"]["y_pred"])


def test_stress_calibration_on_2019_and_ready_rule(runs):
    u = runs["unc"][0]
    assert u["scores"]["target_date"].dt.year.eq(2019).all()
    assert [round(v["q_hat"], 3) for v in u["q_hat"].values()] == [3.076, 6.249, 7.966]
    p = u["pred"]
    assert (p["calibration_ready"] == (p["origin_date"] >= "2019-12-31")).all()
    assert u["checks"]["n_test_rows_not_calibration_ready"] == {1: 0, 3: 10, 7: 30}


def test_aci_streams_continue_across_lockdown(runs):
    p = runs["unc"][0]["pred"]
    t = p[(p["split"] == "test") & p["calibration_ready"]].sort_values("origin_date", kind="stable")
    assert (t.groupby(["city", "horizon"])["aci_alpha"].first() == 0.10).all()
    entering = t[t["stress_period"] == "lockdown"].groupby(["city", "horizon"])["aci_alpha"].first()
    assert len(entering) == 15 and (entering != 0.10).all()


def test_uncertainty_guard(cfg):
    c = copy.deepcopy(cfg)
    c["stress_test"]["horizons"] = [1]
    with pytest.raises(ValueError, match="stress_test.horizons"):
        F.run_uncertainty(c, {"models": None})
