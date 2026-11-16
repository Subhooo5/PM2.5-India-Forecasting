import copy
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src import data_audit as da
from src import intervals as iv
from src import uncertainty as U
from src.conformal import conformal_quantile


CORE = ["Delhi", "Lucknow", "Chennai", "Hyderabad", "Bengaluru"]


ACI_COLS = ["aci_alpha", "aci_lo", "aci_hi", "aci_miss", "aci_update_origin", "aci_alpha_after"]


META_KEYS = ["uncertainty_config_hash", "config_hash", "dataset_sha256", "kaggle_version", "params", "feature_set",
             "features", "horizon", "train_date_range", "n_train", "train_cities", "held_out_city"]


@pytest.fixture(scope="module")
def sha(cfg):
    return da.sha256_file(da.resolve(cfg, "raw_city_day"))


def test_q_hat_is_the_order_statistic_of_2018_scores(unc_run, cfg):
    res = unc_run[0]
    assert list(res["q_hat"]) == [1, 3, 7]
    assert [round(v["q_hat"], 3) for v in res["q_hat"].values()] == [2.456, 4.668, 5.199]
    for h, v in res["q_hat"].items():
        s = res["scores"][res["scores"]["horizon"] == h]
        assert s["target_date"].dt.year.eq(2018).all() and len(s) == v["n_cal"]
        assert v["q_hat"] == conformal_quantile(s["score"], cfg["uncertainty"]["alpha"])


def test_q_hat_ignores_2019_labels(unc_run, cfg):
    p = unc_run[0]["pred"]
    shuffled = p.assign(y_true=p["y_true"].where(p["split"] != "test", p["y_true"] * 3 + 100))
    assert iv.q_hat_by_horizon(iv.calibration_scores(shuffled, cfg, CORE), cfg) == unc_run[0]["q_hat"]


def test_calibration_ready_rule(unc_run):
    p = unc_run[0]["pred"]
    assert (p["calibration_ready"] == (p["origin_date"] >= "2018-12-31")).all()
    assert unc_run[0]["checks"]["n_test_rows_not_calibration_ready"] == {1: 0, 3: 10, 7: 30}
    early = (p["split"] == "test") & ~p["calibration_ready"]
    assert early.sum() == 40 and p.loc[early, ACI_COLS].isna().all().all()
    assert p.loc[p["split"] == "calibration", ACI_COLS].isna().all().all()


def test_aci_streams_start_at_first_ready_forecast_and_mature_after_target(unc_run):
    p = unc_run[0]["pred"]
    t = p[(p["split"] == "test") & p["calibration_ready"]].sort_values("origin_date", kind="stable")
    first = t.groupby(["city", "horizon"]).head(1)
    assert (first["aci_alpha"] == 0.10).all() and (first["origin_date"] == pd.Timestamp("2018-12-31")).all()
    upd = t.dropna(subset=["aci_update_origin"])
    assert (upd["aci_update_origin"] >= upd["target_date"]).all()


def test_early_rows_never_consumed(unc_run, cfg):
    p, scores = unc_run[0]["pred"], unc_run[0]["scores"]
    early = (p["split"] == "test") & ~p["calibration_ready"]
    p2 = p.assign(y_true=p["y_true"].where(~early, p["y_true"] + 1e3), q05=p["q05"].where(~early, -1e6),
                  q95=p["q95"].where(~early, 1e6))
    a = iv.aci_columns(p2, scores, cfg, 0.02)
    pd.testing.assert_frame_equal(a, p[a.columns], check_exact=True)
    for keys in ([], ["horizon"], ["city", "horizon"]):
        pd.testing.assert_frame_equal(iv.grouped_metrics(p2, iv.METHODS, keys, cfg),
                                      iv.grouped_metrics(p, iv.METHODS, keys, cfg), check_exact=True)


def test_aci_streams_are_independent(unc_run, cfg):
    p, scores = unc_run[0]["pred"], unc_run[0]["scores"]
    other = (p["city"] != "Delhi") & (p["split"] == "test")
    p2 = p.assign(y_true=p["y_true"].where(~other, p["y_true"] + 500))
    a = iv.aci_columns(p2, scores, cfg, 0.02)
    delhi = p["city"] == "Delhi"
    pd.testing.assert_frame_equal(a[delhi], p.loc[delhi, a.columns], check_exact=True)


def _copy_models(dirs, tmp, stem, tamper):
    for f in (dirs["models"] / stem).parent.glob(f"{(dirs['models'] / stem).name}_h*"):
        dst = tmp / f.relative_to(dirs["models"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst)
        if dst.name.endswith(".meta.json"):
            dst.write_text(json.dumps(tamper(json.loads(dst.read_text()))))


@pytest.mark.parametrize("key", META_KEYS)
@pytest.mark.parametrize("loco", [False, True])
def test_tampered_model_metadata_rejected(unc_run, cfg, dirs, sha, tmp_path, key, loco):
    stem = U.LOCO_STEM.format(city="Delhi") if loco else U.MODEL_STEM
    _copy_models(dirs, tmp_path, stem, lambda m: {**m, key: "tampered"})
    train = unc_run[0]["frames"]["train"]
    train = train[train["city"] != "Delhi"] if loco else train
    with pytest.raises(ValueError, match=key):
        iv.quantile_models(cfg, train, tmp_path, stem, sha, **({"held_out_city": "Delhi"} if loco else {}))


@pytest.mark.parametrize("tamper", ["duplicate_key", "wrong_target_date", "unknown_label", "mislabelled",
                                    "missing_city"])
def test_frames_sample_integrity(cfg, tmp_path, tamper):
    df = pd.read_parquet(da.resolve(cfg, "forecast_dataset"))
    i = df.index[df["city"].isin(CORE) & (df["split_main"] == "calibration")][0]
    if tamper == "duplicate_key":
        df = pd.concat([df, df.loc[[i]]], ignore_index=True)
    elif tamper == "wrong_target_date":
        df.loc[[i], "target_date"] += pd.Timedelta(days=1)
    elif tamper == "missing_city":
        df = df[~((df["split_main"] == "train") & (df["city"] == "Delhi"))]
    else:
        df["split_main"] = df["split_main"].astype(object)
        df.loc[i, "split_main"] = "foo" if tamper == "unknown_label" else "test"
    df.to_parquet(tmp_path / "d.parquet")
    c = copy.deepcopy(cfg)
    c["paths"]["forecast_dataset"] = str(tmp_path / "d.parquet")
    with pytest.raises(ValueError, match="forecast sample|missing"):
        iv.frames(c)


def test_calibration_and_q_hat_guards(unc_run, cfg):
    p = unc_run[0]["pred"].copy()
    p.loc[(p["split"] == "calibration") & (p["horizon"] == 3), "eval_mask"] = False
    with pytest.raises(ValueError, match="missing"):
        iv.calibration_scores(p, cfg, CORE)
    with pytest.raises(ValueError, match="too small"):
        iv.q_hat_by_horizon(unc_run[0]["scores"].groupby("horizon").head(5), cfg)


def test_interval_checks_hand_counts():
    p = pd.DataFrame({"city": ["A"] * 5 + ["B"], "horizon": 1, "lo": [5.0, np.nan, 0.0, -2.0, 0.0, 1.0],
                      "hi": [1.0, 3.0, np.inf, 4.0, 4.0, 3.0]})
    a = iv.interval_checks(p, "lo", "hi", ["city", "horizon"])[0]
    assert {k: a[k] for k in ("n", "n_crossed", "n_nan", "n_inf", "n_width_le0", "n_lower_below_zero")} == {
        "n": 5, "n_crossed": 1, "n_nan": 1, "n_inf": 1, "n_width_le0": 1, "n_lower_below_zero": 1}
