import json

import numpy as np
import pandas as pd
import pytest

from src import leakage_protocols as P
from src.experiments import POINT_METHODS, assert_frames_identical, assert_no_calibration
from src.models import load_model
from src.splits import LeakageError


@pytest.fixture(scope="module")
def saved(cfg, dirs):
    return P.load_saved(cfg, dirs)


@pytest.mark.parametrize("h, n_train", [(1, 4973), (3, 4935), (7, 4889)])
def test_point_models_trained_on_2015_2017_causal_features(dirs, cfg, h, n_train):
    _, meta = load_model(dirs["models"] / P.POINT_MODEL_FILE.format(h=h))
    assert (meta["n_train"], meta["train_date_range"]) == (n_train, ["2015-01-01", "2017-12-31"])
    assert meta["feature_set"] == "causal_common" and meta["train_cities"] == sorted(cfg["cities"]["core"])


def test_protocol_c_predictions(saved, frames):
    C, test = saved["C"], frames[1]
    assert len(C) == 3 * len(test) and C["target_date"].dt.year.eq(2019).all()
    for h in (1, 3, 7):
        g = C[C["horizon"] == h]
        lg, pe, sn = (g[g["method"] == m].reset_index(drop=True) for m in POINT_METHODS)
        th = test[test["horizon"] == h].reset_index(drop=True)
        np.testing.assert_array_equal(pe["y_pred"], th["pm25_origin"])
        np.testing.assert_array_equal(sn["y_pred"], th["snaive_ref"])
        assert (lg["eval_mask"] == pe["eval_mask"]).all() and len(lg) == 1825
    h7 = C[C["horizon"] == 7]
    np.testing.assert_array_equal(h7.loc[h7["method"] == "persistence", "y_pred"],
                                  h7.loc[h7["method"] == "seasonal_naive", "y_pred"])


def test_protocol_c_and_b_are_reproducible(cfg, dirs, tmp_dirs, saved):
    c = P.run_protocol_c(cfg, tmp_dirs)
    cols = ["city", "origin_date", "target_date", "horizon", "method", "y_true", "y_pred", "eval_mask"]
    assert_frames_identical(c["pred"].assign(method=c["pred"]["method"].astype(str)), saved["C"],
                            ["horizon", "method", "city", "origin_date"], cols, "Protocol C rerun")
    for h in (1, 3, 7):
        f = P.POINT_MODEL_FILE.format(h=h)
        assert (tmp_dirs["models"] / f).read_bytes() == (dirs["models"] / f).read_bytes()
    b = P.run_protocol_b(cfg, tmp_dirs)
    assert_frames_identical(b["pred"], saved["B"], ["city", "target_date"], list(saved["B"].columns), "B rerun")
    assert b["meta"]["features"] == cfg["feature_sets"]["same_day_estimation"] and b["meta"]["horizon"] is None
    assert len(b["train"]) == 5014 and len(b["test"]) == 1825
    assert b["train"]["target_date"].dt.year.isin([2015, 2016, 2017]).all()


def test_same_day_frame_rejects_duplicates(cfg, monkeypatch, same_day):
    dup = pd.concat([same_day, same_day.head(1)])
    monkeypatch.setattr(P, "read_without_calibration", lambda path, c: dup)
    with pytest.raises(AssertionError):
        P.same_day_frame(cfg)


@pytest.fixture(scope="module")
def pool(cfg):
    sd = P.same_day_frame(cfg)
    return sd.loc[sd["split_B"].isin(["train", "test"])]


def test_protocol_a_split_properties(cfg, pool, same_day):
    train, test = P.protocol_a_split(pool, cfg, 42)
    keys = lambda d: set(zip(d["city"].astype(str), d["target_date"]))
    assert keys(train).isdisjoint(keys(test)) and keys(train) | keys(test) == keys(pool)
    assert 2018 not in set(pool["target_date"].dt.year) and (len(train), len(test)) == (5470, 1369)
    share = test.groupby("city", observed=True).size() / pool.groupby("city", observed=True).size()
    assert share.between(0.2001, 0.2003).all()
    stored = same_day.set_index(["city", "target_date"])["split_A"].astype(str)
    assert (stored.loc[list(keys(test))] == "test").all()
    other, _ = P.protocol_a_split(pool, cfg, 43)
    assert keys(other) != keys(train)


def test_protocol_a_outputs(saved, dirs):
    s = saved["A_seeds"].astype({"seed": str})
    assert list(s.columns) == ["seed", "subset", "n_train", "n_test", "mae", "rmse", "r2", "n"]
    per = s[s["seed"].isin([str(x) for x in P.SEEDS])]
    assert len(per) == 10 and set(per["subset"]) == {"all_test", "test_2019_only"}
    mean = s[(s["seed"] == "mean") & (s["subset"] == "all_test")]["mae"].item()
    assert mean == pytest.approx(per[per["subset"] == "all_test"]["mae"].mean(), abs=1e-12)
    comp = json.loads((dirs["predictions"] / "leakage_protocol_a.meta.json").read_text())["split_composition"]
    assert comp["A"]["test_rows_per_year"] == {"2015": 317, "2016": 334, "2017": 354, "2019": 364}
    assert comp["B"]["n_test"] == 1825 and saved["A"]["task"].eq("same_day_estimation").all()


def test_protocol_a_regeneration_and_tamper_detection(cfg, saved):
    preds = P.regenerate_protocol_a(cfg, saved)
    assert set(preds) == set(P.SEEDS)
    tampered = {**saved, "A": saved["A"].assign(y_pred=saved["A"]["y_pred"] + 1e-9)}
    with pytest.raises(AssertionError, match="y_pred"):
        P.regenerate_protocol_a(cfg, tampered)


def test_no_calibration_rows_anywhere(saved):
    for k in ("A", "B", "C"):
        assert not saved[k]["target_date"].dt.year.eq(2018).any()
    with pytest.raises(LeakageError):
        assert_no_calibration(pd.Series(pd.to_datetime(["2018-01-01"])), {"splits": {"main": {
            "train": ["2015-01-01", "2017-12-31"], "calibration": ["2018-01-01", "2018-12-31"],
            "test": ["2019-01-01", "2019-12-31"]}}}, "x")
