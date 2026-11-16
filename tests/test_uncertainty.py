import copy
import json

import numpy as np
import pandas as pd
import pytest

from src import intervals as iv
from src import uncertainty as U
from src.metrics import interval_metrics


CORE = ["Delhi", "Lucknow", "Chennai", "Hyderabad", "Bengaluru"]


def test_rerun_reuses_models_and_reproduces_every_file(unc_run, dirs):
    _, out = unc_run
    files = [U.PRED_FILE, U.HELD_OUT_PRED_FILE]
    for name in files:
        assert (out["predictions"] / name).read_bytes() == (dirs["predictions"] / name).read_bytes(), name
    for name in U.TABLE_FILES.values():
        assert (out["tables"] / name).read_bytes() == (dirs["tables"] / name).read_bytes(), name


def test_tables_score_identical_ready_rows(unc_run, dirs):
    t = unc_run[0]["tables"]
    assert t["overall"]["n"].tolist() == [5435] * 3 and t["horizon"]["n"].tolist() == [1825, 1815, 1795] * 3
    assert t["raw_and_cqr"].query("scope == 'overall'")["n"].tolist() == [5435, 5435]
    assert t["overall"]["picp"].tolist() == pytest.approx([0.806, 0.878, 0.888], abs=6e-4)
    g = t["gamma"]
    primary = g[g["primary"]].reset_index(drop=True)
    cqr_aci = t["horizon"][t["horizon"]["method"] == "cqr_aci"].reset_index(drop=True)
    np.testing.assert_array_equal(primary["picp"], cqr_aci["picp"])
    assert set(g["gamma"]) == {0.005, 0.01, 0.02, 0.05} and g["label"].eq(U.SENS_LABEL).all()


def test_held_out_tables_recomputed_from_prediction_files(unc_run, dirs):
    t = pd.read_csv(dirs["tables"] / U.TABLE_FILES["held_out"])
    files = {"held_out": iv.load_predictions(dirs["predictions"] / U.HELD_OUT_PRED_FILE),
             "familiar": iv.load_predictions(dirs["predictions"] / U.PRED_FILE)}
    methods = {**iv.METHODS, **U.LOCO_METHODS, **U.LOCAL_METHODS}
    for _, row in t.iterrows():
        f = files["familiar" if row["setting"] == "familiar" else "held_out"]
        g = iv.scored(f[(f["city"] == row["city"]) & (f["horizon"] == row["horizon"])])
        lo, hi = methods[row["method"]]
        m = interval_metrics(g["y_true"], g[lo], g[hi], 0.1)
        assert m["n"] == row["n"] and m["picp"] == pytest.approx(row["picp"], abs=1e-12)


def test_held_out_models_exclude_their_city_and_calibration_is_split(unc_run):
    for city, v in unc_run[0]["held_out"].items():
        assert city not in set(v["train"]["city"].astype(str))
        assert set(v["source"]["city"]) == set(CORE) - {city} and set(v["local"]["city"]) == {city}
    c = unc_run[0]["held_out_checks"]["held_out_intervals_test_rows"]
    assert sorted(c) == ["aci_source", "cqr_local", "cqr_source", "raw_loco"] and all(len(v) == 15 for v in c.values())


@pytest.mark.parametrize("how", ["shuffle", "drop", "y_true"])
def test_misaligned_predictions_fail_before_scores(cfg, dirs, tmp_path, monkeypatch, how):
    tamper = {"shuffle": lambda p: p.sample(frac=1, random_state=0), "drop": lambda p: p.iloc[1:],
              "y_true": lambda p: p.assign(y_true=p["y_true"] + 1)}[how]
    real = iv.predict_lgbm
    monkeypatch.setattr(iv, "predict_lgbm", lambda m, g: tamper(real(m, g)))
    monkeypatch.setattr(U, "calibration_scores", lambda *a, **k: pytest.fail("scores built after a failure"))
    with pytest.raises(AssertionError, match="aligned"):
        U.run(cfg, {"models": dirs["models"], "predictions": tmp_path, "tables": tmp_path})


@pytest.mark.parametrize("path, value", [(("aci", "alpha_init"), 0.2), (("quantiles",), [0.1, 0.9])])
def test_config_consumption_guard(cfg, path, value):
    c = copy.deepcopy(cfg)
    node = c["uncertainty"]
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    with pytest.raises(ValueError, match="alpha_init"):
        U.check_config(c)


def test_strata_thresholds_and_validation(unc_run, cfg, raw):
    th = unc_run[0]["thresholds"]
    train = raw[raw["City"].isin(CORE) & raw["Date"].between("2015-01-01", "2017-12-31")]
    assert th == {c: float(g["PM2.5"].quantile(0.9)) for c, g in train.groupby("City")}
    st = U.with_strata(unc_run[0]["pred"], th, cfg)
    assert set(st["season"]) == {"winter", "pre_monsoon", "monsoon", "post_monsoon"}
    with pytest.raises(ValueError):
        U.with_strata(unc_run[0]["pred"], {k: v for k, v in th.items() if k != "Delhi"}, cfg)


def test_sidecars_and_figures(dirs):
    meta = json.loads((dirs["predictions"] / U.PRED_FILE.replace(".csv", ".meta.json")).read_text())
    assert {"uncertainty_config_hash", "q_hat", "checks", "aci_summary"} <= set(meta)
    assert len(list(dirs["figures"].glob("uncertainty_*.png"))) == 6
