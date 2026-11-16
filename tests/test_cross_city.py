import json

import numpy as np
import pandas as pd
import pytest

from src import cross_city as X
from src.experiments import assert_frames_identical
from src.metrics import relative_change_pct
from src.models import load_model, predict_lgbm

CORE = ["Delhi", "Lucknow", "Chennai", "Hyderabad", "Bengaluru"]


@pytest.fixture(scope="module")
def pred(dirs):
    return pd.read_csv(dirs["predictions"] / X.FILES["predictions"], parse_dates=["origin_date", "target_date"],
                       keep_default_na=False, na_values=[""])


@pytest.fixture(scope="module")
def tables(dirs):
    return {k: pd.read_csv(dirs["tables"] / X.FILES[k]) for k in ("pooled", "loco", "transfer")}


def _meta(dirs, name):
    return load_model(dirs["models"] / X.MODELS_SUBDIR / name)[1]


@pytest.mark.parametrize("city", CORE)
@pytest.mark.parametrize("h", [1, 3, 7])
def test_model_training_cities(dirs, city, h):
    per_city, loco = _meta(dirs, f"per_city_{city}_h{h}.txt"), _meta(dirs, f"loco_{city}_h{h}.txt")
    assert per_city["train_cities"] == [city] and per_city["feature_set"] == "causal_common"
    assert loco["train_cities"] == sorted(set(CORE) - {city}) and loco["held_out_city"] == city
    assert loco["train_date_range"] == per_city["train_date_range"] == ["2015-01-01", "2017-12-31"]
    assert loco["params"] == per_city["params"]


@pytest.mark.parametrize("city", CORE)
def test_loco_models_reproduce_saved_predictions(dirs, frames, pred, city):
    test = frames[1]
    for h in (1, 3, 7):
        booster, _ = load_model(dirs["models"] / X.MODELS_SUBDIR / f"loco_{city}_h{h}.txt")
        rows = test[(test["horizon"] == h) & (test["city"] == city)]
        saved = pred[(pred["experiment"] == "loco") & (pred["horizon"] == h) & (pred["city"] == city)]
        assert_frames_identical(predict_lgbm(booster, rows), saved, ["origin_date"], ["origin_date", "y_true", "y_pred"],
                                f"LOCO {city} h={h}")


def test_loco_run_excludes_held_out_city_and_reproduces(cfg, dirs, tmp_dirs, tables, pred, monkeypatch):
    for h in (1, 3, 7):
        for f in (f"point_h{h}.txt", f"point_h{h}.meta.json"):
            (tmp_dirs["models"] / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp_dirs["models"] / f).write_bytes((dirs["models"] / f).read_bytes())
    tmp_dirs["tables"].mkdir(parents=True)
    (tmp_dirs["tables"] / X.FILES["pooled"]).write_bytes((dirs["tables"] / X.FILES["pooled"]).read_bytes())
    seen, real_fit = [], X.fit_lgbm
    monkeypatch.setattr(X, "fit_lgbm", lambda df, *a: seen.append(df) or real_fit(df, *a))
    out = X.run_loco_transfer(cfg, tmp_dirs)
    assert len(seen) == 15
    for i, held in enumerate(CORE):
        for df in seen[3 * i: 3 * i + 3]:
            assert set(df["city"].astype(str)) == set(CORE) - {held}
            assert df["target_date"].max() <= pd.Timestamp("2017-12-31")
    pd.testing.assert_frame_equal(out["loco"], tables["loco"], check_dtype=False, rtol=0, atol=1e-12)
    pd.testing.assert_frame_equal(out["transfer"].reset_index(drop=True), tables["transfer"], check_dtype=False,
                                  rtol=0, atol=1e-12)


def test_pooled_rows_equal_protocol_c(cfg, frames, dirs):
    out = X.pooled_predictions(cfg, frames[1], dirs)
    assert set(out) == {1, 3, 7} and all(len(v[2]) == 1825 for v in out.values())


def test_prediction_file(pred, cfg):
    assert list(pred.columns) == X.PRED_COLUMNS
    assert list(dict.fromkeys(pred["experiment"])) == ["pooled", "per_city", "loco", "transfer"]
    assert pred["target_date"].dt.year.eq(2019).all()
    core = pred[pred["experiment"] != "transfer"]
    assert set(core["city"]) == set(CORE)
    t = pred[pred["experiment"] == "transfer"]
    assert set(t["city"]) == {"Kolkata", "Mumbai", "Visakhapatnam"} and t["held_out_city"].isna().all()
    assert (t["train_cities"] == "|".join(sorted(CORE))).all()


def test_pooled_vs_per_city_table(tables):
    t = tables["pooled"]
    assert list(t.columns) == X.METRIC_COLUMNS
    p, c, d = (t[t["experiment"] == e].reset_index(drop=True) for e in ("pooled", "per_city", "pooled_minus_per_city"))
    assert (p["n"] == c["n"]).all() and len(d) == len(p) == 18
    np.testing.assert_allclose(d["mae"], p["mae"] - c["mae"], atol=1e-12)
    assert p[p["scope"] == "overall"]["mae"].round(3).tolist() == [15.308, 22.762, 24.611]


def test_loco_gap_definition(tables):
    t = tables["loco"]
    assert list(t.columns) == X.LOCO_COLUMNS
    np.testing.assert_allclose(t["gap_vs_pooled_pct"], relative_change_pct(t["mae"], t["mae_pooled"]), atol=1e-12)
    np.testing.assert_allclose(t["gap_vs_per_city_pct"], relative_change_pct(t["mae"], t["mae_per_city"]), atol=1e-12)


def test_transfer_table_and_wording(tables, dirs):
    t = tables["transfer"]
    assert list(t.columns) == X.TRANSFER_COLUMNS and set(t["city"]) == {"Kolkata", "Mumbai", "Visakhapatnam"}
    assert t.loc[t["city"] == "Visakhapatnam", "label"].str.startswith("OPTIONAL").all()
    assert not t["optional"][t["city"] != "Visakhapatnam"].any()
    for f in dirs["tables"].glob("cross_city_*"):
        assert "zero-shot" not in f.read_text().lower()
    assert all("zero-shot" not in json.dumps(_meta(dirs, f"loco_{c}_h1.txt")).lower() for c in CORE)
