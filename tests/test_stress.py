import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src import data_audit as da
from src import intervals as iv
from src import stress as S
from src import stress_forecasts as F


def test_diagnostics_are_read_only_and_reproduce(cfg, dirs, tmp_path, monkeypatch):
    for module, name in ((F, "fit_lgbm"), (F, "predict_lgbm"), (iv, "load_model"), (iv, "fit_lgbm"), (iv, "predict_lgbm")):
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("diagnostics touched a model"))
    for k in ("predictions", "tables"):
        shutil.copytree(dirs[k], tmp_path / k)
    out = S.run_diagnostics(cfg, {"predictions": tmp_path / "predictions", "tables": tmp_path / "tables"})
    for key in ("distribution", "shift", "alpha"):
        assert (tmp_path / "tables" / F.FILES[key]).read_bytes() == (dirs["tables"] / F.FILES[key]).read_bytes()
    shift = out["shift"]
    share = shift[shift["metric"].isin(["picp", "share_below", "share_above"])].groupby(
        ["scope", "city", "horizon", "method"])["pre_lockdown"].sum()
    assert np.allclose(share, 1.0)


def test_period_shift_check_detects_disagreement(dirs, cfg):
    s = S.load_saved(dirs)
    shift = S.period_shift(s["point"], s["unc"], cfg)
    S.check_against_saved(shift, s["pres"], s["ures"])
    s["pres"].loc[s["pres"].index[s["pres"]["period"] == "pre_lockdown"][0], "mae"] += 1
    with pytest.raises(AssertionError, match="period shift"):
        S.check_against_saved(shift, s["pres"], s["ures"])


def test_previous_year_same_dates_leap_day():
    out = S.previous_year_same_dates(pd.DatetimeIndex(["2020-02-28", "2020-02-29", "2020-03-01"]))
    assert list(out[[0, 2]]) == [pd.Timestamp("2019-02-28"), pd.Timestamp("2019-03-01")] and pd.isna(out[1])


def test_manifest(cfg, tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "outputs" / "a.csv").write_text("x\n1\n")
    (tmp_path / "configs" / "c.yaml").write_text("k: 1\n")
    m = S.write_manifest(cfg, tmp_path)
    assert m["path"].tolist() == ["configs/c.yaml", "outputs/a.csv"]
    assert m.loc[1, "sha256"] == da.sha256_file(tmp_path / "outputs" / "a.csv")


def test_sidecars_state_the_protocol(dirs):
    for key in ("point_pred", "unc_pred"):
        meta = json.loads((dirs["predictions"] / F.FILES[key].replace(".csv", ".meta.json")).read_text())
        assert "not causal" in meta["notes"] and "stress_config_hash" in meta
    assert len(list(dirs["figures"].glob("stress_2020_*.png"))) == 6
