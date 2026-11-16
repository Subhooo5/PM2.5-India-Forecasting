import copy
import io
import json

import pandas as pd
import pytest

from src import data_audit as da


def test_load_does_not_modify_raw(cfg):
    path = da.resolve(cfg, "raw_city_day")
    before = da.sha256_file(path)
    raw = da.load_city_day(cfg)
    assert da.sha256_file(path) == before == cfg["data"]["sha256"]
    assert pd.api.types.is_datetime64_any_dtype(raw["Date"]) and pd.api.types.is_float_dtype(raw["PM2.5"])


def test_verify_raw_passes(raw, cfg):
    s = da.verify_raw(raw, cfg)
    assert (s["n_rows"], s["n_cities"], s["date_min"], s["date_max"]) == (29531, 26, "2015-01-01", "2020-07-01")


@pytest.mark.parametrize("tamper", [
    lambda d: d.iloc[1:],
    lambda d: d.drop(columns=["AQI_Bucket"]),
    lambda d: d[d["City"] != "Delhi"],
    lambda d: pd.concat([d, d.iloc[[0]]]),
    lambda d: d[d["Date"] < "2020-07-01"],
])
def test_verify_raw_rejects_tampered(raw, cfg, tamper):
    with pytest.raises(da.RawDataError):
        da.verify_raw(tamper(raw.copy()), cfg)


def test_verify_raw_rejects_wrong_hash(raw, cfg, tmp_path):
    bad = tmp_path / "city_day.csv"
    bad.write_bytes(da.resolve(cfg, "raw_city_day").read_bytes() + b"\n")
    with pytest.raises(da.RawDataError, match="sha256"):
        da.verify_raw(raw, cfg, path=bad)


@pytest.fixture(scope="module")
def cal(raw, cfg):
    return da.pm25_calendar(raw, cfg)


def test_calendar_is_full_daily_grid_and_keeps_observations(cal, raw):
    assert len(cal) == 26 * ((raw["Date"].max() - raw["Date"].min()).days + 1)
    assert not cal.duplicated(["City", "Date"]).any()
    merged = raw[["City", "Date", "PM2.5"]].merge(cal, on=["City", "Date"])
    pd.testing.assert_series_equal(merged["PM2.5"], merged["pm25"], check_names=False)


def test_completeness_invariants(cal, cfg):
    t = da.completeness_by_year(cal)
    assert (t["pm25_observed"] <= t["rows_present"]).all() and (t["rows_present"] <= t["expected_days"]).all()
    days = t.groupby("year")["expected_days"].first()
    assert (days[2016], days[2019], days[2020]) == (366, 365, 183)
    assert t["pm25_observed"].sum() == cal["pm25"].count()
    p = da.completeness_by_period(cal, cfg)
    assert len(p) == 26 and all(p[c].between(0, 100).all() for c in p.columns if c.endswith("_pct"))


def test_gap_runs_on_known_series():
    s = [False, False, True, False, True, False, False, True, False, False, False, False, True, False]
    assert da.gap_runs(pd.Series(s)) == (2, 1, [1, 2, 4])


def test_gap_stats_accounting(cal, cfg):
    g = da.gap_stats(cal, cfg).set_index("City")
    total = g[["leading_missing_days", "trailing_missing_days", "internal_missing_days"]].sum(axis=1)
    assert (total + cal.groupby("City")["pm25"].count() == cal.groupby("City").size()).all()
    bins = g[["gaps_len_1", "gaps_len_2", "gaps_len_3_7", "gaps_len_gt7"]].sum(axis=1)
    assert (bins == g["n_internal_gaps"]).all()


@pytest.mark.parametrize("section, key, changes_hash", [
    ("lightgbm", "selected", False), ("uncertainty", "alpha", False), ("stress_test", "cities", False),
    ("lightgbm", "base", True), ("splits", "main", True),
])
def test_config_hash_scope(cfg, section, key, changes_hash):
    c = copy.deepcopy(cfg)
    c[section][key] = "changed"
    assert (da.config_hash(c) != da.config_hash(cfg)) == changes_hash


@pytest.mark.parametrize("fn, section, key", [
    (da.uncertainty_config_hash, "uncertainty", "alpha"),
    (da.stress_config_hash, "stress_test", "horizons"),
    (da.stress_config_hash, "lightgbm", "selected"),
])
def test_stage_hashes_are_sensitive(cfg, fn, section, key):
    c = copy.deepcopy(cfg)
    c[section][key] = "changed"
    assert fn(c) != fn(cfg) and fn(cfg) == fn(copy.deepcopy(cfg))


def test_write_csv_sidecar_provenance(cfg, tmp_path):
    path = da.write_csv(pd.DataFrame({"a": [1, 2]}), tmp_path / "t.csv", cfg, "test", "note", q_hat={1: 2.0})
    meta = json.loads((tmp_path / "t.meta.json").read_text())
    assert path.exists() and meta["dataset_sha256"] == cfg["data"]["sha256"]
    assert meta["kaggle_version"] == 12 and meta["config_hash"] == da.config_hash(cfg)
    assert meta["n_rows"] == 2 and meta["q_hat"] == {"1": 2.0} and "versions" in meta


@pytest.mark.parametrize("name", ["pm25_completeness_by_city_year", "pm25_completeness_by_city_period",
                                  "pm25_gap_stats"])
def test_saved_audit_tables_reproduce(cfg, dirs, cal, name):
    fn = {"pm25_completeness_by_city_year": lambda: da.completeness_by_year(cal),
          "pm25_completeness_by_city_period": lambda: da.completeness_by_period(cal, cfg),
          "pm25_gap_stats": lambda: da.gap_stats(cal, cfg)}[name]
    saved = pd.read_csv(dirs["tables"] / f"{name}.csv")
    pd.testing.assert_frame_equal(pd.read_csv(io.StringIO(fn().to_csv(index=False))), saved)
