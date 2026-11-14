"""Day 1 items 1-3: executable checks on the raw data and the completeness audit."""
import pandas as pd
import pytest

from src import data_audit as da


# ---- item 1: load -------------------------------------------------------------
def test_load_does_not_modify_raw(cfg):
    path = da.resolve(cfg, "raw_city_day")
    before = da.sha256_file(path)
    da.load_city_day(cfg)
    assert da.sha256_file(path) == before == cfg["data"]["sha256"]


def test_loaded_types(raw):
    assert pd.api.types.is_datetime64_any_dtype(raw["Date"])
    assert pd.api.types.is_float_dtype(raw["PM2.5"])


# ---- item 2: verify -----------------------------------------------------------
def test_verify_raw_passes(raw, cfg):
    s = da.verify_raw(raw, cfg)
    assert s["n_rows"] == 29531 and s["n_cities"] == 26
    assert (s["date_min"], s["date_max"]) == ("2015-01-01", "2020-07-01")


@pytest.mark.parametrize("tamper", [
    lambda d: d.iloc[1:],                                    # row count
    lambda d: d.drop(columns=["AQI_Bucket"]),                # columns
    lambda d: d[d["City"] != "Delhi"],                       # cities
    lambda d: pd.concat([d, d.iloc[[0]]]),                   # duplicate (City, Date)
    lambda d: d[d["Date"] < "2020-07-01"],                   # date range
])
def test_verify_raw_rejects_tampered(raw, cfg, tamper):
    with pytest.raises(da.RawDataError):
        da.verify_raw(tamper(raw.copy()), cfg)


def test_verify_raw_rejects_wrong_hash(raw, cfg, tmp_path):
    bad = tmp_path / "city_day.csv"
    bad.write_bytes(da.resolve(cfg, "raw_city_day").read_bytes() + b"\n")
    with pytest.raises(da.RawDataError, match="sha256"):
        da.verify_raw(raw, cfg, path=bad)


# ---- item 3: completeness ------------------------------------------------------
@pytest.fixture(scope="module")
def cal(raw, cfg):
    return da.pm25_calendar(raw, cfg)


def test_calendar_is_full_daily_grid(cal, raw):
    n_days = (raw["Date"].max() - raw["Date"].min()).days + 1
    assert len(cal) == 26 * n_days
    assert not cal.duplicated(["City", "Date"]).any()


def test_calendar_preserves_every_observation(cal, raw):
    assert cal["pm25"].count() == raw["PM2.5"].count()
    merged = raw[["City", "Date", "PM2.5"]].merge(cal, on=["City", "Date"])
    pd.testing.assert_series_equal(merged["PM2.5"], merged["pm25"], check_names=False)


def test_completeness_by_year_invariants(cal):
    t = da.completeness_by_year(cal)
    assert (t["pm25_observed"] <= t["rows_present"]).all()
    assert (t["rows_present"] <= t["expected_days"]).all()
    assert t["pct_observed"].between(0, 100).all()
    days = t.groupby("year")["expected_days"].first()
    assert days[2016] == 366 and days[2019] == 365 and days[2020] == 183
    assert t["pm25_observed"].sum() == cal["pm25"].count()


def test_completeness_by_period_bounds(cal, cfg):
    t = da.completeness_by_period(cal, cfg)
    assert len(t) == 26
    for c in [c for c in t.columns if c.endswith("_pct")]:
        assert t[c].between(0, 100).all()


def test_gap_runs_on_known_series():
    #           lead  obs  gap1 obs gap2  obs gap4         obs  trail
    s = [False, False, True, False, True, False, False, True, False, False, False, False, True, False]
    lead, trail, gaps = da.gap_runs(pd.Series(s))
    assert (lead, trail, gaps) == (2, 1, [1, 2, 4])


def test_gap_stats_accounting(cal, cfg):
    g = da.gap_stats(cal, cfg)
    n_days = cal.groupby("City").size()
    obs = cal.groupby("City")["pm25"].count()
    total = (g.set_index("City")[["leading_missing_days", "trailing_missing_days",
                                  "internal_missing_days"]].sum(axis=1) + obs)
    assert (total == n_days).all()
    bins = g[["gaps_len_1", "gaps_len_2", "gaps_len_3_7", "gaps_len_gt7"]].sum(axis=1)
    assert (bins == g["n_internal_gaps"]).all()
