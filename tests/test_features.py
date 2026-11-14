"""Day 1 items 4-6: city selection, causal features, availability at the forecast origin.

Required checks covered here:
 (a) features at origin t never use data after t (perturbation test + a leaky-control test)
 (d) the target (and y_t) are never imputed
 (e) AQI / AQI_Bucket / co-pollutants are absent from the strict feature set
"""
import numpy as np
import pandas as pd
import pytest

from src import data_audit as da
from src import features as F
from src.preprocessing import city_daily_series, ffill_inputs, select_cities

ALL_CITIES = ["Delhi", "Lucknow", "Chennai", "Hyderabad", "Bengaluru",
              "Kolkata", "Mumbai", "Visakhapatnam"]


# ---------------------------------------------------------------- item 4: cities
def test_city_roles(cfg):
    roles = select_cities(cfg)
    assert [c for c, r in roles.items() if r == "core"] == \
        ["Delhi", "Lucknow", "Chennai", "Hyderabad", "Bengaluru"]
    assert {c for c, r in roles.items() if r == "transfer"} == {"Kolkata", "Mumbai"}
    assert roles["Visakhapatnam"] == "transfer_optional"
    assert "Patna" not in roles and "Gurugram" not in roles


def test_core_cities_meet_training_coverage(raw, cfg):
    per = da.completeness_by_period(da.pm25_calendar(raw, cfg), cfg).set_index("City")
    for c in cfg["cities"]["core"]:
        assert per.loc[c, "train_2015_2017_pct"] >= cfg["audit"]["core_min_train_pct"], c


def test_dataset_cities_and_roles(ds, cfg):
    assert set(ds["city"].unique()) == set(ALL_CITIES)
    roles = select_cities(cfg)
    assert (ds["city"].astype(str).map(roles) == ds["city_role"].astype(str)).all()
    # transfer cities have no observed PM2.5 before April 2018 -> no main-train samples
    tr = ds[ds["city"].isin(["Kolkata", "Mumbai"])]
    assert not (tr["split_main"] == "train").any()


# ------------------------------------------- (a) causality: perturb the future
def _assert_causal(feature_fn, s, origins, rng):
    """feature_fn(series) -> DataFrame indexed by date. Rows <= t must not change when
    every value after t is scrambled (random values, random NaNs, previously-NaN filled)."""
    base = feature_fn(s)
    for t in origins:
        p = s.copy()
        after = p.index > t
        p[after] = rng.uniform(0, 1000, after.sum())
        p[after & (rng.random(len(p)) < 0.3)] = np.nan
        pert = feature_fn(p)
        pd.testing.assert_frame_equal(base.loc[:t], pert.loc[:t], check_exact=True)


def _origins(s, rng, n=40):
    obs = s.index[s.notna()]
    picks = list(rng.choice(obs[14:-14], n, replace=False))
    # also origins right before / inside gaps, where ffill and min_obs rules kick in
    gap_edges = s.index[(s.isna() & s.shift(1).notna()).to_numpy()]
    return picks + list(gap_edges[:10]) + [s.index[0], s.index[-1]]


@pytest.mark.parametrize("city", ALL_CITIES)
def test_origin_features_do_not_use_future(raw, cfg, city):
    rng = np.random.default_rng(cfg["seed"])
    s = city_daily_series(raw, city, cfg)
    _assert_causal(lambda x: F.origin_features(x, cfg), s, _origins(s, rng), rng)


@pytest.mark.parametrize("city", ["Delhi", "Chennai", "Mumbai"])
def test_full_samples_do_not_use_future(raw, cfg, city):
    """End-to-end: every input column of a sample with origin <= t is unchanged when the
    future is scrambled (targets may change; that is expected)."""
    rng = np.random.default_rng(cfg["seed"] + 1)
    s = city_daily_series(raw, city, cfg)
    inputs = [c for c in F.strict_feature_columns(cfg) if c != "city"] + ["snaive_ref", "target_date"]
    base = F.build_city_samples(s, city, cfg)
    for t in rng.choice(s.index[s.notna()][30:-30], 10, replace=False):
        p = s.copy()
        p[p.index > t] = rng.uniform(0, 1000, (p.index > t).sum())
        pert = F.build_city_samples(p, city, cfg)
        key = ["origin_date", "horizon"]
        b = base[base["origin_date"] <= t].set_index(key)[inputs]
        q = pert[pert["origin_date"] <= t].set_index(key)[inputs]
        common = b.index.intersection(q.index)
        assert len(common) > 0
        pd.testing.assert_frame_equal(b.loc[common], q.loc[common], check_exact=True)


def test_causality_check_catches_leaks(raw, cfg):
    """Control: the perturbation harness must FAIL for deliberately leaky features."""
    rng = np.random.default_rng(0)
    s = city_daily_series(raw, "Delhi", cfg)
    leaky = {
        "lead_1": lambda x: x.shift(-1).to_frame("f"),
        "centered_rolling": lambda x: x.rolling(7, center=True, min_periods=1).mean().to_frame("f"),
        "backfill": lambda x: x.bfill(limit=2).to_frame("f"),
    }
    for name, fn in leaky.items():
        with pytest.raises(AssertionError):
            _assert_causal(fn, s, _origins(s, rng, 10), rng)


# ------------------------------------------------- alignment of each feature
def test_feature_values_match_definitions(raw, cfg, ds):
    rng = np.random.default_rng(cfg["seed"])
    lim = cfg["preprocessing"]["ffill_limit"]
    for city in ALL_CITIES:
        s = city_daily_series(raw, city, cfg)
        filled = ffill_inputs(s, lim)
        sub = ds[ds["city"] == city]
        for _, r in sub.iloc[rng.choice(len(sub), 60, replace=False)].iterrows():
            t, h = r["origin_date"], int(r["horizon"])
            assert r["target_date"] - t == pd.Timedelta(days=h)
            assert r["pm25_origin"] == s[t]
            assert r["y_target"] == s[r["target_date"]]
            ref = s[r["target_date"] - pd.Timedelta(days=7)]
            assert (np.isnan(ref) and np.isnan(r["snaive_ref"])) or r["snaive_ref"] == ref
            for k in cfg["features"]["lags"]:
                d = t - pd.Timedelta(days=k)
                exp = filled.get(d, np.nan)
                got = r[f"pm25_lag{k}"]
                assert (np.isnan(exp) and np.isnan(got)) or got == exp
                was_filled = int(np.isnan(s.get(d, np.nan)) and not np.isnan(exp))
                assert r[f"pm25_lag{k}_imputed"] == was_filled
            for w, m in cfg["features"]["rolling_min_obs"].items():
                win = s.loc[t - pd.Timedelta(days=w - 1): t].dropna()   # raw, [t-w+1, t]
                if len(win) >= m:
                    assert r[f"pm25_rmean{w}"] == pytest.approx(win.mean(), rel=1e-9)
                    assert r[f"pm25_rstd{w}"] == pytest.approx(win.std(ddof=1), rel=1e-9, nan_ok=True)
                else:
                    assert np.isnan(r[f"pm25_rmean{w}"]) and np.isnan(r[f"pm25_rstd{w}"])
            td = r["target_date"]
            assert r["tgt_month"] == td.month and r["tgt_dow"] == td.day_of_week


def test_ffill_limit_is_respected(raw, cfg):
    s = city_daily_series(raw, "Visakhapatnam", cfg)   # has long gaps
    lim = cfg["preprocessing"]["ffill_limit"]
    filled = ffill_inputs(s, lim)
    run = s.isna().groupby(s.notna().cumsum()).cumsum()  # position inside each NaN run
    assert filled[run > lim].isna().all()
    assert filled[(run >= 1) & (run <= lim) & (s.ffill().notna())].notna().all()


# --------------------------------------------- (d) target / origin never imputed
def test_target_and_origin_never_imputed(ds, raw_lookup):
    tgt = raw_lookup.reindex(pd.MultiIndex.from_arrays([ds["city"].astype(str), ds["target_date"]]))
    org = raw_lookup.reindex(pd.MultiIndex.from_arrays([ds["city"].astype(str), ds["origin_date"]]))
    assert tgt.notna().all() and org.notna().all()          # both genuinely observed
    np.testing.assert_array_equal(ds["y_target"].to_numpy(), tgt.to_numpy())
    np.testing.assert_array_equal(ds["pm25_origin"].to_numpy(), org.to_numpy())


def test_target_independent_of_fill_policy(raw, cfg):
    """Changing the input fill limit must not change any target or the sample set."""
    big = {**cfg, "preprocessing": {**cfg["preprocessing"], "ffill_limit": 1000}}
    zero = {**cfg, "preprocessing": {**cfg["preprocessing"], "ffill_limit": 0}}
    a, b = F.build_dataset(raw, big), F.build_dataset(raw, zero)
    key = ["city", "origin_date", "horizon", "target_date", "y_target", "pm25_origin"]
    pd.testing.assert_frame_equal(a[key], b[key])


def test_missing_origin_samples_dropped(raw, cfg, ds):
    for city in ALL_CITIES:
        s = city_daily_series(raw, city, cfg)
        origins = set(ds.loc[ds["city"] == city, "origin_date"])
        assert origins.isdisjoint(set(s.index[s.isna()]))


# ------------------------------------------- (e) forbidden columns are absent
def test_strict_features_exclude_aqi_and_copollutants(cfg, ds):
    forbidden = set(cfg["features"]["forbidden"])
    assert {"AQI", "AQI_Bucket", "PM10", "NO2", "CO", "SO2", "O3"} <= forbidden
    strict = set(F.strict_feature_columns(cfg))
    assert strict.isdisjoint(forbidden)
    assert set(ds.columns).isdisjoint(forbidden)
    assert not any("aqi" in c.lower() for c in strict | set(ds.columns))


def test_strict_feature_set_is_exactly_the_frozen_spec(cfg):
    expected = {"pm25_origin", "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
                "pm25_lag1_imputed", "pm25_lag2_imputed", "pm25_lag3_imputed", "pm25_lag7_imputed",
                "pm25_rmean3", "pm25_rstd3", "pm25_rmean7", "pm25_rstd7",
                "pm25_rmean14", "pm25_rstd14",
                "tgt_month", "tgt_dow", "tgt_doy_sin", "tgt_doy_cos", "city"}
    assert set(F.strict_feature_columns(cfg)) == expected


def test_city_is_categorical(ds):
    assert isinstance(ds["city"].dtype, pd.CategoricalDtype)
