import numpy as np
import pandas as pd
import pytest

from src import data_audit as da
from src import features as F
from src.models import baseline_predictions
from src.preprocessing import city_daily_series, ffill_inputs, select_cities

ALL_CITIES = ["Delhi", "Lucknow", "Chennai", "Hyderabad", "Bengaluru", "Kolkata", "Mumbai", "Visakhapatnam"]
CAUSAL_COMMON = ["pm25_origin", "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7", "pm25_lag1_imputed",
                 "pm25_lag2_imputed", "pm25_lag3_imputed", "pm25_lag7_imputed", "pm25_rmean3", "pm25_rstd3",
                 "pm25_rmean7", "pm25_rstd7", "pm25_rmean14", "pm25_rstd14", "tgt_month", "tgt_dow",
                 "tgt_doy_sin", "tgt_doy_cos"]


def test_city_roles_and_training_coverage(raw, cfg, ds):
    roles = select_cities(cfg)
    assert [c for c, r in roles.items() if r == "core"] == ["Delhi", "Lucknow", "Chennai", "Hyderabad", "Bengaluru"]
    assert {c for c, r in roles.items() if r == "transfer"} == {"Kolkata", "Mumbai"}
    assert roles["Visakhapatnam"] == "transfer_optional" and set(ds["city"].unique()) == set(ALL_CITIES)
    assert (ds["city"].astype(str).map(roles) == ds["city_role"].astype(str)).all()
    assert not (ds[ds["city"].isin(["Kolkata", "Mumbai"])]["split_main"] == "train").any()
    per = da.completeness_by_period(da.pm25_calendar(raw, cfg), cfg).set_index("City")
    assert (per.loc[cfg["cities"]["core"], "train_2015_2017_pct"] >= cfg["audit"]["core_min_train_pct"]).all()


def _assert_causal(feature_fn, s, origins, rng):
    base = feature_fn(s)
    for t in origins:
        p = s.copy()
        after = p.index > t
        p[after] = rng.uniform(0, 1000, after.sum())
        p[after & (rng.random(len(p)) < 0.3)] = np.nan
        pd.testing.assert_frame_equal(base.loc[:t], feature_fn(p).loc[:t], check_exact=True)


def _origins(s, rng, n=40):
    picks = list(rng.choice(s.index[s.notna()][14:-14], n, replace=False))
    gap_edges = s.index[(s.isna() & s.shift(1).notna()).to_numpy()]
    return picks + list(gap_edges[:10]) + [s.index[0], s.index[-1]]


@pytest.mark.parametrize("city", ALL_CITIES)
def test_origin_features_do_not_use_future(raw, cfg, city):
    rng = np.random.default_rng(cfg["seed"])
    s = city_daily_series(raw, city, cfg)
    _assert_causal(lambda x: F.origin_features(x, cfg), s, _origins(s, rng), rng)


@pytest.mark.parametrize("city", ["Delhi", "Chennai", "Mumbai"])
def test_full_samples_do_not_use_future(raw, cfg, city):
    rng = np.random.default_rng(cfg["seed"] + 1)
    s = city_daily_series(raw, city, cfg)
    inputs, key = F.causal_columns(cfg) + ["snaive_ref", "target_date"], ["origin_date", "horizon"]
    base = F.build_city_samples(s, city, cfg)
    for t in rng.choice(s.index[s.notna()][30:-30], 10, replace=False):
        p = s.copy()
        p[p.index > t] = rng.uniform(0, 1000, (p.index > t).sum())
        pert = F.build_city_samples(p, city, cfg)
        b, q = (d[d["origin_date"] <= t].set_index(key)[inputs] for d in (base, pert))
        common = b.index.intersection(q.index)
        assert len(common) > 0
        pd.testing.assert_frame_equal(b.loc[common], q.loc[common], check_exact=True)


@pytest.mark.parametrize("leaky", [lambda x: x.shift(-1).to_frame("f"),
                                   lambda x: x.rolling(7, center=True, min_periods=1).mean().to_frame("f"),
                                   lambda x: x.bfill(limit=2).to_frame("f")])
def test_causality_check_catches_leaks(raw, cfg, leaky):
    rng = np.random.default_rng(0)
    s = city_daily_series(raw, "Delhi", cfg)
    with pytest.raises(AssertionError):
        _assert_causal(leaky, s, _origins(s, rng, 10), rng)


def test_feature_values_match_definitions(raw, cfg, ds):
    rng = np.random.default_rng(cfg["seed"])
    for city in ALL_CITIES:
        s = city_daily_series(raw, city, cfg)
        filled = ffill_inputs(s, cfg["preprocessing"]["ffill_limit"])
        sub = ds[ds["city"] == city]
        for _, r in sub.iloc[rng.choice(len(sub), 60, replace=False)].iterrows():
            t, td = r["origin_date"], r["target_date"]
            assert td - t == pd.Timedelta(days=int(r["horizon"]))
            assert r["pm25_origin"] == s[t] and r["y_target"] == s[td]
            ref = s[td - pd.Timedelta(days=7)]
            assert (np.isnan(ref) and np.isnan(r["snaive_ref"])) or r["snaive_ref"] == ref
            for k in cfg["features"]["lags"]:
                d = t - pd.Timedelta(days=k)
                exp, got = filled.get(d, np.nan), r[f"pm25_lag{k}"]
                assert (np.isnan(exp) and np.isnan(got)) or got == exp
                assert r[f"pm25_lag{k}_imputed"] == int(np.isnan(s.get(d, np.nan)) and not np.isnan(exp))
            for w, m in cfg["features"]["rolling_min_obs"].items():
                win = s.loc[t - pd.Timedelta(days=w - 1): t].dropna()
                if len(win) >= m:
                    assert r[f"pm25_rmean{w}"] == pytest.approx(win.mean(), rel=1e-9)
                    assert r[f"pm25_rstd{w}"] == pytest.approx(win.std(ddof=1), rel=1e-9, nan_ok=True)
                else:
                    assert np.isnan(r[f"pm25_rmean{w}"]) and np.isnan(r[f"pm25_rstd{w}"])
            assert r["tgt_month"] == td.month and r["tgt_dow"] == td.day_of_week


def test_ffill_limit_is_respected(raw, cfg):
    s = city_daily_series(raw, "Visakhapatnam", cfg)
    lim = cfg["preprocessing"]["ffill_limit"]
    filled = ffill_inputs(s, lim)
    run = s.isna().astype("int64").groupby(s.notna().cumsum()).cumsum()
    assert isinstance(filled, pd.Series) and filled[run > lim].isna().all()
    assert filled[(run >= 1) & (run <= lim) & s.ffill().notna()].notna().all()


def test_target_and_origin_never_imputed(ds, raw_lookup):
    for col, date in (("y_target", "target_date"), ("pm25_origin", "origin_date")):
        ref = raw_lookup.reindex(pd.MultiIndex.from_arrays([ds["city"].astype(str), ds[date]]))
        assert ref.notna().all()
        np.testing.assert_array_equal(ds[col].to_numpy(), ref.to_numpy())


def test_target_independent_of_fill_policy(raw, cfg):
    a, b = (F.build_dataset(raw, {**cfg, "preprocessing": {"ffill_limit": lim}}) for lim in (1000, 0))
    key = ["city", "origin_date", "horizon", "target_date", "y_target", "pm25_origin"]
    pd.testing.assert_frame_equal(a[key], b[key])


def test_missing_origin_samples_dropped(raw, cfg, ds):
    for city in ALL_CITIES:
        s = city_daily_series(raw, city, cfg)
        assert set(ds.loc[ds["city"] == city, "origin_date"]).isdisjoint(set(s.index[s.isna()]))


def test_causal_columns_are_the_frozen_set_without_forbidden(cfg, ds):
    forbidden = set(cfg["features"]["forbidden"])
    assert F.causal_columns(cfg) == CAUSAL_COMMON == F.feature_set(cfg, "causal_common")
    assert F.feature_set(cfg, "causal_city_aware") == CAUSAL_COMMON + ["city"]
    assert set(CAUSAL_COMMON).isdisjoint(forbidden) and set(ds.columns).isdisjoint(forbidden)
    assert isinstance(ds["city"].dtype, pd.CategoricalDtype)


def test_encode_city_marks_unseen_cities_missing():
    df = pd.DataFrame({"city": ["Delhi", "Mumbai", "Chennai"]})
    enc = F.encode_city(df, ["Chennai", "Delhi"])
    assert list(enc.cat.categories) == ["Chennai", "Delhi"] and enc.isna().tolist() == [False, True, False]


def test_baselines_and_eval_mask(ds):
    b = baseline_predictions(ds)
    assert isinstance(b, pd.DataFrame)
    np.testing.assert_array_equal(b["yhat_persistence"], ds["pm25_origin"])
    np.testing.assert_array_equal(b["yhat_seasonal_naive"], ds["snaive_ref"])
    h7 = ds[ds["horizon"] == 7]
    np.testing.assert_array_equal(h7["snaive_ref"], h7["pm25_origin"])
    assert (ds["eval_mask"] == (ds["y_target"].notna() & ds["pm25_origin"].notna() & ds["snaive_ref"].notna())).all()


def test_same_day_table(same_day, raw, cfg):
    assert not same_day.duplicated(["city", "target_date"]).any() and same_day["y_target"].notna().all()
    assert list(same_day.columns) == ["city", "target_date", *F.feature_set(cfg, F.SAME_DAY_SET), "y_target",
                                      "split_A", "split_B"]
    m = same_day.merge(raw.rename(columns={"City": "city", "Date": "target_date"}), on=["city", "target_date"],
                       suffixes=("", "_raw"))
    np.testing.assert_array_equal(m["y_target"], m["PM2.5"])
    np.testing.assert_array_equal(m["AQI"], m["AQI_raw"])
    assert "AQI_Bucket" not in same_day.columns


@pytest.mark.parametrize("key, reader", [("forecast_dataset", "ds"), ("same_day_dataset", "same_day")])
def test_saved_parquet_equals_rebuild(cfg, request, key, reader):
    saved = pd.read_parquet(da.resolve(cfg, key))
    pd.testing.assert_frame_equal(saved, request.getfixturevalue(reader), check_categorical=False)


def test_saved_descriptive_tables_reproduce(ds, same_day, cfg, dirs):
    for name, t in (("usable_samples_by_city_split_horizon", F.usable_samples_table(ds)),
                    ("feature_missingness_by_city_horizon_split", F.feature_missingness_table(ds, cfg)),
                    ("leakage_same_day_split_counts", F.same_day_split_table(same_day))):
        saved = pd.read_csv(dirs["tables"] / f"{name}.csv", keep_default_na=False, na_values=[""])
        assert len(saved) == len(t) and list(saved.columns) == list(t.columns)
        num = saved.select_dtypes("number").columns
        np.testing.assert_allclose(saved[num].to_numpy(float), t[num].to_numpy(float), rtol=0, atol=1e-12)
