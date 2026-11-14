"""Explicit return-type checks (guards against e.g. accidental trailing commas -> tuples)."""
import pandas as pd

from src.features import encode_city, feature_set
from src.models import baseline_predictions, lightgbm_params, persistence, seasonal_naive
from src.preprocessing import city_daily_series, ffill_inputs, select_cities
from src.splits import random_split_unique


def test_ffill_inputs_returns_series(raw, cfg):
    s = city_daily_series(raw, "Delhi", cfg)
    assert isinstance(s, pd.Series)
    for lim in (0, 2, 3):
        out = ffill_inputs(s, lim)
        assert type(out) is pd.Series and not isinstance(out, tuple)
        assert out.index.equals(s.index)
    assert ffill_inputs(s, 0) is not s                     # a copy, never the same object


def test_baseline_predictions_returns_dataframe(ds):
    for eval_only in (True, False):
        out = baseline_predictions(ds, eval_only=eval_only)
        assert type(out) is pd.DataFrame
        assert {"yhat_persistence", "yhat_seasonal_naive", "y_target"} <= set(out.columns)
    assert type(persistence(ds)) is pd.Series and type(seasonal_naive(ds)) is pd.Series


def test_other_return_types(ds, cfg):
    assert type(select_cities(cfg)) is dict
    assert type(feature_set(cfg, "causal_common")) is list
    assert type(lightgbm_params({**cfg, "tuning": {**cfg["tuning"], "mode": "fixed"}})) is dict
    assert type(encode_city(ds, cfg["cities"]["core"])) is pd.Series
    units = ds.loc[ds["horizon"] == 1, ["city", "target_date"]]
    assert type(random_split_unique(units, 0.2, 0)) is pd.Series
