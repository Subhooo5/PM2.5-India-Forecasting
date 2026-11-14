"""Day 1 items 8-9: baselines on the common evaluation sample; processed-data export."""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import features as F
from src.models import baseline_predictions, persistence, seasonal_naive


# ---------------------------------------------------------------- item 8
def test_persistence_is_origin_value(ds):
    np.testing.assert_array_equal(persistence(ds).to_numpy(), ds["pm25_origin"].to_numpy())


def test_seasonal_naive_is_raw_value_seven_days_before_target(ds, raw_lookup):
    ref_dates = ds["target_date"] - pd.Timedelta(days=7)
    raw = raw_lookup.reindex(pd.MultiIndex.from_arrays([ds["city"].astype(str), ref_dates]))
    np.testing.assert_array_equal(seasonal_naive(ds).to_numpy(), raw.to_numpy())  # NaN == NaN
    assert (ref_dates <= ds["origin_date"]).all()          # causal for h <= 7


def test_h7_seasonal_naive_equals_persistence(ds):
    h7 = ds[ds["horizon"] == 7]
    np.testing.assert_array_equal(seasonal_naive(h7).to_numpy(), persistence(h7).to_numpy())


def test_common_evaluation_sample(ds):
    m = ds["eval_mask"]
    assert (m == (ds["y_target"].notna() & ds["pm25_origin"].notna() & ds["snaive_ref"].notna())).all()
    preds = baseline_predictions(ds)
    assert len(preds) == int(m.sum())
    assert preds[["y_target", "yhat_persistence", "yhat_seasonal_naive"]].notna().all().all()
    # both baselines are scored on exactly the same rows
    assert preds["yhat_persistence"].index.equals(preds["yhat_seasonal_naive"].index)


# ---------------------------------------------------------------- item 9
def test_export_roundtrip_and_provenance(ds, cfg, tmp_path):
    path = F.export_dataset(ds, cfg, tmp_path / "forecast_dataset.parquet")
    back = pd.read_parquet(path)
    pd.testing.assert_frame_equal(back, ds, check_dtype=False, check_categorical=False)
    assert isinstance(back["city"].dtype, pd.CategoricalDtype)
    meta = json.loads(pq.read_schema(path).metadata[b"pm25_provenance"])
    assert meta["dataset_sha256"] == cfg["data"]["sha256"]
    assert meta["kaggle_version"] == 12
    side = json.loads((tmp_path / "forecast_dataset.meta.json").read_text())
    assert side["dataset_sha256"] == cfg["data"]["sha256"]
    assert side["strict_features"] == F.strict_feature_columns(cfg)


def test_usable_samples_table_consistent(ds):
    u = F.usable_samples_table(ds)
    main = u[u["scheme"] == "main"]
    in_main = ds[ds["split_main"] != "outside"]
    assert main["n_samples"].sum() == len(in_main)
    assert main["n_eval_mask"].sum() == int(in_main["eval_mask"].sum())
    stress = u[u["scheme"] == "stress"]
    assert stress["n_samples"].sum() == int((ds["split_stress"] != "outside").sum())
