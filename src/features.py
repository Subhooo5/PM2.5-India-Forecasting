"""Strictly causal feature construction and Day 1 dataset build (items 4, 5, 8-support, 9).

Forecast origin t: X_t -> y_{t+h}, direct, h in {1, 3, 7}. Every feature is knowable at t:
- pm25_origin          = y_t (raw; samples with missing y_t are DROPPED)
- pm25_lag{k}          = y_{t-k}, k in {1,2,3,7}, forward-filled <= ffill_limit days, else NaN
- pm25_lag{k}_imputed  = 1 if that lag value came from the forward-fill
- pm25_rmean{w}/rstd{w} over raw observed values in [t-w+1, t], min obs 2/3, 4/7, 7/14
- tgt_month, tgt_dow, tgt_doy_sin, tgt_doy_cos of the TARGET date (known in advance)
- city (categorical)
Never: AQI, AQI_Bucket, co-pollutants (config `features.forbidden`).

Non-feature columns: city_role, origin_date, target_date, horizon, y_target (raw, never
filled), snaive_ref (= raw y[t+h-7], seasonal-naive input), eval_mask, split labels.

Run:  python -m src.features      (verifies raw data, builds, exports parquet + audit tables)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import data_audit as da
from src.preprocessing import city_daily_series, ffill_inputs, select_cities
from src.splits import assign_splits


# ------------------------------------------------------------------ column sets
def lag_columns(cfg: dict) -> list[str]:
    return [f"pm25_lag{k}" for k in cfg["features"]["lags"]]


def flag_columns(cfg: dict) -> list[str]:
    return [f"{c}_imputed" for c in lag_columns(cfg)]


def rolling_columns(cfg: dict) -> list[str]:
    return [f"pm25_{stat}{w}" for w in cfg["features"]["rolling_windows"]
            for stat in ("rmean", "rstd")]


def origin_feature_columns(cfg: dict) -> list[str]:
    """Columns that depend only on the PM2.5 history up to origin t."""
    return ["pm25_origin", *lag_columns(cfg), *flag_columns(cfg), *rolling_columns(cfg)]


def strict_feature_columns(cfg: dict) -> list[str]:
    """Protocol C (strictly causal) model inputs."""
    return [*origin_feature_columns(cfg), *cfg["features"]["calendar"], "city"]


# ------------------------------------------------------------------ features
def origin_features(s_raw: pd.Series, cfg: dict) -> pd.DataFrame:
    """Per-origin-date history features for one city (index = origin date t).

    `s_raw` must be the raw daily series on a complete calendar (NaN = unobserved).
    """
    if not s_raw.index.equals(pd.date_range(s_raw.index[0], s_raw.index[-1], freq="D")):
        raise ValueError("series must be on a complete daily calendar")
    f = cfg["features"]
    filled = ffill_inputs(s_raw, cfg["preprocessing"]["ffill_limit"])
    out = pd.DataFrame(index=s_raw.index)
    out["pm25_origin"] = s_raw                                 # never filled
    for k in f["lags"]:
        out[f"pm25_lag{k}"] = filled.shift(k)                  # value at t-k (past only)
    for k in f["lags"]:
        out[f"pm25_lag{k}_imputed"] = (s_raw.shift(k).isna() & filled.shift(k).notna()).astype("int8")
    for w in f["rolling_windows"]:
        r = s_raw.rolling(window=w, min_periods=int(f["rolling_min_obs"][w]))  # [t-w+1, t]
        out[f"pm25_rmean{w}"] = r.mean()
        out[f"pm25_rstd{w}"] = r.std()                         # ddof=1; needs >=2 obs
    return out


def calendar_features(target_date: pd.Series) -> pd.DataFrame:
    """Calendar features of the TARGET date (deterministic, known in advance)."""
    td = pd.DatetimeIndex(target_date)
    angle = 2 * np.pi * td.day_of_year.to_numpy() / 365.25
    return pd.DataFrame({
        "tgt_month": td.month.astype("int8"),
        "tgt_dow": td.day_of_week.astype("int8"),
        "tgt_doy_sin": np.sin(angle),
        "tgt_doy_cos": np.cos(angle),
    }, index=target_date.index)


def build_city_samples(s_raw: pd.Series, city: str, cfg: dict) -> pd.DataFrame:
    """Long samples (one row per origin x horizon) for one city.

    Keeps a sample only if y_t AND y_{t+h} are observed (neither is ever imputed).
    """
    period = cfg["baselines"]["seasonal_period"]
    X = origin_features(s_raw, cfg)
    X = X.loc[X["pm25_origin"].notna()]
    parts = []
    for h in cfg["features"]["horizons"]:
        if not 1 <= h <= period:
            raise ValueError(f"horizon {h}: seasonal naive y[t+h-{period}] would not be causal")
        target_date = X.index + pd.Timedelta(days=h)
        part = X.reset_index(names="origin_date")
        part.insert(1, "target_date", target_date)
        part.insert(2, "horizon", np.int8(h))
        part["y_target"] = s_raw.reindex(target_date).to_numpy()           # raw only
        part["snaive_ref"] = s_raw.reindex(target_date - pd.Timedelta(days=period)).to_numpy()
        parts.append(part.loc[part["y_target"].notna()])
    out = pd.concat(parts, ignore_index=True)
    out = pd.concat([out, calendar_features(out["target_date"])], axis=1)
    out.insert(0, "city", city)
    out["eval_mask"] = (out["y_target"].notna() & out["pm25_origin"].notna()
                        & out["snaive_ref"].notna())
    return out


def build_dataset(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """All selected cities -> one long table with split labels (Day 1 items 4, 5, 7)."""
    roles = select_cities(cfg)
    parts = [build_city_samples(city_daily_series(df, c, cfg), c, cfg) for c in roles]
    ds = pd.concat(parts, ignore_index=True)
    ds["city"] = pd.Categorical(ds["city"], categories=sorted(roles))
    ds.insert(1, "city_role", pd.Categorical(ds["city"].map(roles).astype(str),
                                             categories=["core", "transfer", "transfer_optional"]))
    ds = assign_splits(ds, cfg)
    first = ["city", "city_role", "origin_date", "target_date", "horizon"]
    feats = [c for c in strict_feature_columns(cfg) if c != "city"]
    rest = ["y_target", "snaive_ref", "eval_mask", "split_main", "split_stress", "stress_period"]
    ds = ds[first + feats + rest].sort_values(["city", "horizon", "origin_date"],
                                              ignore_index=True)
    forbidden = set(cfg["features"]["forbidden"]) & set(ds.columns)
    if forbidden:
        raise AssertionError(f"forbidden columns in strict dataset: {forbidden}")
    return ds


# ------------------------------------------------------------------ audit tables
def usable_samples_table(ds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme in ("main", "stress"):
        col = f"split_{scheme}"
        keys = ["city", "city_role", "horizon", col] + (["stress_period"] if scheme == "stress" else [])
        g = ds.groupby(keys, observed=True)
        t = pd.DataFrame({"n_samples": g.size(), "n_eval_mask": g["eval_mask"].sum()}).reset_index()
        t = t.rename(columns={col: "split"}).loc[lambda x: x["split"] != "outside"]
        t.insert(0, "scheme", scheme)
        rows.append(t)
    out = pd.concat(rows, ignore_index=True)
    out["stress_period"] = out.get("stress_period", pd.Series(dtype=str)).fillna("").astype(str)
    order = {"train": 0, "calibration": 1, "test": 2}
    return (out.assign(_o=out["split"].astype(str).map(order))
               .sort_values(["scheme", "city_role", "city", "horizon", "_o", "stress_period"])
               .drop(columns="_o").reset_index(drop=True))


def feature_missingness_table(ds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per city x horizon x main split: % NaN per feature and % forward-filled per lag."""
    feats = lag_columns(cfg) + rolling_columns(cfg)
    g = ds.groupby(["city", "horizon", "split_main"], observed=True)
    nan_pct = g[feats].agg(lambda x: round(100 * x.isna().mean(), 2)).add_suffix("_nan_pct")
    imp_pct = g[flag_columns(cfg)].mean().mul(100).round(2).add_suffix("_pct")
    n = g.size().rename("n_samples")
    return pd.concat([n, nan_pct, imp_pct], axis=1).reset_index()


# ------------------------------------------------------------------ export (item 9)
def export_dataset(ds: pd.DataFrame, cfg: dict, path=None):
    """Write the processed parquet with provenance in schema metadata + .meta.json sidecar."""
    path = da.ROOT / cfg["paths"]["processed_dataset"] if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = da.build_metadata(ds, path.name, cfg, "src.features.build_dataset",
                             notes="Strict (protocol C) causal features; splits by target_date.")
    meta.update(strict_features=strict_feature_columns(cfg),
                features_config=cfg["features"], preprocessing_config=cfg["preprocessing"],
                splits_config=cfg["splits"], cities=cfg["cities"])
    table = pa.Table.from_pandas(ds, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"pm25_provenance": json.dumps(meta, default=str).encode()})
    pq.write_table(table, path)
    da.write_sidecar(meta, path)
    return path


def run_day1(cfg: dict | None = None) -> dict:
    cfg = cfg or da.load_config()
    df = da.load_city_day(cfg)
    da.verify_raw(df, cfg)
    ds = build_dataset(df, cfg)
    path = export_dataset(ds, cfg)
    usable = usable_samples_table(ds)
    miss = feature_missingness_table(ds, cfg)
    da.write_table(usable, "usable_samples_by_city_split_horizon", cfg,
                   "src.features.usable_samples_table",
                   notes="Splits by target_date. n_eval_mask = common evaluation sample.")
    da.write_table(miss, "feature_missingness_by_city_horizon_split", cfg,
                   "src.features.feature_missingness_table")
    return {"dataset": ds, "path": path, "usable": usable, "missingness": miss}


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_rows", 200)
    res = run_day1()
    ds = res["dataset"]
    print(f"Wrote {res['path']} : {len(ds):,} rows x {ds.shape[1]} cols")
    u = res["usable"]
    m = u[u["scheme"] == "main"].pivot_table(index=["city_role", "city"], columns=["horizon", "split"],
                                             values="n_eval_mask", observed=True, aggfunc="sum")
    print("\nMain scheme — eval-mask samples by city x horizon x split:\n", m.to_string())
