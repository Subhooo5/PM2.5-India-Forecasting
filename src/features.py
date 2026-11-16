from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import data_audit as da
from src.preprocessing import ROLES, city_daily_series, ffill_inputs, select_cities
from src.splits import assign_same_day_splits, assign_splits

SAME_DAY_SET = "same_day_estimation"


def lag_columns(cfg: dict) -> list[str]:
    return [f"pm25_lag{k}" for k in cfg["features"]["lags"]]


def flag_columns(cfg: dict) -> list[str]:
    return [f"{c}_imputed" for c in lag_columns(cfg)]


def rolling_columns(cfg: dict) -> list[str]:
    return [f"pm25_{stat}{w}" for w in cfg["features"]["rolling_windows"] for stat in ("rmean", "rstd")]


def causal_columns(cfg: dict) -> list[str]:
    return ["pm25_origin", *lag_columns(cfg), *flag_columns(cfg), *rolling_columns(cfg),
            *cfg["features"]["calendar"]]


def origin_features(s_raw: pd.Series, cfg: dict) -> pd.DataFrame:
    if not s_raw.index.equals(pd.date_range(s_raw.index[0], s_raw.index[-1], freq="D")):
        raise ValueError("series must be on a complete daily calendar")
    f = cfg["features"]
    filled = ffill_inputs(s_raw, cfg["preprocessing"]["ffill_limit"])
    out = pd.DataFrame(index=s_raw.index)
    out["pm25_origin"] = s_raw
    for k in f["lags"]:
        out[f"pm25_lag{k}"] = filled.shift(k)
    for k in f["lags"]:
        out[f"pm25_lag{k}_imputed"] = (s_raw.shift(k).isna() & filled.shift(k).notna()).astype("int8")
    for w in f["rolling_windows"]:
        r = s_raw.rolling(window=w, min_periods=int(f["rolling_min_obs"][w]))
        out[f"pm25_rmean{w}"] = r.mean()
        out[f"pm25_rstd{w}"] = r.std()
    return out


def calendar_features(target_date: pd.Series) -> pd.DataFrame:
    td = pd.DatetimeIndex(target_date)
    angle = 2 * np.pi * td.day_of_year.to_numpy() / 365.25
    return pd.DataFrame({"tgt_month": td.month.astype("int8"), "tgt_dow": td.day_of_week.astype("int8"),
                         "tgt_doy_sin": np.sin(angle), "tgt_doy_cos": np.cos(angle)}, index=target_date.index)


def build_city_samples(s_raw: pd.Series, city: str, cfg: dict) -> pd.DataFrame:
    period = cfg["baselines"]["seasonal_period"]
    X = origin_features(s_raw, cfg)
    X = X.loc[X["pm25_origin"].notna()]
    parts = []
    for h in cfg["features"]["horizons"]:
        if not 1 <= h <= period:
            raise ValueError(f"horizon {h}: seasonal naive y[t+h-{period}] would not be causal")
        target_date = pd.DatetimeIndex(X.index) + pd.Timedelta(days=h)
        part = X.reset_index(names="origin_date")
        part.insert(1, "target_date", target_date)
        part.insert(2, "horizon", np.int8(h))
        part["y_target"] = s_raw.reindex(target_date).to_numpy()
        part["snaive_ref"] = s_raw.reindex(target_date - pd.Timedelta(days=period)).to_numpy()
        parts.append(part.loc[part["y_target"].notna()])
    out = pd.concat(parts, ignore_index=True)
    out = pd.concat([out, calendar_features(out["target_date"])], axis=1)
    out.insert(0, "city", city)
    out["eval_mask"] = out["y_target"].notna() & out["pm25_origin"].notna() & out["snaive_ref"].notna()
    return out


def build_dataset(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    roles = select_cities(cfg)
    ds = pd.concat([build_city_samples(city_daily_series(df, c, cfg), c, cfg) for c in roles], ignore_index=True)
    ds["city"] = pd.Categorical(ds["city"], categories=sorted(roles))
    role = pd.Categorical(ds["city"].astype(str).map(roles), categories=list(ROLES))
    ds.insert(1, "city_role", pd.Series(role, index=ds.index))
    ds = assign_splits(ds, cfg)
    cols = ["city", "city_role", "origin_date", "target_date", "horizon", *causal_columns(cfg),
            "y_target", "snaive_ref", "eval_mask", "split_main", "split_stress", "stress_period"]
    ds = ds[cols].sort_values(["city", "horizon", "origin_date"], ignore_index=True)
    if set(cfg["features"]["forbidden"]) & set(ds.columns):
        raise AssertionError("forbidden columns in the forecasting dataset")
    return ds


def feature_set(cfg: dict, name: str) -> list[str]:
    spec = cfg["feature_sets"][name]
    cols = feature_set(cfg, spec["extends"]) + list(spec["add"]) if isinstance(spec, dict) else list(spec)
    if len(set(cols)) != len(cols):
        raise ValueError(f"duplicate columns in feature set {name!r}")
    return cols


def encode_city(df: pd.DataFrame, train_cities: list[str]) -> pd.Series:
    city = df["city"].astype(str)
    return pd.Series(pd.Categorical(city.where(city.isin(train_cities)), categories=sorted(train_cities)),
                     index=df.index, name="city")


def build_same_day_dataset(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    tgt = cfg["data"]["target"]
    cities = list(cfg["cities"][cfg["leakage"]["cities"]])
    inputs = [c for c in feature_set(cfg, SAME_DAY_SET) if c not in cfg["features"]["calendar"]]
    sd = (df.loc[df["City"].isin(cities) & df[tgt].notna(), ["City", "Date", tgt, *inputs]]
          .rename(columns={"City": "city", "Date": "target_date", tgt: "y_target"})
          .sort_values(["city", "target_date"], ignore_index=True))
    if sd.duplicated(["city", "target_date"]).any():
        raise AssertionError("same-day dataset must have one row per (city, target_date)")
    sd = pd.concat([sd, calendar_features(sd["target_date"])], axis=1)
    sd["city"] = pd.Categorical(sd["city"], categories=sorted(cities))
    sd = assign_same_day_splits(sd, cfg)
    return sd[["city", "target_date", *feature_set(cfg, SAME_DAY_SET), "y_target", "split_A", "split_B"]]


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
    g = ds.groupby(["city", "horizon", "split_main"], observed=True)
    nan_pct = g[lag_columns(cfg) + rolling_columns(cfg)].agg(lambda x: round(100 * x.isna().mean(), 2))
    imp_pct = g[flag_columns(cfg)].mean().mul(100).round(2).add_suffix("_pct")
    return pd.concat([g.size().rename("n_samples"), nan_pct.add_suffix("_nan_pct"), imp_pct], axis=1).reset_index()


def same_day_split_table(sd: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for p in ("A", "B"):
        t = sd.groupby(["city", f"split_{p}"], observed=True).size().rename("n_rows").reset_index()
        t = t.rename(columns={f"split_{p}": "split"}).assign(protocol=p)
        t["split"] = t["split"].astype(str)
        in_pool = t["split"].isin(["train", "test"])
        pool = t[in_pool].groupby("city", observed=True)["n_rows"].sum()
        t["share_of_pool"] = np.where(in_pool, (t["n_rows"] / t["city"].map(pool).astype(float)).round(4), np.nan)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)[["protocol", "city", "split", "n_rows", "share_of_pool"]]


def write_parquet(df: pd.DataFrame, path, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"pm25_provenance": json.dumps(meta, default=str).encode()})
    pq.write_table(table, path)
    da.write_sidecar(meta, path)
    return path


def run_preprocess(cfg: dict) -> dict:
    df = da.load_verified(cfg)
    ds, sd = build_dataset(df, cfg), build_same_day_dataset(df, cfg)
    for data, key, gen, notes, extra in (
            (ds, "forecast_dataset", "build_dataset", "Causal features; splits by target date.",
             {"features": causal_columns(cfg), "splits": cfg["splits"], "cities": cfg["cities"]}),
            (sd, "same_day_dataset", "build_same_day_dataset",
             "Same-day estimation table; one row per (city, target_date); never a forecasting input.",
             {"features": feature_set(cfg, SAME_DAY_SET), "leakage": cfg["leakage"]})):
        path = da.resolve(cfg, key)
        write_parquet(data, path, {**da.metadata(data, path.name, cfg, f"src.features.{gen}", notes), **extra})
    return {"dataset": ds, "same_day": sd}


def run_features(cfg: dict) -> dict:
    ds = pd.read_parquet(da.resolve(cfg, "forecast_dataset"))
    sd = pd.read_parquet(da.resolve(cfg, "same_day_dataset"))
    tables = {"usable_samples_by_city_split_horizon": usable_samples_table(ds),
              "feature_missingness_by_city_horizon_split": feature_missingness_table(ds, cfg),
              "leakage_same_day_split_counts": same_day_split_table(sd)}
    tables_dir = da.output_dirs(cfg)["tables"]
    for name, t in tables.items():
        da.write_csv(t, tables_dir / f"{name}.csv", cfg, f"src.features.{name}", "Splits by target date.")
    return tables
