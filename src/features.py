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
      python -m src.features rq1  (RQ1 A/B same-day estimation table; separate parquet)

Named model-input sets live in config `feature_sets` (resolved by `feature_set`):
same_day (RQ1 A/B only), causal_common (C, RQ2 primary, RQ3; no city ID),
causal_city_aware (pooled-only ablation; unseen cities encoded as NaN by `encode_city`).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import data_audit as da
from src.preprocessing import city_daily_series, ffill_inputs, select_cities
from src.splits import assign_rq1_splits, assign_splits


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
        target_date = pd.DatetimeIndex(X.index) + pd.Timedelta(days=h)
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
    role = pd.Categorical(ds["city"].astype(str).map(roles),
                          categories=["core", "transfer", "transfer_optional"])
    ds.insert(1, "city_role", pd.Series(role, index=ds.index))
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


# ------------------------------------------------------------------ named feature sets
def feature_set(cfg: dict, name: str) -> list[str]:
    """Resolve a named model-input set from config `feature_sets` (supports `extends`)."""
    spec = cfg["feature_sets"][name]
    if isinstance(spec, dict):
        cols = feature_set(cfg, spec["extends"]) + list(spec["add"])
    else:
        cols = list(spec)
    if len(set(cols)) != len(cols):
        raise ValueError(f"duplicate columns in feature set {name!r}")
    return cols


def encode_city(df: pd.DataFrame, train_cities: list[str]) -> pd.Series:
    """City as categorical restricted to the TRAINING cities; any other city -> NaN (unseen).

    Used only with the city-aware ablation, so a model can never look up a code for a
    city it did not train on (e.g. transfer cities evaluated with the pooled model).
    """
    city = df["city"].astype(str)
    return pd.Series(pd.Categorical(city.where(city.isin(train_cities)),
                                    categories=sorted(train_cities)),
                     index=df.index, name="city")


# ------------------------------------------------------------------ RQ1 same-day table
def build_same_day_dataset(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """RQ1 protocols A and B: same-day estimation, exactly ONE row per (city, target_date).

    y_target = PM2.5 on day d (observed only); inputs = same-day AQI + co-pollutants on d
    (unfilled) + calendar features of d. No PM2.5 history, no AQI_Bucket.
    """

    tgt = cfg["data"]["target"]
    cities = list(cfg["cities"][cfg["rq1"]["cities"]])
    same_day_inputs = [c for c in feature_set(cfg, "same_day") if c not in cfg["features"]["calendar"]]
    sub = df.loc[df["City"].isin(cities) & df[tgt].notna(), ["City", "Date", tgt, *same_day_inputs]]
    sd = sub.rename(columns={"City": "city", "Date": "target_date", tgt: "y_target"})
    sd = sd.sort_values(["city", "target_date"], ignore_index=True)
    if sd.duplicated(["city", "target_date"]).any():
        raise AssertionError("same-day dataset must have one row per (city, target_date)")
    sd = pd.concat([sd, calendar_features(sd["target_date"])], axis=1)
    sd["city"] = pd.Categorical(sd["city"], categories=sorted(cities))
    sd = assign_rq1_splits(sd, cfg)
    cols = ["city", "target_date", *feature_set(cfg, "same_day"), "y_target", "split_A", "split_B"]
    return sd[cols]


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
def write_parquet_with_provenance(df: pd.DataFrame, path, meta: dict):
    """Parquet with provenance in schema metadata (key b'pm25_provenance') + .meta.json sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"pm25_provenance": json.dumps(meta, default=str).encode()})
    pq.write_table(table, path)
    da.write_sidecar(meta, path)
    return path


def export_dataset(ds: pd.DataFrame, cfg: dict, path=None):
    """Write the processed forecasting parquet (protocol C / RQ2 / RQ3)."""
    path = da.ROOT / cfg["paths"]["processed_dataset"] if path is None else path
    meta = da.build_metadata(ds, path.name, cfg, "src.features.build_dataset",
                             notes="Strict (protocol C) causal features; splits by target_date.")
    meta.update(strict_features=strict_feature_columns(cfg),
                features_config=cfg["features"], preprocessing_config=cfg["preprocessing"],
                splits_config=cfg["splits"], cities=cfg["cities"])
    return write_parquet_with_provenance(ds, path, meta)


def export_same_day_dataset(sd: pd.DataFrame, cfg: dict, path=None):
    """Write the RQ1 same-day estimation parquet (protocols A and B only)."""
    path = da.ROOT / cfg["rq1"]["same_day_dataset"] if path is None else path
    meta = da.build_metadata(sd, path.name, cfg, "src.features.build_same_day_dataset",
                             notes="RQ1 A/B same-day estimation; one row per (city, target_date). "
                                   "Contains same-day AQI/co-pollutants: NEVER use for protocol C.")
    meta.update(same_day_features=feature_set(cfg, "same_day"), rq1_config=cfg["rq1"],
                splits_config=cfg["splits"])
    return write_parquet_with_provenance(sd, path, meta)


def rq1_split_table(sd: pd.DataFrame) -> pd.DataFrame:
    """Row counts per city x protocol x split label for the same-day dataset, plus each
    split's share of the common A/B pool (train+test). A's test share is ~20% (rounded per
    city); B's test share is the full 2019 year (~25-27%), so the proportions differ."""
    parts = []
    for p in ("A", "B"):
        t = sd.groupby(["city", f"split_{p}"], observed=True).size().rename("n_rows").reset_index()
        t = t.rename(columns={f"split_{p}": "split"}).assign(protocol=p)
        t["split"] = t["split"].astype(str)
        pool = t[t["split"].isin(["train", "test"])].groupby("city", observed=True)["n_rows"].sum()
        in_pool = t["split"].isin(["train", "test"])
        t["share_of_pool"] = np.where(in_pool, (t["n_rows"] / t["city"].map(pool).astype(float)).round(4), np.nan)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)[["protocol", "city", "split", "n_rows", "share_of_pool"]]


def run_rq1_dataset(cfg: dict | None = None) -> dict:
    """Build + export the RQ1 same-day dataset (no model training)."""
    cfg = cfg or da.load_config()
    df = da.load_city_day(cfg)
    da.verify_raw(df, cfg)
    sd = build_same_day_dataset(df, cfg)
    path = export_same_day_dataset(sd, cfg)
    counts = rq1_split_table(sd)
    da.write_table(counts, "rq1_same_day_split_counts", cfg, "src.features.rq1_split_table",
                   notes="A: random 80/20 over unique city-dates (2015-17 + 2019), stratified by city. "
                         "B: chronological. 2018 reserved for calibration.")
    return {"dataset": sd, "path": path, "counts": counts}


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


def _main_rq1() -> None:
    res = run_rq1_dataset()
    sd = res["dataset"]
    print(f"Wrote {res['path']} : {len(sd):,} rows x {sd.shape[1]} cols "
          f"(unique city-dates: {not sd.duplicated(['city', 'target_date']).any()})")
    print(res["counts"].pivot_table(index="city", columns=["protocol", "split"], values="n_rows",
                                    observed=True).to_string())


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 200, "display.max_rows", 200)
    if sys.argv[1:] == ["rq1"]:
        _main_rq1()
        sys.exit(0)
    res = run_day1()
    ds = res["dataset"]
    print(f"Wrote {res['path']} : {len(ds):,} rows x {ds.shape[1]} cols")
    u = res["usable"]
    m = u[u["scheme"] == "main"].pivot_table(index=["city_role", "city"], columns=["horizon", "split"],
                                             values="n_eval_mask", observed=True, aggfunc="sum")
    print("\nMain scheme — eval-mask samples by city x horizon x split:\n", m.to_string())
