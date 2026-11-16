from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import data_audit as da
from src.conformal import aci_stream, conformal_quantile, conformity_scores, cqr_interval
from src.metrics import interval_metrics
from src.models import check_feature_set, fit_lgbm, load_model, predict_lgbm, quantile_params, save_model
from src.splits import SPLIT_ORDER, LeakageError, main_ranges

KEYS = ["city", "origin_date", "horizon"]
ALIGN = ["city", "origin_date", "target_date", "horizon", "y_true"]
METHODS = {"raw_quantile": ("q05", "q95"), "cqr": ("cqr_lo", "cqr_hi"), "cqr_aci": ("aci_lo", "aci_hi")}
ROLLING_WINDOW = 30
MIN_N = 30


def require_coverage(df: pd.DataFrame, cfg: dict, cities, what: str) -> None:
    have = set(zip(df["city"].astype(str), df["horizon"].astype(int)))
    need = {(c, h) for c in cities for h in cfg["uncertainty"]["horizons"]}
    extra = set(df["city"].astype(str)) - set(cities)
    if need - have or extra:
        raise ValueError(f"{what}: {len(df)} rows; missing (city, horizon) {sorted(need - have)}; "
                         f"unexpected cities {sorted(extra)}")


def frames(cfg: dict) -> dict[str, pd.DataFrame]:
    u = cfg["uncertainty"]
    df = pd.read_parquet(da.resolve(cfg, "forecast_dataset"))
    df = df[df["city"].astype(str).isin(cfg["cities"][u["cities"]]) & df["horizon"].isin(u["horizons"])]
    lab = df[u["split_column"]]
    in_range = {s: df["target_date"].between(lo, hi) for s, (lo, hi) in zip(SPLIT_ORDER, main_ranges(cfg))}
    for bad, msg in (
            (df.duplicated(["city", "origin_date", "horizon"]), "duplicated (city, origin_date, horizon) keys"),
            (df["target_date"] != df["origin_date"] + pd.to_timedelta(df["horizon"].astype(int), unit="D"),
             "rows with target_date != origin_date + horizon days"),
            (~lab.isin([*SPLIT_ORDER, "outside"]), "split labels outside {train, calibration, test, outside}"),
            *[(in_range[s] & (lab != s), f"rows with a {s}-range target_date not labelled {s!r}") for s in SPLIT_ORDER],
            ((lab == "outside") & np.logical_or.reduce(list(in_range.values())),
             "rows labelled 'outside' with a target_date inside a split range")):
        if bad.any():
            raise ValueError(f"forecast sample: {int(bad.sum())} {msg}")
    out = {s: df[lab == s] for s in SPLIT_ORDER}
    for s, (lo, hi) in zip(SPLIT_ORDER, main_ranges(cfg)):
        if not out[s]["target_date"].between(lo, hi).all():
            raise LeakageError(f"{s} rows have target dates outside {lo.date()}..{hi.date()}")
        require_coverage(out[s], cfg, cfg["cities"][u["cities"]], f"{s} frame")
    return out


def quantile_models(cfg: dict, train: pd.DataFrame, models_dir: Path, stem: str, dataset_sha256: str,
                    train_range=None, **extra) -> dict:
    """Load each saved quantile model whose metadata matches the expected values exactly, else fit and save it."""
    u, models = cfg["uncertainty"], {}
    lo, hi = train_range or main_ranges(cfg)[0]
    for h in u["horizons"]:
        for q in u["quantiles"]:
            path, params = models_dir / f"{stem}_h{h}_q{round(q * 100):02d}.txt", quantile_params(cfg, q)
            if path.exists():
                model, saved = load_model(path)
                expected = {"uncertainty_config_hash": da.uncertainty_config_hash(cfg),
                            "config_hash": da.config_hash(cfg), "dataset_sha256": dataset_sha256,
                            "kaggle_version": cfg["data"]["kaggle_version"], "params": params,
                            "feature_set": u["feature_set"], "features": check_feature_set(cfg, u["feature_set"]),
                            "horizon": h, "train_date_range": [str(lo.date()), str(hi.date())],
                            "n_train": int((train["horizon"] == h).sum()),
                            "train_cities": sorted(train["city"].astype(str).unique()), "held_out_city": None, **extra}
                for k, v in expected.items():
                    if k not in saved and k != "held_out_city":
                        raise ValueError(f"{path}: required metadata key {k!r} is missing")
                    if saved.get(k) != v:
                        raise ValueError(f"{path}: metadata {k!r} = {saved.get(k)!r}, expected {v!r}")
            else:
                model, meta = fit_lgbm(train, u["feature_set"], h, params, (lo, hi), cfg)
                meta.update(quantile=q, uncertainty_config_hash=da.uncertainty_config_hash(cfg), **extra)
                save_model(model, meta, path)
            models[h, q] = model
    return models


def _rows(g: pd.DataFrame) -> pd.DataFrame:
    return g.assign(city=g["city"].astype(str), horizon=g["horizon"].astype(int), y_true=g["y_target"])[ALIGN]


def predict_intervals(models: dict, data: dict, cfg: dict) -> pd.DataFrame:
    u, parts, inputs = cfg["uncertainty"], [], []
    for s in ("calibration", "test"):
        for h in sorted(u["horizons"]):
            g = data[s][data[s]["horizon"] == h]
            out = {}
            for name, q in zip(("q05", "q95"), u["quantiles"]):
                p = predict_lgbm(models[h, q], g)
                if not p[ALIGN].astype({"city": str, "horizon": int}).equals(_rows(g)):
                    raise AssertionError(f"{s} h={h} q={q}: predictions not aligned with the input rows")
                out[name] = p["y_pred"]
            parts.append(_rows(g).assign(split=s)[["city", "origin_date", "target_date", "horizon", "split", "y_true"]]
                         .assign(**out, eval_mask=g["eval_mask"]))
            inputs.append(_rows(g).assign(split=s, eval_mask=g["eval_mask"]))
    pred, inputs = pd.concat(parts, ignore_index=True), pd.concat(inputs, ignore_index=True)
    if not pred[ALIGN + ["split", "eval_mask"]].equals(inputs[ALIGN + ["split", "eval_mask"]]):
        raise AssertionError("concatenated predictions not aligned with the input rows")
    for s in ("calibration", "test"):
        if int((pred["split"] == s).sum()) != len(data[s]):
            raise AssertionError(f"{s}: {int((pred['split'] == s).sum())} predictions for {len(data[s])} rows")
    return pred


def interval_checks(pred: pd.DataFrame, lo: str, hi: str, by=("split", "horizon")) -> list[dict]:
    b, w = pred[[lo, hi]], pred[hi] - pred[lo]
    d = pred.assign(width=w, crossed=pred[lo] > pred[hi], nan=b.isna().any(axis=1), inf=b.abs().eq(np.inf).any(axis=1),
                    width_le0=w <= 0, lower_lt0=pred[lo] < 0)
    out = d.groupby(list(by), sort=True).agg(
        n=("width", "size"), n_crossed=("crossed", "sum"), n_nan=("nan", "sum"), n_inf=("inf", "sum"),
        width_min=("width", "min"), width_median=("width", "median"), width_max=("width", "max"),
        n_width_le0=("width_le0", "sum"), n_lower_below_zero=("lower_lt0", "sum")).reset_index()
    return json.loads(out.to_json(orient="records"))


def calibration_scores(pred: pd.DataFrame, cfg: dict, cities, cal_range=None) -> pd.DataFrame:
    cal = pred[(pred["split"] == "calibration") & pred["eval_mask"]]
    lo, hi = cal_range or main_ranges(cfg)[1]
    if not cal["target_date"].between(lo, hi).all():
        raise LeakageError("CQR calibration rows outside the calibration year")
    require_coverage(cal, cfg, cities, "calibration eval_mask rows")
    return cal[KEYS + ["target_date"]].assign(score=conformity_scores(cal["y_true"], cal["q05"], cal["q95"]))


def q_hat_by_horizon(scores: pd.DataFrame, cfg: dict) -> dict:
    out, alpha = {}, cfg["uncertainty"]["alpha"]
    for h in cfg["uncertainty"]["horizons"]:
        s = scores.loc[scores["horizon"] == h, "score"]
        if s.empty:
            raise ValueError(f"h={h}: no calibration scores")
        q = conformal_quantile(s, alpha)
        if not np.isfinite(q):
            raise ValueError(f"h={h}: q_hat not finite (n_cal={len(s)} too small for alpha={alpha})")
        out[int(h)] = {"q_hat": q, "n_cal": len(s)}
    return out


def mark_calibration_ready(pred: pd.DataFrame, cal_end) -> pd.DataFrame:
    """A forecast is calibration-ready when it is issued on or after the last calibration day."""
    out = pred.copy()
    out.insert(list(out.columns).index("eval_mask") + 1, "calibration_ready", out["origin_date"] >= pd.Timestamp(cal_end))
    return out


def with_cqr(pred: pd.DataFrame, q_hat: dict, name: str) -> pd.DataFrame:
    lo, hi = cqr_interval(pred["q05"], pred["q95"], pred["horizon"].map(lambda h: q_hat[h]["q_hat"]))
    return pred.assign(**{f"{name}_lo": lo, f"{name}_hi": hi})


def aci_columns(pred: pd.DataFrame, scores: pd.DataFrame, cfg: dict, gamma: float) -> pd.DataFrame:
    u, parts = cfg["uncertainty"], []
    test = pred[(pred["split"] == "test") & pred["calibration_ready"]].sort_values("origin_date", kind="stable")
    for (_, h), g in test.groupby(["city", "horizon"], sort=True):
        out = aci_stream(g["origin_date"], g["target_date"], g["y_true"], g["q05"], g["q95"],
                         scores.loc[scores["horizon"] == h, "score"], u["alpha"], gamma, u["aci"]["alpha_bounds"])
        parts.append(pd.DataFrame({("aci_alpha" if k == "alpha_issue" else f"aci_{k}"): v for k, v in out.items()},
                                  index=g.index))
    return pd.concat(parts).reindex(pred.index)


def scored(pred: pd.DataFrame) -> pd.DataFrame:
    return pred[(pred["split"] == "test") & pred["eval_mask"] & pred["calibration_ready"]]


def grouped_metrics(pred: pd.DataFrame, methods: dict, keys: list, cfg: dict) -> pd.DataFrame:
    alpha, rows, test = cfg["uncertainty"]["alpha"], [], scored(pred)
    for method, (lo, hi) in methods.items():
        for k, g in (test.groupby(keys, sort=True) if keys else [((), test)]):
            rows.append({"method": method, **dict(zip(keys, k)),
                         **interval_metrics(g["y_true"], g[lo], g[hi], alpha), "nominal": 1 - alpha})
    t = pd.DataFrame(rows)
    if not (t.groupby(keys)["n"].nunique().eq(1).all() if keys else t["n"].nunique() == 1):
        raise AssertionError("methods are not scored on identical rows")
    t["coverage_gap"], t["low_n"] = t["picp"] - t["nominal"], t["n"] < MIN_N
    return t[["method", *keys, "n", "nominal", "picp", "coverage_gap", "mpiw", "winkler", "low_n"]]


def rolling_coverage(pred: pd.DataFrame) -> pd.DataFrame:
    m = scored(pred)
    m = m[m["aci_update_origin"].notna()].sort_values("target_date", kind="stable")
    return m.assign(rolling_coverage=m.groupby(["city", "horizon"])["aci_miss"].transform(
        lambda s: 1 - s.rolling(ROLLING_WINDOW).mean()))


def not_ready(pred: pd.DataFrame, horizons) -> dict:
    t = pred[(pred["split"] == "test") & ~pred["calibration_ready"]]
    return {int(h): int((t["horizon"] == h).sum()) for h in horizons}


def aci_summary(pred: pd.DataFrame, cfg: dict) -> dict:
    b_lo, b_hi = cfg["uncertainty"]["aci"]["alpha_bounds"]
    test, roll, out = scored(pred), rolling_coverage(pred), {}
    for h in cfg["uncertainty"]["horizons"]:
        a, rc = test.loc[test["horizon"] == h, "aci_alpha"], roll.loc[roll["horizon"] == h, "rolling_coverage"]
        out[int(h)] = {"mean_alpha": a.mean(), "min_alpha": a.min(), "max_alpha": a.max(),
                       "share_alpha_at_lower_bound": float(np.isclose(a, b_lo).mean()),
                       "share_alpha_at_upper_bound": float(np.isclose(a, b_hi).mean()),
                       "share_pending_at_end": test.loc[test["horizon"] == h, "aci_update_origin"].isna().mean(),
                       "rolling_coverage_min": rc.min(), "rolling_coverage_max": rc.max()}
    return out


def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["origin_date", "target_date", "aci_update_origin"],
                       keep_default_na=False, na_values=[""])
