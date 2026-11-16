from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import data_audit as da
from src import figures
from src import intervals as iv
from src.experiments import POINT_METHODS
from src.metrics import interval_metrics, point_metrics, relative_change_pct
from src.splits import split_ranges
from src.stress_forecasts import FILES, MIN_N, PERIODS, lockdown, run_point, run_uncertainty

MANIFEST_FILE = "result_manifest.csv"
LOCK_LABELS = ("pre-lockdown", "lockdown")


def pm25_distribution(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    r, lock, f = split_ranges(cfg, "stress"), lockdown(cfg), "%m-%d"
    md, year = raw["Date"].dt.strftime(f), raw["Date"].dt.year
    lo, hi = r["test"]
    period = np.select([(md >= lo.strftime(f)) & (md < lock.strftime(f)), (md >= lock.strftime(f)) & (md <= hi.strftime(f))],
                       ["pre_lockdown", "lockdown"], "")
    cal_year, (tr_lo, tr_hi) = r["calibration"][0].year, r["train"]
    window = np.select([year == lo.year, year == cal_year, year.between(tr_lo.year, tr_hi.year)],
                       [str(lo.year), f"{cal_year}_same_dates", f"{tr_lo.year}_{tr_hi.year}_same_dates"], "")
    d = raw.assign(period=period, window=window)
    d = d[d["City"].isin(cfg["cities"][cfg["stress_test"]["cities"]]) & (d["period"] != "") & (d["window"] != "")]
    g = d.dropna(subset=[cfg["data"]["target"]]).groupby(["City", "period", "window"])[cfg["data"]["target"]]
    t = g.agg(n_obs="count", mean="mean", median="median", sd="std", p90=lambda v: v.quantile(0.90),
              p95=lambda v: v.quantile(0.95), p99=lambda v: v.quantile(0.99), max="max").reset_index()
    return t.rename(columns={"City": "city"}).assign(low_n=lambda x: x["n_obs"] < MIN_N)


def _period_values(g: pd.DataFrame, family: str, cols, alpha: float) -> tuple[dict, int]:
    y, mean = g["y_true"], g["y_true"].mean()
    if family == "point":
        m = point_metrics(y, g["y_pred"])
        return {"mae": m["mae"], "rmse": m["rmse"], "mae_over_mean_pm25": m["mae"] / mean}, m["n"]
    lo, hi = g[cols[0]], g[cols[1]]
    m = interval_metrics(y, lo, hi, alpha)
    return {"picp": m["picp"], "mpiw": m["mpiw"], "winkler": m["winkler"], "mpiw_over_mean_pm25": m["mpiw"] / mean,
            "share_below": float((y < lo).mean()), "share_above": float((y > hi).mean())}, m["n"]


def period_shift(point: pd.DataFrame, unc_pred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows, keys = [], ["scope", "city", "horizon", "family", "method", "metric"]
    for family, d, methods in (("point", point[point["eval_mask"]], {m: None for m in POINT_METHODS}),
                               ("interval", iv.scored(unc_pred), iv.METHODS)):
        for method, cols in methods.items():
            dm = d[d["method"] == method] if family == "point" else d
            for scope, by in (("overall", ["horizon"]), ("city", ["city", "horizon"])):
                for k, g in dm.groupby(by + ["stress_period"], sort=True):
                    rec = {"scope": scope, "city": "all", **dict(zip(by + ["stress_period"], k)), "family": family,
                           "method": method}
                    vals, n = _period_values(g, family, cols, cfg["uncertainty"]["alpha"])
                    rows += [{**rec, "metric": mt, "value": v, "n": n} for mt, v in vals.items()]
    long = pd.DataFrame(rows)
    if (long.groupby(["scope", "city", "horizon", "family", "stress_period"])["n"].nunique() != 1).any():
        raise AssertionError("methods are not scored on identical rows")
    pre, lock = (long[long["stress_period"] == p].set_index(keys)[["value", "n"]] for p in PERIODS[1:])
    t = pre.merge(lock, how="outer", left_index=True, right_index=True, suffixes=("_pre", "_lock")).reset_index()
    return t.assign(pre_lockdown=t["value_pre"], lockdown=t["value_lock"], abs_change=t["value_lock"] - t["value_pre"],
                    rel_change_pct=relative_change_pct(t["value_lock"].to_numpy(), t["value_pre"].to_numpy()),
                    n_lockdown=t["n_lock"])[keys + ["pre_lockdown", "lockdown", "abs_change", "rel_change_pct",
                                                    "n_pre", "n_lockdown"]]


def aci_alpha_summary(unc_pred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    b_lo, b_hi = cfg["uncertainty"]["aci"]["alpha_bounds"]
    t = unc_pred[(unc_pred["split"] == "test") & unc_pred["calibration_ready"]].sort_values("origin_date", kind="stable")
    rows = []
    for (c, h), g in t.groupby(["city", "horizon"], sort=True):
        rec = {"city": c, "horizon": h, "alpha_1": g["aci_alpha"].iloc[0],
               "alpha_entering_lockdown": g.loc[g["stress_period"] == "lockdown", "aci_alpha"].iloc[0]}
        for p in PERIODS[1:]:
            a, ev = g.loc[g["stress_period"] == p, "aci_alpha"], g[(g["stress_period"] == p) & g["eval_mask"]]
            rec |= {f"mean_alpha_{p}": a.mean(), f"min_alpha_{p}": a.min(), f"max_alpha_{p}": a.max(),
                    f"miss_rate_{p}": ev["aci_miss"].mean(), f"n_{p}": len(ev)}
        rows.append(rec | {"n_matured": int(g["aci_update_origin"].notna().sum()),
                           "n_pending_at_end": int(g["aci_update_origin"].isna().sum()),
                           "share_at_lower_bound": float(np.isclose(g["aci_alpha"], b_lo).mean()),
                           "share_at_upper_bound": float(np.isclose(g["aci_alpha"], b_hi).mean())})
    return pd.DataFrame(rows)


def check_against_saved(shift: pd.DataFrame, pres: pd.DataFrame, ures: pd.DataFrame) -> None:
    for fam, saved, metrics, scope_map in (
            ("point", pres, ["mae", "rmse"], {"overall": "overall", "city": "city"}),
            ("interval", ures, ["picp", "mpiw", "winkler"], {"overall": "horizon", "city": "city_horizon"})):
        for _, r in shift[(shift["family"] == fam) & shift["metric"].isin(metrics)].iterrows():
            for p, n in (("pre_lockdown", "n_pre"), ("lockdown", "n_lockdown")):
                s = saved[(saved["period"] == p) & (saved["scope"] == scope_map[r["scope"]]) & (saved["city"] == r["city"])
                          & (saved["horizon"].astype(str) == str(r["horizon"])) & (saved["method"] == r["method"])]
                if len(s) != 1 or s[r["metric"]].iloc[0] != r[p] or s["n"].iloc[0] != r[n]:
                    raise AssertionError(f"period shift disagrees with the saved table: {tuple(r[:6])} {p}")


def load_saved(dirs: dict) -> dict:
    read = lambda p, dates: pd.read_csv(p, parse_dates=dates, keep_default_na=False, na_values=[""])
    return {"point": read(dirs["predictions"] / FILES["point_pred"], ["origin_date", "target_date"]),
            "unc": iv.load_predictions(dirs["predictions"] / FILES["unc_pred"]),
            "pres": read(dirs["tables"] / FILES["point_results"], []),
            "ures": read(dirs["tables"] / FILES["unc_results"], [])}


def run_diagnostics(cfg: dict, dirs: dict | None = None) -> dict:
    dirs = dirs or da.output_dirs(cfg)
    s = load_saved(dirs)
    t = {"distribution": pm25_distribution(da.load_verified(cfg), cfg), "shift": period_shift(s["point"], s["unc"], cfg),
         "alpha": aci_alpha_summary(s["unc"], cfg)}
    check_against_saved(t["shift"], s["pres"], s["ures"])
    for name, df in t.items():
        da.write_csv(df, dirs["tables"] / FILES[name], cfg, "src.stress.run_diagnostics",
                     "Read-only diagnostics from saved predictions; pre/lockdown also differ in season.",
                     stress_config_hash=da.stress_config_hash(cfg))
    return t


def run(cfg: dict) -> dict:
    return {"point": run_point(cfg), "uncertainty": run_uncertainty(cfg), "diagnostics": run_diagnostics(cfg)}


def previous_year_same_dates(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.to_datetime([f"{d.year - 1}-{d:%m-%d}" for d in days], errors="coerce")


def _n_range(n: pd.Series) -> str:
    return f"{n.min()}" if n.min() == n.max() else f"{n.min()}-{n.max()}"


def _bars(t: pd.DataFrame, methods, col: str, hs: list) -> dict:
    out = {}
    for p in PERIODS[1:]:
        d = t[(t["period"] == p) & (t["city"] == "all") & t["horizon"].astype(str).isin([str(h) for h in hs])]
        by = {m: d[d["method"] == m].assign(h=d["horizon"].astype(int)).set_index("h").loc[hs] for m in methods}
        out[p] = {"x": [f"h = {h}\nn = {k:,}" for h, k in zip(hs, next(iter(by.values()))["n"])],
                  "series": {m: v[col].tolist() for m, v in by.items()}}
    return out


def make_figures(cfg: dict, dirs: dict | None = None) -> list[Path]:
    dirs = dirs or da.output_dirs(cfg)
    s, raw, f = load_saved(dirs), da.load_verified(cfg), dirs["figures"]
    st, lock = cfg["stress_test"], lockdown(cfg)
    core, hs, nominal = cfg["cities"][st["cities"]], st["horizons"], 1 - cfg["uncertainty"]["alpha"]
    days = pd.date_range(*split_ranges(cfg, "stress")["test"])
    obs = raw.set_index(["City", "Date"])[cfg["data"]["target"]]
    get = lambda c, d: obs.reindex(pd.MultiIndex.from_product([[c], d])).to_numpy()
    rows = {c: f"{c} (n = {int(np.isfinite(get(c, days)).sum())})" for c in core}
    test = s["unc"][(s["unc"]["split"] == "test") & s["unc"]["calibration_ready"]].sort_values("origin_date", kind="stable")
    alpha = {k: (g["origin_date"], g["aci_alpha"]) for k, g in test.groupby(["city", "horizon"])}
    cu = s["ures"][s["ures"]["scope"] == "city_horizon"].astype({"horizon": int})
    gap = {f"{lab}, {p} (n = {_n_range(cu.loc[cu['period'] == p, 'n'])} per cell)":
           cu[(cu["method"] == m) & (cu["period"] == p)].pivot(index="city", columns="horizon", values="coverage_gap")
           .loc[core, hs].to_numpy() for lab, m in (("CQR", "cqr"), ("CQR+ACI", "cqr_aci")) for p in PERIODS[1:]}
    note = "2020 stress test (distribution shift, not causal; partial year; periods by target date)"
    return [
        figures.stream_grid({(rows[c], ""): (days, get(c, days)) for c in core}, list(rows.values()), [""],
                            f / "stress_2020_pm25_timeseries.png", "observed daily PM2.5 2020 (µg/m³)",
                            f"Observed daily PM2.5, 2020-01-01 to 2020-07-01: {note}; gaps = missing days", {},
                            series2={(rows[c], ""): (days, get(c, previous_year_same_dates(days))) for c in core},
                            label2="2019 same dates (seasonal reference; no 2019 value for Feb 29)",
                            vline=(lock, LOCK_LABELS), col_fmt="{}", xlabel="target date", sharey=False),
        figures.bars(_bars(s["pres"], POINT_METHODS, "mae", hs), f / "stress_2020_point_errors.png", "MAE (µg/m³)",
                     f"{note}: MAE by horizon; seasonal naive = persistence at h = 7 by definition"),
        figures.bars(_bars(s["ures"], iv.METHODS, "picp", hs), f / "stress_2020_coverage_comparison.png", "PICP",
                     f"{note}: coverage (marginal)", nominal),
        figures.bars(_bars(s["ures"], iv.METHODS, "mpiw", hs), f / "stress_2020_interval_width_comparison.png",
                     "MPIW (µg/m³)", f"{note}: mean interval width"),
        figures.heatmaps(gap, core, hs, f / "stress_2020_city_horizon_coverage.png",
                         f"{note}: PICP − 0.90 by city and horizon", "PICP − 0.90", ncols=2),
        figures.stream_grid(alpha, core, hs, f / "stress_2020_aci_alpha_evolution.png", "alpha_t at issue",
                            f"{note}\nCQR+ACI alpha_t (gamma {cfg['uncertainty']['aci']['gamma']}), calibration-ready "
                            f"forecasts, n = {_n_range(test.groupby(['city', 'horizon']).size())} per city x horizon; "
                            "marker = first origin with a lockdown target (2020-03-25 minus h)",
                            {b: f"bound {b:.2f}" for b in cfg["uncertainty"]["aci"]["alpha_bounds"]},
                            vline=({h: lock - pd.Timedelta(days=h) for h in hs}, LOCK_LABELS),
                            xlabel="2020 (forecast origin)")]


def write_manifest(cfg: dict, base=None) -> pd.DataFrame:
    base = Path(base or da.ROOT)
    out = base / "outputs" / MANIFEST_FILE
    skip = {out, out.with_name(out.stem + ".meta.json")}
    files = sorted(f for d in ("outputs", "data/processed", "configs") for f in (base / d).rglob("*")
                   if f.is_file() and f.name != ".DS_Store" and f not in skip and not f.name.endswith("test_report.txt"))
    m = pd.DataFrame([{"path": f.relative_to(base).as_posix(), "bytes": f.stat().st_size, "sha256": da.sha256_file(f)}
                      for f in files])
    da.write_csv(m, out, cfg, "src.stress.write_manifest",
                 "Snapshot of the files after a full run, not a freeze; sidecars carry timestamps.")
    return m
