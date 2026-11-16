from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import data_audit as da
from src import figures
from src.intervals import (METHODS, ROLLING_WINDOW, aci_columns, aci_summary, calibration_scores, frames,
                           grouped_metrics, interval_checks, load_predictions, mark_calibration_ready, not_ready,
                           predict_intervals, q_hat_by_horizon, quantile_models, rolling_coverage, scored, with_cqr)
from src.metrics import interval_metrics
from src.splits import LeakageError, main_ranges

MODEL_STEM = "uncertainty/quantile"
LOCO_STEM = "uncertainty/loco/quantile_loco_{city}"
PRED_FILE = "uncertainty_predictions.csv"
HELD_OUT_PRED_FILE = "uncertainty_held_out_predictions.csv"
TABLE_FILES = {"overall": "uncertainty_overall.csv", "horizon": "uncertainty_by_horizon.csv",
               "city": "uncertainty_by_city.csv", "season": "uncertainty_by_season.csv",
               "regime": "uncertainty_by_pollution_regime.csv", "held_out": "uncertainty_held_out_city.csv",
               "gamma": "uncertainty_aci_gamma_sensitivity.csv", "raw_and_cqr": "uncertainty_raw_and_cqr.csv"}
LOCO_METHODS = {"raw_loco": ("q05", "q95"), "cqr_source": ("cqr_source_lo", "cqr_source_hi"),
                "aci_source": ("aci_lo", "aci_hi")}
LOCAL_METHODS = {"cqr_local": ("cqr_local_lo", "cqr_local_hi")}
HELD_OUT_COLUMNS = ["city", "origin_date", "target_date", "horizon", "split", "y_true", "eval_mask",
                    "calibration_ready", "q05", "q95", "cqr_source_lo", "cqr_source_hi", "cqr_local_lo",
                    "cqr_local_hi", "aci_alpha", "aci_lo", "aci_hi", "aci_miss", "aci_update_origin", "aci_alpha_after"]
SENS_LABEL = "sensitivity only - not used for selection"
NOTES = "Marginal results on 2019 test rows that are eval_mask and calibration_ready; see docs/design.md."
HELD_OUT_NOTE = "Held-out city rows (calibration rows included for transparency, never scored); see docs/design.md."


def regime_thresholds(raw: pd.DataFrame, cfg: dict) -> dict:
    lo, hi = main_ranges(cfg)[0]
    r = raw[raw["City"].isin(cfg["cities"][cfg["uncertainty"]["cities"]]) & raw["Date"].between(lo, hi)]
    q = 1 - cfg["uncertainty"]["regime"]["top_fraction"]
    return {str(c): float(v) for c, v in r.groupby("City")[cfg["data"]["target"]].quantile(q).items()}


def with_strata(pred: pd.DataFrame, thresholds: dict, cfg: dict) -> pd.DataFrame:
    season = {m: s for s, months in cfg["uncertainty"]["seasons"].items() for m in months}
    if set(thresholds) != set(cfg["cities"][cfg["uncertainty"]["cities"]]):
        raise ValueError(f"regime thresholds for {sorted(thresholds)}, expected the core cities")
    out = pred.assign(season=pred["target_date"].dt.month.map(season),
                      regime=np.where(pred["y_true"] > pred["city"].map(thresholds), "top10", "normal"))
    if out["season"].isna().any() or not set(out["regime"]) <= {"top10", "normal"}:
        raise ValueError("a target month has no season or a regime label is invalid")
    return out


def raw_and_cqr_table(pred: pd.DataFrame, q_hat: dict, cfg: dict) -> pd.DataFrame:
    alpha, rows, ev = cfg["uncertainty"]["alpha"], [], scored(pred)
    for method in ("raw_quantile", "cqr"):
        lo, hi = METHODS[method]
        for scope, keys in (("overall", []), ("horizon", ["horizon"]), ("city_horizon", ["city", "horizon"])):
            for k, g in (ev.groupby(keys, sort=True) if keys else [((), ev)]):
                rec = dict(zip(keys, k))
                h, cqr = rec.get("horizon"), method == "cqr"
                rows.append({"method": method, "scope": scope, "city": rec.get("city", "all"), "horizon": h,
                             **interval_metrics(g["y_true"], g[lo], g[hi], alpha), "nominal": 1 - alpha,
                             "n_crossed": int((g[lo] > g[hi]).sum()), "n_lower_below_zero": int((g[lo] < 0).sum()),
                             "q_hat": q_hat[h]["q_hat"] if cqr and h is not None else np.nan,
                             "n_cal": (q_hat[h]["n_cal"] if h is not None else sum(v["n_cal"] for v in q_hat.values()))
                             if cqr else np.nan})
    return pd.DataFrame(rows).astype({"horizon": "Int64", "n_cal": "Int64"})[
        ["method", "scope", "city", "horizon", "picp", "mpiw", "winkler", "n", "nominal", "n_crossed",
         "n_lower_below_zero", "q_hat", "n_cal"]]


def held_out_city(cfg: dict, data: dict, city: str, models_dir: Path, dataset_sha256: str) -> dict:
    core = cfg["cities"][cfg["uncertainty"]["cities"]]
    train = data["train"][data["train"]["city"].astype(str) != city]
    if set(train["city"].astype(str)) != set(core) - {city}:
        raise LeakageError(f"LOCO training cities for {city} are not core minus {city}")
    models = quantile_models(cfg, train, models_dir, LOCO_STEM.format(city=city), dataset_sha256, held_out_city=city)
    pred = mark_calibration_ready(predict_intervals(models, data, cfg), main_ranges(cfg)[1][1])
    own = pred["city"] == city
    source = calibration_scores(pred[~own], cfg, sorted(set(core) - {city}))
    local = calibration_scores(pred[own], cfg, [city])
    q_hat = {"source": q_hat_by_horizon(source, cfg), "local": q_hat_by_horizon(local, cfg)}
    p = with_cqr(with_cqr(pred[own], q_hat["source"], "cqr_source"), q_hat["local"], "cqr_local")
    p = p.join(aci_columns(p, source, cfg, cfg["uncertainty"]["aci"]["gamma"]))
    return {"pred": p, "train": train, "source": source, "local": local, "q_hat": q_hat, "models": models}


def check_config(cfg: dict) -> None:
    u = cfg["uncertainty"]
    if u["aci"]["alpha_init"] != u["alpha"] or u["quantiles"] != cfg["lightgbm"]["quantile_alphas"]:
        raise ValueError("uncertainty.aci.alpha_init must equal uncertainty.alpha and uncertainty.quantiles "
                         "must equal lightgbm.quantile_alphas")


def tables(pred: pd.DataFrame, st: pd.DataFrame, aci: dict, ho: dict, q_hat: dict, cfg: dict) -> dict:
    u = cfg["uncertainty"]
    t = {name: grouped_metrics(pred, METHODS, keys, cfg)
         for name, keys in (("overall", []), ("horizon", ["horizon"]), ("city", ["city", "horizon"]))}
    t["season"] = grouped_metrics(st, METHODS, ["season", "horizon"], cfg)
    regime = pd.concat([grouped_metrics(st.assign(city="all"), METHODS, ["city", "regime", "horizon"], cfg)
                        .assign(scope="overall"),
                        grouped_metrics(st, METHODS, ["city", "regime", "horizon"], cfg).assign(scope="city")])
    t["regime"] = regime[["scope", *regime.columns[:-1]]]
    base, sens = pred.drop(columns=aci[u["aci"]["gamma"]].columns), []
    for g, a in aci.items():
        p = base.join(a)
        m = grouped_metrics(p, {"cqr_aci": METHODS["cqr_aci"]}, ["horizon"], cfg)
        sens.append(m.assign(gamma=g, mean_alpha=m["horizon"].map(scored(p).groupby("horizon")["aci_alpha"].mean()),
                             primary=g == u["aci"]["gamma"], label=SENS_LABEL))
    t["gamma"] = pd.concat(sens, ignore_index=True)[["gamma", *sens[0].columns[:-4], "mean_alpha", "primary", "label"]]
    held = pd.concat([t["city"].assign(setting="familiar")]
                     + [grouped_metrics(v["pred"], LOCO_METHODS, ["city", "horizon"], cfg)
                        .assign(setting="held_out_source_calibrated") for v in ho.values()]
                     + [grouped_metrics(v["pred"], LOCAL_METHODS, ["city", "horizon"], cfg)
                        .assign(setting="held_out_local_calibrated") for v in ho.values()], ignore_index=True)
    t["held_out"] = held[["setting", *t["city"].columns]]
    t["raw_and_cqr"] = raw_and_cqr_table(pred, q_hat, cfg)
    return t


def run(cfg: dict, dirs: dict | None = None) -> dict:
    dirs = dirs or da.output_dirs(cfg)
    check_config(cfg)
    u, core = cfg["uncertainty"], cfg["cities"][cfg["uncertainty"]["cities"]]
    sha, data = da.sha256_file(da.resolve(cfg, "raw_city_day")), frames(cfg)
    pred = predict_intervals(quantile_models(cfg, data["train"], dirs["models"], MODEL_STEM, sha), data, cfg)
    pred = mark_calibration_ready(pred, main_ranges(cfg)[1][1])
    scores = calibration_scores(pred, cfg, core)
    q_hat = q_hat_by_horizon(scores, cfg)
    pred = with_cqr(pred, q_hat, "cqr")
    aci = {g: aci_columns(pred, scores, cfg, g) for g in u["aci"]["gamma_sensitivity"]}
    pred = pred.join(aci[u["aci"]["gamma"]])
    thresholds = regime_thresholds(da.load_city_day(cfg), cfg)
    ho = {c: held_out_city(cfg, data, c, dirs["models"], sha) for c in core}
    t = tables(pred, with_strata(pred, thresholds, cfg), aci, ho, q_hat, cfg)
    is_test = pred["split"] == "test"
    checks = {"row_alignment_identical": True,
              **{m: interval_checks(pred[is_test & pred["calibration_ready"]] if m == "cqr_aci" else pred, lo, hi)
                 for m, (lo, hi) in METHODS.items()},
              "n_test_rows_not_in_eval_mask": {int(h): int((is_test & ~pred["eval_mask"] & (pred["horizon"] == h)).sum())
                                               for h in u["horizons"]},
              "n_test_rows_not_calibration_ready": not_ready(pred, u["horizons"])}
    hp = pd.concat([v["pred"] for v in ho.values()], ignore_index=True)[HELD_OUT_COLUMNS]
    ho_checks = {"held_out_intervals_test_rows": {
        m: interval_checks(hp[(hp["split"] == "test") & (hp["calibration_ready"] | (m != "aci_source"))], lo, hi,
                           ["city", "horizon"]) for m, (lo, hi) in {**LOCO_METHODS, **LOCAL_METHODS}.items()},
        "n_test_rows_not_calibration_ready": not_ready(hp, u["horizons"])}
    ho_q_hat = {c: v["q_hat"] for c, v in ho.items()}
    gen, uhash = "src.uncertainty.run", da.uncertainty_config_hash(cfg)
    da.write_csv(pred, dirs["predictions"] / PRED_FILE, cfg, gen, NOTES, uncertainty_config_hash=uhash, q_hat=q_hat,
                 checks=checks, aci_summary=aci_summary(pred, cfg))
    da.write_csv(hp, dirs["predictions"] / HELD_OUT_PRED_FILE, cfg, gen, HELD_OUT_NOTE, uncertainty_config_hash=uhash,
                 held_out_q_hat=ho_q_hat, checks=ho_checks)
    extra = {"held_out": {"held_out_q_hat": ho_q_hat, "checks": ho_checks},
             "regime": {"regime_thresholds": thresholds}}
    for name, df in t.items():
        da.write_csv(df, dirs["tables"] / TABLE_FILES[name], cfg, gen, NOTES, uncertainty_config_hash=uhash,
                     q_hat=q_hat, **extra.get(name, {}))
    return {"pred": pred, "tables": t, "q_hat": q_hat, "scores": scores, "thresholds": thresholds, "held_out": ho,
            "held_out_pred": hp, "checks": checks, "held_out_checks": ho_checks, "frames": data}


def bar_panel(t: pd.DataFrame, hs: list, col: str) -> dict:
    by = {m: t[t["method"] == m].set_index("horizon").loc[hs] for m in METHODS}
    return {"x": [f"h = {h}\nn = {k:,}" for h, k in zip(hs, by["cqr"]["n"])],
            "series": {m: d[col].tolist() for m, d in by.items()}}


def make_figures(cfg: dict, dirs: dict | None = None) -> list[Path]:
    dirs = dirs or da.output_dirs(cfg)
    u, f = cfg["uncertainty"], dirs["figures"]
    hs, core, nominal = u["horizons"], cfg["cities"][u["cities"]], 1 - u["alpha"]
    pred = load_predictions(dirs["predictions"] / PRED_FILE)
    t = {k: pd.read_csv(dirs["tables"] / TABLE_FILES[k]) for k in ("horizon", "city", "regime")}
    test, roll = scored(pred), rolling_coverage(pred)
    rolling = {k: (g["target_date"], g["rolling_coverage"]) for k, g in roll.groupby(["city", "horizon"])}
    alpha = {k: (g["origin_date"], g["aci_alpha"]) for k, g in test.sort_values("origin_date").groupby(["city", "horizon"])}
    tc = t["city"]
    gap = {lab: tc[tc["method"] == m].pivot(index="city", columns="horizon", values="coverage_gap").loc[core, hs].to_numpy()
           for lab, m in (("CQR", "cqr"), ("CQR+ACI", "cqr_aci"))}
    tr = t["regime"][t["regime"]["scope"] == "overall"]
    n_cell, gamma = f"{tc['n'].min()}-{tc['n'].max()}", u["aci"]["gamma"]
    return [
        figures.bars({"": bar_panel(t["horizon"], hs, "picp")}, f / "uncertainty_coverage_comparison.png", "PICP",
                     "Uncertainty: 2019 coverage by method and horizon (marginal)", nominal),
        figures.bars({"": bar_panel(t["horizon"], hs, "mpiw")}, f / "uncertainty_interval_width_comparison.png",
                     "MPIW (µg/m³)", "Uncertainty: 2019 mean prediction-interval width by method and horizon"),
        figures.stream_grid(rolling, core, hs, f / "uncertainty_aci_rolling_coverage.png",
                            f"rolling coverage (last {ROLLING_WINDOW} matured targets)",
                            f"CQR+ACI (gamma {gamma}): coverage of matured 2019 predictions, n = {n_cell} per city x horizon",
                            {nominal: "0.90"}),
        figures.stream_grid(alpha, core, hs, f / "uncertainty_aci_alpha_evolution.png", "alpha_t at issue",
                            f"CQR+ACI (gamma {gamma}): alpha_t by forecast origin, n = {n_cell} per city x horizon",
                            {b: f"bound {b:.2f}" for b in u["aci"]["alpha_bounds"]}),
        figures.heatmaps(gap, core, hs, f / "uncertainty_city_horizon_coverage.png",
                         f"Uncertainty: PICP − 0.90 by city and horizon (2019, n = {n_cell} per cell)", "PICP − 0.90"),
        figures.bars({f"{g} days": bar_panel(tr[tr["regime"] == g], hs, "picp") for g in ("normal", "top10")},
                     f / "uncertainty_extreme_pollution_coverage.png", "PICP",
                     "Uncertainty: coverage on normal vs top-10% days (realised-outcome strata, 2019)", nominal)]
