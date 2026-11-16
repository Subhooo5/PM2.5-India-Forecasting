from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import data_audit as da
from src import figures
from src.leakage_protocols import (FILES, load_saved, regenerate_protocol_a, run_protocol_a, run_protocol_b,
                                   run_protocol_c)
from src.metrics import inflation_pct, point_metrics, skill_score
from src.splits import main_ranges

DIAG_FOCUS_CITY = "Chennai"
LABEL_AB = ("interleaved random vs chronological split, with different test composition; "
            "NOT a pure split effect")
LABEL_BC = "same-day estimation vs genuine forecasting; two DIFFERENT tasks, not an isolated leakage effect"
SIGN_NOTE = ("rel_inflation_*_pct = 100*(1 - error_left/error_right), left = optimistic protocol "
             "(A in A_vs_B, B in B_vs_C); positive = left has lower error. *_diff = left - right; "
             "r2_diff_points = 100*(r2_left - r2_right). Skill: A/B vs persistence h=1 (y[d-1]) on "
             "rows where y[d-1] is observed (n in skill_n_*); C vs h-matched persistence y[d-h] "
             "on identical rows.")
COMPARISON_COLUMNS = ["comparison", "scope", "seed", "horizon", "n_left", "n_right", "mae_left", "rmse_left",
                      "r2_left", "mae_right", "rmse_right", "r2_right", "mae_diff", "rmse_diff", "r2_diff_points",
                      "rel_inflation_mae_pct", "rel_inflation_rmse_pct", "skill_mae_left", "skill_mae_right",
                      "skill_n_left", "skill_n_right", "notes", "dataset_sha256", "kaggle_version",
                      "config_hash", "selected_candidate"]


def persistence_h1(cfg: dict) -> pd.Series:
    raw = da.load_city_day(cfg)
    s = raw.loc[raw["City"].isin(cfg["cities"]["core"])].set_index(["City", "Date"])["PM2.5"]
    return s.reset_index().assign(Date=lambda x: x["Date"] + pd.Timedelta(days=1)).set_index(["City", "Date"])["PM2.5"]


def same_day_skill(rows: pd.DataFrame, pers1: pd.Series) -> tuple[float, int]:
    ref = pers1.reindex(pd.MultiIndex.from_arrays([rows["city"].astype(str), rows["target_date"]]))
    ok = ref.notna().to_numpy()
    if ok.sum() == 0:
        return float("nan"), 0
    m = point_metrics(rows["y_true"].to_numpy()[ok], rows["y_pred"].to_numpy()[ok])
    r = point_metrics(rows["y_true"].to_numpy()[ok], ref.to_numpy()[ok])
    return float(skill_score(m["mae"], r["mae"])), int(ok.sum())


def forecast_skill(rows: pd.DataFrame, c_pers: pd.DataFrame) -> tuple[float, int]:
    keys = ["city", "target_date", "horizon"]
    ref = rows[keys + ["y_true", "y_pred"]].merge(
        c_pers[keys + ["y_pred"]].rename(columns={"y_pred": "y_ref"}), on=keys, how="left")
    if ref["y_ref"].isna().any():
        raise AssertionError("C persistence missing on scored rows")
    m, r = point_metrics(ref["y_true"], ref["y_pred"]), point_metrics(ref["y_true"], ref["y_ref"])
    return float(skill_score(m["mae"], r["mae"])), int(len(ref))


def compare_row(comparison, scope, seed, horizon, left, right, skill_left, skill_right, notes, meta_cols) -> dict:
    ml, mr = point_metrics(left["y_true"], left["y_pred"]), point_metrics(right["y_true"], right["y_pred"])
    return {"comparison": comparison, "scope": scope, "seed": seed, "horizon": horizon, "n_left": ml["n"],
            "n_right": mr["n"], "mae_left": ml["mae"], "rmse_left": ml["rmse"], "r2_left": ml["r2"],
            "mae_right": mr["mae"], "rmse_right": mr["rmse"], "r2_right": mr["r2"],
            "mae_diff": ml["mae"] - mr["mae"], "rmse_diff": ml["rmse"] - mr["rmse"],
            "r2_diff_points": 100 * (ml["r2"] - mr["r2"]),
            "rel_inflation_mae_pct": float(inflation_pct(ml["mae"], mr["mae"])),
            "rel_inflation_rmse_pct": float(inflation_pct(ml["rmse"], mr["rmse"])),
            "skill_mae_left": skill_left[0], "skill_mae_right": skill_right[0], "skill_n_left": skill_left[1],
            "skill_n_right": skill_right[1], "notes": notes, **meta_cols}


def seed_summary(rows: list[dict], comparison: str, scope: str, meta_cols: dict, notes: str) -> list[dict]:
    df = pd.DataFrame(rows)
    num = [c for c in COMPARISON_COLUMNS if c not in ("comparison", "scope", "seed", "horizon", "notes", *meta_cols)]
    return [{"comparison": comparison, "scope": scope, "seed": stat, "horizon": 0,
             **df[num].astype(float).agg(fn).to_dict(), "notes": notes + " sd = sample sd (ddof=1).", **meta_cols}
            for stat, fn in (("5-seed mean", "mean"), ("5-seed sd", "std"))]


def rows_on(df: pd.DataFrame, keys: pd.MultiIndex) -> pd.DataFrame:
    out = df.loc[pd.MultiIndex.from_frame(df[["city", "target_date"]]).isin(keys)]
    if len(out) != len(keys):
        raise AssertionError("rows on key set are not unique / complete")
    return out.sort_values(["city", "target_date"])


def bc_common_keys(B: pd.DataFrame, C: pd.DataFrame, h: int) -> pd.MultiIndex:
    c = C[(C["method"] == "lightgbm") & (C["horizon"] == h) & C["eval_mask"]]
    return (pd.MultiIndex.from_frame(B[["city", "target_date"]])
            .intersection(pd.MultiIndex.from_frame(c[["city", "target_date"]])).sort_values())


def diagnostics(cfg: dict, saved: dict) -> dict:
    (tr_lo, tr_hi), _, (te_lo, te_hi) = main_ranges(cfg)
    A, C = saved["A"], saved["C"]
    err = A.assign(abs_err=(A["y_pred"] - A["y_true"]).abs(), sq=(A["y_pred"] - A["y_true"]) ** 2,
                   year=A["target_date"].dt.year)
    by = lambda k: err.groupby(k).agg(n=("abs_err", "size"), mae=("abs_err", "mean"), rmse=("sq", "mean"),
                                      mean_y=("y_true", "mean")).assign(rmse=lambda d: np.sqrt(d["rmse"])).reset_index()
    raw = da.load_city_day(cfg)
    core = raw[raw["City"].isin(cfg["cities"]["core"])]
    y19 = core[core["Date"].between(te_lo, te_hi) & core["PM2.5"].notna()]
    stats = y19.groupby("City")["PM2.5"].agg(n="count", mean="mean", var="var", std="std",
                                             p99=lambda s: s.quantile(0.99), max="max").reset_index()
    stats["cv"] = stats["std"] / stats["mean"]
    stats["days_gt_mean_3sd"] = [int((g > g.mean() + 3 * g.std()).sum()) for _, g in y19.groupby("City")["PM2.5"]]
    c1 = C[(C["horizon"] == 1) & C["eval_mask"]]
    lg = c1[c1["method"] == "lightgbm"].set_index(["city", "target_date"])
    pe = c1[c1["method"] == "persistence"].set_index(["city", "target_date"])
    spikes = [{"date": str(r["Date"].date()), "y": float(r["PM2.5"]),
               "c_h1_lightgbm": float(lg["y_pred"].get((DIAG_FOCUS_CITY, r["Date"]), np.nan)),
               "c_h1_persistence": float(pe["y_pred"].get((DIAG_FOCUS_CITY, r["Date"]), np.nan))}
              for _, r in y19[y19["City"] == DIAG_FOCUS_CITY].nlargest(5, "PM2.5").iterrows()]
    med = core[core["Date"].between(tr_lo, tr_hi)].groupby("City")["PM2.5"].median()
    j = lg[["y_true", "y_pred"]].join(pe[["y_pred"]].rename(columns={"y_pred": "y_pers"})).reset_index()
    j["level"] = np.where(j["y_true"] > j["city"].map(med), "above_train_median", "at_or_below_train_median")
    rows = []
    for (scope, key), g in [(("overall", "all"), j)] + [(("city", c), g) for c, g in j.groupby("city")]:
        for level, gg in g.groupby("level"):
            ml, mp = point_metrics(gg["y_true"], gg["y_pred"]), point_metrics(gg["y_true"], gg["y_pers"])
            rows.append({"scope": scope, "city": key, "level": level, "n": ml["n"], "mae_lightgbm": ml["mae"],
                         "mae_persistence": mp["mae"], "mae_diff_lgbm_minus_pers": ml["mae"] - mp["mae"],
                         "rmse_lightgbm": ml["rmse"], "rmse_persistence": mp["rmse"],
                         "rmse_diff_lgbm_minus_pers": ml["rmse"] - mp["rmse"]})
    return {"i_A_error_by_year": by("year"), "i_A_error_by_city": by("city"),
            "i_A_top10_abs_errors": err.nlargest(10, "abs_err")[["city", "target_date", "y_true", "y_pred", "abs_err"]],
            "ii_2019_pm25_by_city": stats, "ii_chennai_top5_spikes": pd.DataFrame(spikes),
            "iii_training_median_by_city": med.rename("train_median_2015_2017").reset_index(),
            "iii_c_h1_lgbm_vs_persistence_by_level": pd.DataFrame(rows)}


def run_comparison(cfg: dict, dirs: dict | None = None) -> dict:
    dirs = dirs or da.output_dirs(cfg)
    saved = load_saved(cfg, dirs)
    seed_preds = regenerate_protocol_a(cfg, saved)
    B, C = saved["B"], saved["C"]
    meta_cols = {"dataset_sha256": cfg["data"]["sha256"], "kaggle_version": cfg["data"]["kaggle_version"],
                 "config_hash": da.config_hash(cfg), "selected_candidate": cfg["lightgbm"]["selected"]["candidate_id"]}
    pers1, c_pers = persistence_h1(cfg), C[C["method"] == "persistence"]
    b_skill, (_, _, (te_lo, te_hi)) = same_day_skill(B, pers1), main_ranges(cfg)
    ab_note = f"{LABEL_AB}. {SIGN_NOTE}"
    scopes = ("a_A_all_test_vs_B_full_2019", "b_A_2019_rows_vs_B_full_2019", "c_matched_2019_city_dates")
    per = {s: [] for s in scopes}
    for seed, A in sorted(seed_preds.items()):
        A19 = A[A["target_date"].between(te_lo, te_hi)]
        B_m = rows_on(B, pd.MultiIndex.from_frame(A19[["city", "target_date"]].astype({"city": str})))
        A19s = A19.assign(city=A19["city"].astype(str)).sort_values(["city", "target_date"])
        for scope, left, right, rskill in ((scopes[0], A, B, b_skill), (scopes[1], A19, B, b_skill),
                                           (scopes[2], A19s, B_m, same_day_skill(B_m, pers1))):
            per[scope].append(compare_row("A_vs_B", scope, str(seed), 0, left, right, same_day_skill(left, pers1),
                                          rskill, ab_note, meta_cols))
    rows = [r for s in scopes for r in per[s] + seed_summary(per[s], "A_vs_B", s, meta_cols, ab_note)]
    bc_counts = {}
    for h in cfg["leakage"]["protocol_C"]["horizons"]:
        keys = bc_common_keys(B, C, h)
        bc_counts[h] = len(keys)
        c_lgb = C[(C["method"] == "lightgbm") & (C["horizon"] == h)]
        B_k, C_k, C_full = rows_on(B, keys), rows_on(c_lgb, keys), c_lgb[c_lgb["eval_mask"]]
        rows.append(compare_row("B_vs_C", "primary_common_city_dates", "", h, B_k, C_k, same_day_skill(B_k, pers1),
                                forecast_skill(C_k, c_pers), f"{LABEL_BC}. {SIGN_NOTE}", meta_cols))
        rows.append(compare_row("B_vs_C", "full_sets_own_n", "", h, B, C_full, b_skill, forecast_skill(C_full, c_pers),
                                f"{LABEL_BC}. Each protocol on its own full evaluation set. {SIGN_NOTE}", meta_cols))
    table = pd.DataFrame(rows)[COMPARISON_COLUMNS]
    diag = diagnostics(cfg, saved)
    da.write_csv(table, dirs["tables"] / FILES["comparison"], cfg, "src.leakage.run_comparison",
                 "A vs B and B vs C on matched rows; see the notes column.",
                 split_composition=saved["A_meta"]["split_composition"], b_vs_c_common_counts=bc_counts,
                 diagnostics={k: json.loads(v.to_json(orient="records", date_format="iso")) for k, v in diag.items()})
    return {"table": table, "diagnostics": diag, "bc_counts": bc_counts, "seed_preds": seed_preds}


def run(cfg: dict) -> dict:
    return {"C": run_protocol_c(cfg), "B": run_protocol_b(cfg), "A": run_protocol_a(cfg),
            "comparison": run_comparison(cfg)}


def make_figures(cfg: dict, dirs: dict | None = None) -> list[Path]:
    dirs = dirs or da.output_dirs(cfg)
    saved, table = load_saved(cfg, dirs), pd.read_csv(dirs["tables"] / FILES["comparison"])
    seeds = saved["A_seeds"].astype({"seed": str})
    a_mean, a_sd = (seeds[(seeds["seed"] == s) & (seeds["subset"] == "all_test")].iloc[0] for s in ("mean", "sd"))
    B, C, hs = saved["B"], saved["C"], cfg["leakage"]["protocol_C"]["horizons"]
    mb = point_metrics(B["y_true"], B["y_pred"])
    mc, mp = {}, {}
    for h in hs:
        rows = C[(C["horizon"] == h) & C["eval_mask"]]
        lg, pe = rows[rows["method"] == "lightgbm"], rows[rows["method"] == "persistence"]
        mc[h], mp[h] = point_metrics(lg["y_true"], lg["y_pred"]), point_metrics(pe["y_true"], pe["y_pred"])
    labels = ["A\n(random)", "B\n(chrono)"] + [f"C\nh = {h}" for h in hs]
    perf = {metric: {"labels": labels, "groups": [0, 1] + [2] * len(hs),
                     "values": [a_mean[key], mb[key]] + [mc[h][key] for h in hs],
                     "err": [a_sd[key], None] + [None] * len(hs), "persistence": [None, None] + [mp[h][key] for h in hs]}
            for metric, key in (("MAE", "mae"), ("RMSE", "rmse"), ("R²", "r2"))}
    ab = table[(table["comparison"] == "A_vs_B") & (table["scope"] == "c_matched_2019_city_dates")]
    abm, abs_ = ab[ab["seed"] == "5-seed mean"].iloc[0], ab[ab["seed"] == "5-seed sd"].iloc[0]
    bc = table[(table["comparison"] == "B_vs_C") & (table["scope"] == "primary_common_city_dates")]
    infl = {metric: {"labels": ["A vs B\nmatched"] + [f"B vs C\nh = {h}" for h in bc["horizon"]],
                     "values": [abm[col]] + bc[col].tolist(), "err": [abs_[col]] + [None] * len(bc),
                     "groups": [0] + [1] * len(bc)}
            for metric, col in (("MAE", "rel_inflation_mae_pct"), ("RMSE", "rel_inflation_rmse_pct"))}
    return [figures.performance_comparison(perf, dirs["figures"] / "leakage_performance_comparison.png"),
            figures.leakage_inflation(infl, dirs["figures"] / "leakage_inflation.png")]
