from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiments.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def repo_path(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def resolve(cfg: dict, key: str) -> Path:
    return repo_path(cfg["paths"][key])


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def config_hash(cfg: dict) -> str:
    """Hash of the shared design; excludes the tuned selection and the uncertainty and stress sections."""
    shared = copy.deepcopy(cfg)
    shared["lightgbm"].pop("selected", None)
    shared.pop("uncertainty", None)
    shared.pop("stress_test", None)
    return sha256_json(shared)


def uncertainty_config_hash(cfg: dict) -> str:
    return sha256_json(cfg["uncertainty"])


def stress_config_hash(cfg: dict) -> str:
    return sha256_json({"stress_test": cfg["stress_test"], "splits_stress": cfg["splits"]["stress"],
                        "lightgbm_selected": cfg["lightgbm"]["selected"]})


def _git() -> dict:
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True)
        return {"git_commit": run("rev-parse", "HEAD").stdout.strip(),
                "git_dirty": bool(run("status", "--porcelain").stdout.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": None, "git_dirty": None}


def metadata(df: pd.DataFrame, file_name: str, cfg: dict, generator: str, notes: str = "") -> dict:
    return {"file": file_name, "dataset_sha256": sha256_file(resolve(cfg, "raw_city_day")),
            "kaggle_dataset": cfg["data"]["source"], "kaggle_version": cfg["data"]["kaggle_version"],
            "config_sha256": sha256_json(cfg), "config_hash": config_hash(cfg),
            "selected_candidate": cfg["lightgbm"]["selected"]["candidate_id"], "generator": generator,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_rows": int(len(df)),
            "columns": list(map(str, df.columns)), "seed": cfg["seed"],
            "versions": {"lightgbm": lightgbm.__version__, "numpy": np.__version__, "pandas": pd.__version__},
            "notes": notes, **_git()}


def write_sidecar(meta: dict, data_path: Path) -> Path:
    side = data_path.with_name(f"{data_path.stem}.meta.json")
    side.write_text(json.dumps(meta, indent=2, default=str))
    return side


def write_csv(df: pd.DataFrame, path: Path, cfg: dict, generator: str, notes: str = "", **extra) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    write_sidecar({**metadata(df, path.name, cfg, generator, notes), **extra}, path)
    return path


def output_dirs(cfg: dict) -> dict:
    return {"models": resolve(cfg, "models_dir"), "predictions": resolve(cfg, "predictions_dir"),
            "tables": resolve(cfg, "tables_dir"), "figures": resolve(cfg, "figures_dir")}


def load_city_day(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(resolve(cfg, "raw_city_day"), dtype={"City": str})
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    return df


class RawDataError(ValueError):
    pass


def verify_raw(df: pd.DataFrame, cfg: dict, path=None) -> dict:
    d = cfg["data"]
    checks = {"sha256": (sha256_file(path or resolve(cfg, "raw_city_day")), d["sha256"]),
              "n_rows": (len(df), d["n_rows"]), "n_cities": (df["City"].nunique(), d["n_cities"]),
              "date_min": (df["Date"].min().strftime("%Y-%m-%d"), d["date_min"]),
              "date_max": (df["Date"].max().strftime("%Y-%m-%d"), d["date_max"]),
              "columns": (list(df.columns), list(d["columns"])),
              "duplicate_city_date": (int(df.duplicated(["City", "Date"]).sum()), 0)}
    failed = {k: v for k, v in checks.items() if v[0] != v[1]}
    if failed:
        raise RawDataError("raw data verification failed -> " + "; ".join(
            f"{k}: got {g!r}, expected {e!r}" for k, (g, e) in failed.items()))
    return {k: v[0] for k, v in checks.items()}


def load_verified(cfg: dict) -> pd.DataFrame:
    df = load_city_day(cfg)
    verify_raw(df, cfg)
    return df


def pm25_calendar(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    tgt = cfg["data"]["target"]
    days = pd.date_range(cfg["data"]["date_min"], cfg["data"]["date_max"], freq="D")
    idx = pd.MultiIndex.from_product([sorted(df["City"].unique()), days], names=["City", "Date"])
    indexed = df.set_index(["City", "Date"])
    cal = indexed[[tgt]].reindex(idx)
    cal["row_present"] = idx.isin(indexed.index)
    return cal.reset_index().rename(columns={tgt: "pm25"})


def completeness_by_year(cal: pd.DataFrame) -> pd.DataFrame:
    g = cal.assign(year=cal["Date"].dt.year).groupby(["City", "year"])
    out = pd.DataFrame({"expected_days": g.size(), "rows_present": g["row_present"].sum(),
                        "pm25_observed": g["pm25"].count()}).reset_index()
    out["pct_observed"] = (100 * out["pm25_observed"] / out["expected_days"]).round(1)
    return out


def completeness_by_period(cal: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for city, c in cal.groupby("City"):
        obs = c.loc[c["pm25"].notna(), "Date"]
        row = {"City": city, "first_pm25": obs.min().date() if len(obs) else None,
               "last_pm25": obs.max().date() if len(obs) else None, "pm25_observed_total": int(len(obs))}
        for name, (lo, hi) in cfg["audit"]["periods"].items():
            m = c["Date"].between(lo, hi)
            n_obs = int(c.loc[m, "pm25"].notna().sum())
            row[f"{name}_observed"], row[f"{name}_pct"] = n_obs, round(100 * n_obs / int(m.sum()), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def gap_runs(observed) -> tuple[int, int, list[int]]:
    obs = np.asarray(observed, dtype=bool)
    if not obs.any():
        return len(obs), 0, []
    first, last = int(obs.argmax()), int(len(obs) - 1 - obs[::-1].argmax())
    gaps, run = [], 0
    for missing in ~obs[first:last + 1]:
        if missing:
            run += 1
        elif run:
            gaps.append(run)
            run = 0
    return first, len(obs) - 1 - last, gaps


def gap_stats(cal: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    b, rows = cfg["audit"]["gap_bins"], []
    for city, c in cal.sort_values("Date").groupby("City"):
        lead, trail, gaps = gap_runs(c["pm25"].notna())
        g = np.array(gaps, dtype=int)
        rows.append({"City": city, "leading_missing_days": lead, "trailing_missing_days": trail,
                     "n_internal_gaps": len(g), "internal_missing_days": int(g.sum()),
                     f"gaps_len_{b[0]}": int((g == b[0]).sum()), f"gaps_len_{b[1]}": int((g == b[1]).sum()),
                     f"gaps_len_{b[2]}_{b[3]}": int(((g >= b[2]) & (g <= b[3])).sum()),
                     f"gaps_len_gt{b[3]}": int((g > b[3]).sum()),
                     "longest_internal_gap": int(g.max()) if len(g) else 0,
                     "median_internal_gap": float(np.median(g)) if len(g) else 0.0,
                     "days_fillable_limit2": int(np.minimum(g, 2).sum()),
                     "days_fillable_limit3": int(np.minimum(g, 3).sum())})
    return pd.DataFrame(rows)


def run_audit(cfg: dict) -> dict:
    cal = pm25_calendar(load_verified(cfg), cfg)
    tables = {"pm25_completeness_by_city_year": completeness_by_year(cal),
              "pm25_completeness_by_city_period": completeness_by_period(cal, cfg),
              "pm25_gap_stats": gap_stats(cal, cfg)}
    tables_dir = output_dirs(cfg)["tables"]
    for name, t in tables.items():
        write_csv(t, tables_dir / f"{name}.csv", cfg, f"src.data_audit.{name}",
                  "Completeness against the full daily calendar of every city.")
    return tables
