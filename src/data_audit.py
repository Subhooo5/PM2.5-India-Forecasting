"""Raw-data audit (Day 1, items 1-3).

1. Load the canonical city_day.csv (read-only; nothing is ever written to data/raw).
2. Verify SHA-256, dimensions, date range, columns and (City, Date) uniqueness.
3. City-wise PM2.5 completeness and gap-length reports, measured against the FULL
   daily calendar (dataset range), not against row counts.

Every table is written as a plain CSV plus a `<name>.meta.json` sidecar recording the
dataset sha256, Kaggle version, git commit, timestamp and generating function.

Run:  python -m src.data_audit
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiments.yaml"


# --------------------------------------------------------------------------- utils
def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    """Absolute path for cfg['paths'][key] (paths are relative to the repo root)."""
    return ROOT / cfg["paths"][key]


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": None, "git_dirty": None}


def build_metadata(df: pd.DataFrame, file_name: str, cfg: dict, generator: str,
                   notes: str = "") -> dict:
    """Provenance record attached to every output table / dataset."""
    return {
        "file": file_name,
        "dataset_sha256": sha256_file(resolve(cfg, "raw_city_day")),
        "kaggle_dataset": cfg["data"]["source"],
        "kaggle_version": cfg["data"]["kaggle_version"],
        # hash of the config actually used (canonical JSON), not of the file on disk
        "config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
        "generator": generator,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "seed": cfg.get("seed"),
        "notes": notes,
        **_git_commit(),
    }


def write_sidecar(meta: dict, data_path: Path) -> Path:
    """Write <stem>.meta.json next to data_path."""
    side = data_path.with_name(f"{data_path.stem}.meta.json")
    side.write_text(json.dumps(meta, indent=2, default=str))
    return side


def write_table(df: pd.DataFrame, name: str, cfg: dict, generator: str,
                notes: str = "") -> Path:
    """Write outputs/tables/<name>.csv and its <name>.meta.json metadata sidecar."""
    out_dir = resolve(cfg, "tables_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    write_sidecar(build_metadata(df, csv_path.name, cfg, generator, notes), csv_path)
    return csv_path


# ------------------------------------------------------------------ items 1 and 2
def load_city_day(cfg: dict) -> pd.DataFrame:
    """Item 1: read the canonical city_day.csv. Read-only; never re-saves the raw file."""
    df = pd.read_csv(resolve(cfg, "raw_city_day"), dtype={"City": str})
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    return df


class RawDataError(ValueError):
    """Raised when the raw file does not match the frozen dataset identity."""


def verify_raw(df: pd.DataFrame, cfg: dict, path: Path | str | None = None) -> dict:
    """Item 2: verify sha256, shape, cities, date range, columns, uniqueness. Raises on mismatch."""
    d = cfg["data"]
    path = Path(path) if path is not None else resolve(cfg, "raw_city_day")
    checks = {
        "sha256": (sha256_file(path), d["sha256"]),
        "n_rows": (len(df), d["n_rows"]),
        "n_cities": (df["City"].nunique(), d["n_cities"]),
        "date_min": (df["Date"].min().strftime("%Y-%m-%d"), d["date_min"]),
        "date_max": (df["Date"].max().strftime("%Y-%m-%d"), d["date_max"]),
        "columns": (list(df.columns), list(d["columns"])),
        "duplicate_city_date": (int(df.duplicated(["City", "Date"]).sum()), 0),
    }
    failed = {k: v for k, v in checks.items() if v[0] != v[1]}
    if failed:
        msg = "; ".join(f"{k}: got {g!r}, expected {e!r}" for k, (g, e) in failed.items())
        raise RawDataError(f"Raw data verification failed -> {msg}")
    return {k: v[0] for k, v in checks.items()}


# ------------------------------------------------------------------------- item 3
def pm25_calendar(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per (city, day) over the full dataset date range; PM2.5 NaN where absent.

    Also returns `row_present` so missing rows can be told apart from rows with NaN PM2.5.
    """
    tgt = cfg["data"]["target"]
    days = pd.date_range(cfg["data"]["date_min"], cfg["data"]["date_max"], freq="D")
    idx = pd.MultiIndex.from_product([sorted(df["City"].unique()), days], names=["City", "Date"])
    indexed = df.set_index(["City", "Date"])
    cal = indexed[[tgt]].reindex(idx)
    cal["row_present"] = idx.isin(indexed.index)
    return cal.reset_index().rename(columns={tgt: "pm25"})


def completeness_by_year(cal: pd.DataFrame) -> pd.DataFrame:
    g = cal.assign(year=cal["Date"].dt.year).groupby(["City", "year"])
    out = pd.DataFrame({
        "expected_days": g.size(),
        "rows_present": g["row_present"].sum(),
        "pm25_observed": g["pm25"].count(),
    }).reset_index()
    out["pct_observed"] = (100 * out["pm25_observed"] / out["expected_days"]).round(1)
    return out


def completeness_by_period(cal: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for city, c in cal.groupby("City"):
        obs = c.loc[c["pm25"].notna(), "Date"]
        row = {"City": city,
               "first_pm25": obs.min().date() if len(obs) else None,
               "last_pm25": obs.max().date() if len(obs) else None,
               "pm25_observed_total": int(len(obs))}
        for name, (lo, hi) in cfg["audit"]["periods"].items():
            m = c["Date"].between(lo, hi)
            n_exp, n_obs = int(m.sum()), int(c.loc[m, "pm25"].notna().sum())
            row[f"{name}_observed"] = n_obs
            row[f"{name}_pct"] = round(100 * n_obs / n_exp, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def gap_runs(observed: pd.Series) -> tuple[int, int, list[int]]:
    """Split a boolean 'observed' series into (leading_missing, trailing_missing, internal gap lengths).

    Internal gaps are runs of missing days strictly between the first and last observation.
    """
    obs = np.asarray(observed, dtype=bool)
    if not obs.any():
        return len(obs), 0, []
    first, last = int(obs.argmax()), int(len(obs) - 1 - obs[::-1].argmax())
    inner = ~obs[first:last + 1]
    gaps, run = [], 0
    for missing in inner:
        if missing:
            run += 1
        elif run:
            gaps.append(run)
            run = 0
    return first, len(obs) - 1 - last, gaps


def gap_stats(cal: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    b = cfg["audit"]["gap_bins"]  # [1, 2, 3, 7]
    rows = []
    for city, c in cal.sort_values("Date").groupby("City"):
        lead, trail, gaps = gap_runs(c["pm25"].notna())
        g = np.array(gaps, dtype=int)
        rows.append({
            "City": city,
            "leading_missing_days": lead,
            "trailing_missing_days": trail,
            "n_internal_gaps": len(g),
            "internal_missing_days": int(g.sum()),
            f"gaps_len_{b[0]}": int((g == b[0]).sum()),
            f"gaps_len_{b[1]}": int((g == b[1]).sum()),
            f"gaps_len_{b[2]}_{b[3]}": int(((g >= b[2]) & (g <= b[3])).sum()),
            f"gaps_len_gt{b[3]}": int((g > b[3]).sum()),
            "longest_internal_gap": int(g.max()) if len(g) else 0,
            "median_internal_gap": float(np.median(g)) if len(g) else 0.0,
            # input days that ffill(limit=k) would fill inside internal gaps
            "days_fillable_limit2": int(np.minimum(g, 2).sum()),
            "days_fillable_limit3": int(np.minimum(g, 3).sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main
def run_audit(cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    df = load_city_day(cfg)
    summary = verify_raw(df, cfg)
    cal = pm25_calendar(df, cfg)
    tables: dict[str, Any] = {
        "pm25_completeness_by_city_year": completeness_by_year(cal),
        "pm25_completeness_by_city_period": completeness_by_period(cal, cfg),
        "pm25_gap_stats": gap_stats(cal, cfg),
    }
    note = "Completeness measured against the full daily calendar " \
           f"{cfg['data']['date_min']}..{cfg['data']['date_max']} for every city."
    for name, t in tables.items():
        write_table(t, name, cfg, generator=f"src.data_audit.{name}", notes=note)
    tables["_verify_summary"] = summary
    return tables


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 100)
    res = run_audit()
    print("Verification passed:", {k: v for k, v in res["_verify_summary"].items() if k != "columns"})
    per = res["pm25_completeness_by_city_period"]
    pct_cols = [c for c in per.columns if c.endswith("_pct")]
    print("\nPM2.5 % observed by split period:")
    print(per[["City", "first_pm25", "last_pm25", *pct_cols]]
          .sort_values("test_2019_pct", ascending=False).to_string(index=False))
    print("\nGap stats:")
    print(res["pm25_gap_stats"].to_string(index=False))
