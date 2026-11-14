"""Experiment runners.

Day 2 task 2: `run_tuning(config)`, the expanding-window hyperparameter selection, run ONCE.
- Reads the Day 1 forecasting parquet with a read-time filter target_date <= 2017-12-31,
  so 2018+ rows are never materialised; then keeps only the five core cities and the
  tuning horizons, and asserts nothing later than the hard guard remains.
- For each fold (2015->2016, 2015-16->2017) x candidate (c1..c6) x h in {1,3,7}: fits a
  pooled direct LightGBM (lightgbm.base + candidate) on the fold's training rows and scores
  the validation rows on `eval_mask` (MAE/RMSE/R2/n).
- Selects the candidate with the lowest mean validation MAE over the 2 folds x 3 horizons;
  ties go to the earliest candidate in grid order.
There is deliberately no run-everything entry point.

TODO(RQ1 - later day): leakage protocols A (random split + same-day AQI/co-pollutants, invalid demo),
B (chronological + same-day), C (chronological + strictly causal). B vs C are different tasks.
TODO(RQ2 - later day): pooled vs per-city vs leave-one-city-out (zero-shot).
TODO(RQ3 - later day): quantile LightGBM vs CQR vs CQR+ACI; stress test 2020 reported separately.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

from src import data_audit as da
from src.features import feature_set
from src.metrics import point_metrics
from src.models import STRUCTURAL_KEYS, candidate_params, merge_params
from src.splits import LeakageError, assert_training_only, expanding_window_folds

CV_COLUMNS = ["candidate_id", "num_leaves", "min_child_samples", "fold", "horizon",
              "n_train", "n_val", "mae", "rmse", "r2"]


def config_hash(cfg: dict) -> str:
    """sha256 of the config with `lightgbm.selected` removed (the frozen, predeclared part)."""
    frozen = copy.deepcopy(cfg)
    frozen.get("lightgbm", {}).pop("selected", None)
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, default=str).encode()).hexdigest()


def _abs(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else da.ROOT / path


def load_tuning_frame(cfg: dict, path=None) -> pd.DataFrame:
    """Tuning rows only. 2018+ rows are filtered out AT READ TIME and never enter memory."""
    path = _abs(path or cfg["paths"]["processed_dataset"])
    max_date = pd.Timestamp(cfg["tuning"]["max_allowed_target_date"])
    frame = pd.read_parquet(path, filters=[("target_date", "<=", max_date)])
    cities = list(cfg["cities"][cfg["tuning"]["cities"]])
    frame = frame.loc[frame["city"].astype(str).isin(cities)
                      & frame["horizon"].isin(cfg["tuning"]["horizons"])].reset_index(drop=True)
    if frame["target_date"].max() > max_date:
        raise LeakageError("tuning frame contains target dates after the hard guard")
    if set(frame["city"].astype(str)) != set(cities):
        raise ValueError("tuning frame does not contain exactly the core cities")
    assert_training_only(frame["target_date"], cfg)
    return frame


def tune_frame(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """CV results (one row per fold x candidate x horizon). Raises on any row after the guard."""
    assert_training_only(frame["target_date"], cfg)         # (a) refuses 2018+ rows outright
    if frame["y_target"].isna().any():
        raise ValueError("training/validation rows must have an observed target (never imputed)")
    feats = feature_set(cfg, cfg["tuning"]["feature_set"])
    eval_col = cfg["tuning"]["eval_rows"]
    folds = expanding_window_folds(frame, cfg)
    records = []
    for fold_no, (tri, vai) in enumerate(folds, start=1):
        for cand in candidate_params(cfg):
            params = merge_params(cfg, cand)
            for h in cfg["tuning"]["horizons"]:
                tr = frame.loc[tri][lambda d: d["horizon"] == h]
                va = frame.loc[vai][lambda d: (d["horizon"] == h) & d[eval_col]]
                assert_training_only(tr["target_date"], cfg)
                assert_training_only(va["target_date"], cfg)
                model = lightgbm.LGBMRegressor(**params)     # no eval set, no early stopping
                model.fit(tr[feats], tr["y_target"])
                m = point_metrics(va["y_target"], model.predict(va[feats]))
                records.append({"candidate_id": cand["candidate_id"],
                                **{k: cand[k] for k in STRUCTURAL_KEYS},
                                "fold": fold_no, "horizon": h, "n_train": len(tr),
                                "n_val": m["n"], "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"]})
    return pd.DataFrame(records, columns=CV_COLUMNS)


def select_candidate(cv: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Lowest mean validation MAE over all folds x horizons; ties -> earliest in grid order."""
    order = [c["candidate_id"] for c in candidate_params(cfg)]
    expected_cells = len(cfg["tuning"]["folds"]) * len(cfg["tuning"]["horizons"])
    counts = cv.groupby("candidate_id").size().reindex(order)
    if counts.isna().any() or (counts != expected_cells).any():
        raise ValueError("every candidate needs exactly one score per fold x horizon")
    mean_mae = cv.groupby("candidate_id")["mae"].mean().reindex(order)
    per_h = cv.pivot_table(index="candidate_id", columns="horizon", values="mae", aggfunc="mean")
    per_h = per_h.reindex(order).add_prefix("mean_mae_h")
    best_id = next(cid for cid in order if mean_mae[cid] == mean_mae.min())   # earliest tie
    ranking = pd.DataFrame({"candidate_id": order}).merge(
        pd.DataFrame(candidate_params(cfg)), on="candidate_id")
    ranking["mean_val_mae"] = mean_mae.to_numpy()
    ranking = ranking.join(per_h.reset_index(drop=True))
    ranking["grid_order"] = range(1, len(order) + 1)
    ranking["rank"] = ranking.sort_values(["mean_val_mae", "grid_order"]).index.argsort() + 1
    ranking["selected"] = ranking["candidate_id"] == best_id
    selected = next(c for c in candidate_params(cfg) if c["candidate_id"] == best_id)
    return selected, ranking


def run_tuning(config: dict, data_path=None) -> dict:
    """Day 2 task 2. Loads the training-only frame, tunes, writes outputs/models/*. Returns results."""
    cfg = config
    frame = load_tuning_frame(cfg, data_path)
    cv = tune_frame(frame, cfg)
    selected, ranking = select_candidate(cv, cfg)

    cv_path = _abs(cfg["tuning"]["cv_results_file"])
    sel_path = _abs(cfg["tuning"]["selected_params_file"])
    cv_path.parent.mkdir(parents=True, exist_ok=True)
    cv.to_csv(cv_path, index=False)
    meta = da.build_metadata(cv, cv_path.name, cfg, "src.experiments.run_tuning",
                             notes="Expanding-window CV inside 2015-2017 only; eval_mask rows.")
    da.write_sidecar(meta, cv_path)
    payload = {
        "selected_candidate_id": selected["candidate_id"],
        "selected_structural_params": {k: selected[k] for k in STRUCTURAL_KEYS},
        "full_params": merge_params(cfg, selected),
        "ranking": json.loads(ranking.to_json(orient="records")),
        "selection_metric": cfg["tuning"]["selection_metric"],
        "tie_break": cfg["tuning"]["tie_break"],
        "folds": cfg["tuning"]["folds"],
        "horizons": cfg["tuning"]["horizons"],
        "feature_set": cfg["tuning"]["feature_set"],
        "features": feature_set(cfg, cfg["tuning"]["feature_set"]),
        "cities": list(cfg["cities"][cfg["tuning"]["cities"]]),
        "tuning_frame": {"n_rows": int(len(frame)),
                         "min_target_date": str(frame["target_date"].min().date()),
                         "max_target_date": str(frame["target_date"].max().date())},
        "config_hash": config_hash(cfg),
        "dataset_sha256": meta["dataset_sha256"],
        "processed_dataset_sha256": da.sha256_file(_abs(data_path or cfg["paths"]["processed_dataset"])),
        "kaggle_version": cfg["data"]["kaggle_version"],
        "versions": {"lightgbm": lightgbm.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__},
        "seed": cfg["seed"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": meta["git_commit"],
    }
    sel_path.write_text(json.dumps(payload, indent=2, default=str))
    return {"frame": frame, "cv": cv, "ranking": ranking, "selected": selected,
            "payload": payload, "paths": (cv_path, sel_path)}
