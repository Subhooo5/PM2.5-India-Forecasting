from __future__ import annotations

import sys
import time

from src import cross_city, data_audit, experiments, features, leakage, stress, uncertainty


def make_figures(cfg: dict) -> list:
    return [p for module in (leakage, cross_city, uncertainty, stress) for p in module.make_figures(cfg)]


STAGES = {
    "audit": data_audit.run_audit,
    "preprocess": features.run_preprocess,
    "features": features.run_features,
    "tuning": experiments.run_tuning,
    "leakage": leakage.run,
    "cross_city": cross_city.run,
    "uncertainty": uncertainty.run,
    "stress": stress.run,
    "figures": make_figures,
}
USAGE = f"usage: python run.py [all | {' | '.join(STAGES)}]"


def main(argv: list[str], cfg: dict | None = None) -> list[str]:
    names = list(STAGES) if argv in ([], ["all"]) else argv
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; {USAGE}")
    cfg = cfg or data_audit.load_config()
    for name in names:
        start = time.time()
        STAGES[name](cfg)
        print(f"{name}: done in {time.time() - start:.0f} s", flush=True)
    if names == list(STAGES):
        stress.write_manifest(cfg)
        print("manifest: outputs/result_manifest.csv", flush=True)
    return names


if __name__ == "__main__":
    main(sys.argv[1:])
