# Leakage-Free, Uncertainty-Aware PM2.5 Forecasting for Indian Cities

Survey paper plus a small benchmark study: one reproducible daily PM2.5 forecasting pipeline
(h = 1, 3, 7 days) and three controlled experiments: leakage (RQ1), cross-city transfer (RQ2)
and calibrated uncertainty (RQ3). Project rules, frozen design and progress live in **CLAUDE.md**.

## Data
Kaggle "Air Quality Data in India (2015-2020)" by Vopani, version 12, CC0 (from CPCB).
Only `city_day.csv` is used; place it at `data/raw/city_day.csv` (sha256 is pinned in
`configs/experiments.yaml`). `data/raw/` is read-only and never committed.

## Setup (macOS, Python 3.11)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # LightGBM needs OpenMP: brew install libomp
python -m ipykernel install --user --name pm25 --display-name "Python (pm25)"
```

## Run
```bash
python -m src.data_audit   # verify raw data + write completeness/gap tables to outputs/tables/
pytest -q                  # executable verification
```
Every output table has a `<name>.meta.json` sidecar with the dataset sha256 and Kaggle version.
