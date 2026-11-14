import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_audit as da  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return da.load_config()


@pytest.fixture(scope="session")
def raw(cfg):
    return da.load_city_day(cfg)


@pytest.fixture(scope="session")
def ds(raw, cfg):
    from src.features import build_dataset
    return build_dataset(raw, cfg)


@pytest.fixture(scope="session")
def raw_lookup(raw):
    """Raw PM2.5 keyed by (City, Date) for exact alignment checks."""
    return raw.set_index(["City", "Date"])["PM2.5"]


@pytest.fixture(scope="session")
def same_day(raw, cfg):
    from src.features import build_same_day_dataset
    return build_same_day_dataset(raw, cfg)
