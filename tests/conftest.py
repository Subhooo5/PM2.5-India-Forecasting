import pytest

from src import data_audit as da
from src import intervals, uncertainty
from src.experiments import forecast_frames
from src.features import build_dataset, build_same_day_dataset


@pytest.fixture(scope="session")
def cfg():
    return da.load_config()


@pytest.fixture(scope="session")
def raw(cfg):
    return da.load_city_day(cfg)


@pytest.fixture(scope="session")
def ds(raw, cfg):
    return build_dataset(raw, cfg)


@pytest.fixture(scope="session")
def same_day(raw, cfg):
    return build_same_day_dataset(raw, cfg)


@pytest.fixture(scope="session")
def raw_lookup(raw):
    return raw.set_index(["City", "Date"])["PM2.5"]


@pytest.fixture(scope="session")
def dirs(cfg):
    return da.output_dirs(cfg)


@pytest.fixture(scope="session")
def frames(cfg):
    return forecast_frames(cfg)


@pytest.fixture
def tmp_dirs(tmp_path):
    return {k: tmp_path / k for k in ("models", "predictions", "tables", "figures")}


@pytest.fixture(scope="session")
def unc_run(cfg, dirs, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("unc")
    out_dirs = {"models": dirs["models"], "predictions": tmp / "predictions", "tables": tmp / "tables"}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(intervals, "fit_lgbm", lambda *a, **k: pytest.fail("a saved quantile model was refit"))
        res = uncertainty.run(cfg, out_dirs)
    return res, out_dirs

