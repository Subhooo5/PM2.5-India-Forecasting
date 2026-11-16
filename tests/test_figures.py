import numpy as np
import pytest

from src import figures


def test_negative_r2_extends_axis_below_zero():
    lo, hi = figures.metric_ylim("R²", [0.88, -0.35, 0.5], [0.02, None, None], [None, None, 0.3])
    assert lo < -0.35 and hi > 0.9 and lo < hi


def test_all_negative_r2_axis_includes_zero():
    lo, hi = figures.metric_ylim("R²", [-0.4, -0.1], [None, None], [])
    assert lo < -0.4 and hi >= 0


@pytest.mark.parametrize("metric", ["MAE", "RMSE", "R²"])
def test_nonnegative_values_use_zero_based_limits(metric):
    assert figures.metric_ylim(metric, [10.0, 20.0], [1.0, None], [25.0]) == (0.0, 25.0 * 1.18)


@pytest.mark.parametrize("seed", range(5))
def test_r2_limits_contain_every_bar_and_reference(seed):
    rng = np.random.default_rng(seed)
    v, e, r = rng.uniform(-1, 1, 5), rng.uniform(0, 0.2, 5), rng.uniform(-1, 1, 3)
    lo, hi = figures.metric_ylim("R²", v, e, r)
    assert lo <= min(min(v - e), min(r), 0) and hi >= max(max(v + e), max(r), 0)


def test_saved_figures_exist(dirs):
    names = {p.name for p in dirs["figures"].glob("*.png")}
    assert {"leakage_performance_comparison.png", "leakage_inflation.png", "cross_city_heatmap.png",
            "cross_city_model_comparison.png"} <= names and len(names) == 16
