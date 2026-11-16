import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import metrics as MET


def test_point_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.gamma(2, 40, 500), rng.gamma(2, 40, 500)
    assert MET.mae(y, p) == pytest.approx(mean_absolute_error(y, p), rel=1e-12)
    assert MET.rmse(y, p) == pytest.approx(np.sqrt(mean_squared_error(y, p)), rel=1e-12)
    assert MET.r2(y, p) == pytest.approx(r2_score(y, p), rel=1e-12)
    assert MET.point_metrics(y, p)["n"] == 500


def test_point_metrics_edge_cases():
    empty = MET.point_metrics([], [])
    assert empty["n"] == 0 and all(np.isnan(empty[k]) for k in ("mae", "rmse", "r2"))
    const = MET.point_metrics([5.0, 5.0, 5.0], [4.0, 5.0, 6.0])
    assert np.isnan(const["r2"]) and const["mae"] == pytest.approx(2 / 3)
    m = MET.point_metrics([1.0, np.nan, 3.0], [2.0, np.inf, 3.0])
    assert m["n"] == 2 and m["mae"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        MET.point_metrics([1.0, 2.0], [1.0])


@pytest.mark.parametrize("y, p, match", [([1.0, np.inf], [1.0, 2.0], "infinite target"),
                                         ([1.0, -np.inf], [1.0, 2.0], "infinite target"),
                                         ([1.0, 2.0], [1.0, np.nan], "prediction"),
                                         ([1.0, 2.0], [1.0, np.inf], "prediction")])
def test_point_metrics_finite_rule(y, p, match):
    with pytest.raises(ValueError, match=match):
        MET.point_metrics(y, p)


@pytest.mark.parametrize("ref", [0.0, np.nan, np.inf, -np.inf])
def test_ratio_helpers_return_nan_without_warning(ref):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(MET.skill_score(5.0, ref)) and np.isnan(MET.inflation_pct(5.0, ref))
        arr = np.asarray(MET.relative_change_pct(np.array([5.0, 5.0]), np.array([ref, 4.0])))
        assert np.isnan(arr[0]) and arr[1] == 100 * (5.0 / 4.0 - 1)


def test_ratio_helpers_hand_computed():
    rng = np.random.default_rng(1)
    a, b = rng.gamma(2, 10, 200), rng.gamma(2, 10, 200)
    for x, y in zip(a, b):
        assert MET.skill_score(x, y) == 1 - x / y and MET.inflation_pct(x, y) == float(100.0 * (1.0 - x / y))
        assert MET.relative_change_pct(x, y) == 100 * (x / y - 1)
    assert np.array_equal(MET.relative_change_pct(pd.Series(a), pd.Series(b)), (100 * (a / b - 1)))
    assert (MET.skill_score(8.0, 10.0), MET.inflation_pct(8.0, 10.0)) == pytest.approx((0.2, 20.0))


CASE: dict[str, Any] = dict(y_true=[5.0, 0.0, 20.0, 7.0, np.nan], lower=[2.0, 2.0, 2.0, 10.0, np.nan],
                            upper=[10.0, 10.0, 10.0, 4.0, np.nan], alpha=0.1)


def test_interval_metrics_hand_computed():
    m = MET.interval_metrics(**CASE)
    assert list(m) == ["picp", "mpiw", "winkler", "n"]
    assert (m["picp"], m["mpiw"], m["winkler"], m["n"]) == pytest.approx((0.25, 4.5, 79.5, 4), abs=1e-12)


@pytest.mark.parametrize("y, lo", [([1.0], [np.nan]), ([1.0], [np.inf]), ([np.inf], [0.0])])
def test_interval_metrics_finite_rule(y, lo):
    with pytest.raises(ValueError):
        MET.interval_metrics(y, lo, [2.0], 0.1)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5])
def test_interval_metrics_rejects_alpha(alpha):
    with pytest.raises(ValueError):
        MET.interval_metrics([1.0], [0.0], [2.0], alpha)


def test_interval_metrics_empty():
    m = MET.interval_metrics([np.nan], [1.0], [2.0], 0.1)
    assert m["n"] == 0 and np.isnan([m["picp"], m["mpiw"], m["winkler"]]).all()
