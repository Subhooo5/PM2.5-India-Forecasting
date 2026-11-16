import inspect
from typing import Any

import numpy as np
import pytest

from src.conformal import aci_stream, conformal_quantile, conformity_scores, cqr_interval


def test_conformity_scores_below_inside_above():
    s = conformity_scores([1.0, 5.0, 12.0], [2.0, 2.0, 2.0], [10.0, 10.0, 10.0])
    np.testing.assert_array_equal(s, [1.0, -3.0, 2.0])


def test_conformity_scores_reject_non_finite():
    with pytest.raises(ValueError):
        conformity_scores([np.nan, 1.0], [0.0, 0.0], [2.0, 2.0])


@pytest.mark.parametrize("scores, alpha, expected", [
    (np.arange(1, 10), 0.1, 9.0),
    (np.arange(1, 20), 0.1, 18.0),
    (np.arange(1, 100), 0.1, 90.0),
    (np.arange(1, 1800), 0.1, 1620.0),
    (np.arange(1, 6), 0.1, np.inf),
    ([3.0, 1.0, 3.0, 3.0, 1.0], 0.5, 3.0),
])
def test_conformal_quantile_order_statistic(scores, alpha, expected):
    assert conformal_quantile(scores, alpha) == expected


@pytest.mark.parametrize("q_hat, lo, hi", [(2.0, -2.0, 12.0), (-3.0, 3.0, 7.0), (-6.0, 6.0, 4.0)])
def test_cqr_interval_no_clipping_or_swapping(q_hat, lo, hi):
    assert tuple(map(float, cqr_interval(0.0, 10.0, q_hat))) == (lo, hi)


def test_cqr_interval_keeps_negative_lower_bound():
    assert tuple(map(float, cqr_interval(1.0, 5.0, 3.0))) == (-2.0, 8.0)


def test_conformal_api_takes_no_test_labels():
    assert list(inspect.signature(cqr_interval).parameters) == ["lo", "hi", "q_hat"]


D = np.datetime64("2019-01-01", "ns")
DAY = np.timedelta64(1, "D")
CAL = np.arange(1.0, 20.0)
NAT = np.datetime64("NaT", "ns")


def _aci(h, y, gamma=0.5, cal=CAL, bounds=(0.05, 0.30)):
    n = len(y)
    origin = D + np.arange(n) * DAY
    return aci_stream(origin, origin + h * DAY, y, np.zeros(n), np.zeros(n), cal, 0.10, gamma,
                      bounds), origin


HAND: dict[int, dict[str, Any]] = {
    1: dict(y=[0.0, 0.0, 100.0, 0.0], alpha=[0.10, 0.15, 0.20, 0.05], hi=[18, 17, 16, 19],
            miss=[0, 0, 1, 0], upd=[1, 2, 3, None], after=[0.15, 0.20, 0.05, np.nan]),
    3: dict(y=[0.0, 100.0, 0.0, 0.0, 0.0, 0.0], alpha=[0.10, 0.10, 0.10, 0.15, 0.05, 0.10],
            hi=[18, 18, 18, 17, 19, 18], miss=[0, 1, 0, 0, 0, 0], upd=[3, 4, 5, None, None, None],
            after=[0.15, 0.05, 0.10, np.nan, np.nan, np.nan]),
}


@pytest.mark.parametrize("h", [1, 3])
@pytest.mark.parametrize("key", ["alpha", "hi", "miss", "upd", "after"])
def test_aci_hand_trajectory(h, key):
    case = HAND[h]
    out, origin = _aci(h, case["y"])
    got = {"alpha": out["alpha_issue"], "hi": out["hi"], "miss": out["miss"], "after": out["alpha_after"],
           "upd": out["update_origin"]}[key]
    if key == "upd":
        exp = np.array([origin[i] if i is not None else NAT for i in case["upd"]])
        np.testing.assert_array_equal(got, exp)
    else:
        np.testing.assert_allclose(np.asarray(got, dtype=float), np.asarray(case[key], dtype=float),
                                   rtol=0, atol=1e-12)


def test_aci_always_covered_climbs_to_upper_bound():
    out, _ = _aci(1, np.zeros(200), gamma=0.02, cal=np.arange(1.0, 1000.0), bounds=(0.01, 0.30))
    assert out["alpha_issue"].max() == 0.30 and (out["alpha_issue"][-50:] == 0.30).all()


def test_aci_always_missed_falls_to_lower_bound():
    out, _ = _aci(1, np.full(200, 1e6), gamma=0.02, cal=np.arange(1.0, 1000.0), bounds=(0.01, 0.30))
    assert out["alpha_issue"].min() == 0.01 and (out["alpha_issue"][-50:] == 0.01).all()


@pytest.mark.parametrize("h", [1, 3, 7])
def test_aci_updates_only_after_maturity(h):
    out, origin = _aci(h, np.random.default_rng(h).normal(0, 20, 60), gamma=0.02)
    upd = out["update_origin"]
    ok = ~np.isnat(upd)
    assert (upd[ok] >= origin[ok] + h * DAY).all() and (upd[ok] > origin[ok]).all()


@pytest.mark.parametrize("h", [1, 3, 7])
def test_aci_issue_reflects_updates_matured_at_same_origin(h):
    out, origin = _aci(h, np.random.default_rng(h).normal(0, 20, 60), gamma=0.02)
    for i, t in enumerate(origin):
        matured = np.flatnonzero(out["update_origin"] == t)
        if len(matured):
            assert out["alpha_issue"][i] == out["alpha_after"][matured[-1]]


@pytest.mark.parametrize("h", [1, 3, 7])
@pytest.mark.parametrize("k", [0, 10, 30])
def test_aci_no_future_outcome_used(h, k):
    y = np.random.default_rng(k).normal(0, 20, 50)
    out, origin = _aci(h, y, gamma=0.02)
    future = origin + h * DAY > origin[k]
    y2 = np.where(future, y + 1e3, y)
    out2, _ = _aci(h, y2, gamma=0.02)
    for key in ("alpha_issue", "lo", "hi"):
        np.testing.assert_array_equal(out2[key][: k + 1], out[key][: k + 1])


def test_aci_unobserved_target_matures_without_update():
    out, _ = _aci(1, [0.0, np.nan, 0.0])
    assert np.isnat(out["update_origin"][1]) and out["alpha_issue"][2] == out["alpha_issue"][1]


O = D + np.arange(3) * DAY
VALID: dict[str, Any] = dict(origin=O, target=O + DAY, y=[0.0] * 3, lo=[0.0] * 3, hi=[1.0] * 3,
             cal_scores=np.arange(1.0, 101.0), alpha=0.10, gamma=0.02, bounds=(0.01, 0.30))
BAD: dict[str, dict[str, Any]] = {"alpha_0": dict(alpha=0.0), "alpha_1": dict(alpha=1.0), "gamma_0": dict(gamma=0.0),
       "gamma_neg": dict(gamma=-0.1), "bounds_unordered": dict(bounds=(0.30, 0.01)),
       "alpha_outside_bounds": dict(alpha=0.5), "nat": dict(origin=np.array([O[0], NAT, O[2]])),
       "origins_not_increasing": dict(origin=O[[0, 2, 1]], target=O[[0, 2, 1]] + 5 * DAY),
       "targets_not_increasing": dict(target=O[[2, 1, 0]] + 5 * DAY), "target_before_origin": dict(target=O),
       "lengths": dict(y=[0.0] * 2), "cal_too_short": dict(cal_scores=np.arange(1.0, 99.0)),
       "cal_nonfinite": dict(cal_scores=np.r_[np.arange(1.0, 100.0), np.nan]),
       "y_inf": dict(y=[0.0, np.inf, 0.0]), "y_neg_inf": dict(y=[0.0, -np.inf, 0.0]),
       "lo_inf": dict(lo=[0.0, -np.inf, 0.0]), "lo_nan": dict(lo=[0.0, np.nan, 0.0]),
       "hi_inf": dict(hi=[1.0, np.inf, 1.0]), "hi_nan": dict(hi=[1.0, np.nan, 1.0]),
       "y_2d": dict(y=np.zeros((3, 1))), "lo_2d": dict(lo=np.zeros((3, 1))),
       "gamma_inf": dict(gamma=np.inf), "gamma_nan": dict(gamma=np.nan)}


@pytest.mark.parametrize("case", list(BAD))
def test_aci_rejects_invalid_input(case):
    with pytest.raises(ValueError):
        aci_stream(**{**VALID, **BAD[case]})


def test_aci_frozen_config_passes():
    assert np.isfinite(aci_stream(**VALID)["hi"]).all()


@pytest.mark.parametrize("scores, alpha", [(np.arange(5.0), 0.0), (np.arange(5.0), 1.0), (np.arange(5.0), -0.1),
                                           ([], 0.1), (np.ones((2, 2)), 0.1), ([1.0, np.inf], 0.1)])
def test_conformal_quantile_rejects_invalid_input(scores, alpha):
    with pytest.raises(ValueError):
        conformal_quantile(scores, alpha)


def test_aci_accepts_nan_y_as_unobserved():
    out = aci_stream(**{**VALID, "y": [0.0, np.nan, 0.0]})
    assert np.isnan(out["miss"][1]) and np.isnat(out["update_origin"][1])


def test_aci_returns_expected_keys():
    assert list(aci_stream(**VALID)) == ["alpha_issue", "lo", "hi", "miss", "update_origin", "alpha_after"]
