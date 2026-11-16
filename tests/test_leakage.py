import numpy as np
import pandas as pd
import pytest

from src import leakage as L
from src import leakage_protocols as P
from src.metrics import inflation_pct, point_metrics


@pytest.fixture(scope="module")
def saved(cfg, dirs):
    return P.load_saved(cfg, dirs)


@pytest.fixture(scope="module")
def table(dirs):
    return pd.read_csv(dirs["tables"] / P.FILES["comparison"], keep_default_na=False, na_values=[""])


def test_comparison_rows_use_the_inflation_formula(table):
    rows = table[~table["seed"].astype(str).str.startswith("5-seed")]
    for col, left, right in (("rel_inflation_mae_pct", "mae_left", "mae_right"),
                             ("rel_inflation_rmse_pct", "rmse_left", "rmse_right")):
        np.testing.assert_allclose(rows[col], inflation_pct(rows[left].to_numpy(), rows[right].to_numpy()), atol=1e-9)
    bc = table[table["scope"] == "primary_common_city_dates"]
    assert bc["n_left"].tolist() == bc["n_right"].tolist() == [1825, 1825, 1825]
    assert table["notes"].str.contains("NOT a pure split effect|two DIFFERENT tasks").all()


def test_b_vs_c_common_keys_match_set_logic(saved):
    B, C = saved["B"], saved["C"]
    for h in (1, 3, 7):
        c = C[(C["method"] == "lightgbm") & (C["horizon"] == h) & C["eval_mask"]]
        expected = set(zip(B["city"], B["target_date"])) & set(zip(c["city"], c["target_date"]))
        assert set(L.bc_common_keys(B, C, h)) == expected


def test_matched_a_vs_b_rows(saved, table):
    A = saved["A"]
    A19 = A[A["target_date"].dt.year == 2019]
    B_m = L.rows_on(saved["B"], pd.MultiIndex.from_frame(A19[["city", "target_date"]]))
    row = table[(table["scope"] == "c_matched_2019_city_dates") & (table["seed"].astype(str) == "42")].iloc[0]
    assert row["n_left"] == row["n_right"] == len(B_m) == len(A19)
    assert row["mae_right"] == pytest.approx(point_metrics(B_m["y_true"], B_m["y_pred"])["mae"], abs=1e-12)


def test_preselected_diagnostic_city_has_lowest_c_h1_r2(saved):
    c = saved["C"][(saved["C"]["method"] == "lightgbm") & (saved["C"]["horizon"] == 1) & saved["C"]["eval_mask"]]
    r2 = {city: point_metrics(g["y_true"], g["y_pred"])["r2"] for city, g in c.groupby("city")}
    assert min(r2, key=lambda k: r2[k]) == L.DIAG_FOCUS_CITY
