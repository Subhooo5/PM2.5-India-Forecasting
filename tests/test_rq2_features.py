"""RQ2 feature consistency: common city-ID-free set, explicit unseen-city handling, LOCO isolation."""
import pandas as pd
import pytest

from src import features as F
from src.splits import loco_partition


def test_causal_common_matches_strict_features_minus_city(cfg):
    assert F.feature_set(cfg, "causal_common") == [c for c in F.strict_feature_columns(cfg) if c != "city"]
    assert F.feature_set(cfg, "causal_city_aware") == F.feature_set(cfg, "causal_common") + ["city"]


def test_causal_sets_exclude_forbidden_and_same_day_is_isolated(cfg, ds):
    forbidden = set(cfg["features"]["forbidden"])
    for name in ("causal_common", "causal_city_aware"):
        assert set(F.feature_set(cfg, name)).isdisjoint(forbidden)
    assert "city" not in F.feature_set(cfg, "causal_common")
    same_day_only = set(F.feature_set(cfg, "same_day_estimation")) - set(cfg["features"]["calendar"])
    assert same_day_only <= forbidden                 # A/B inputs are exactly the forbidden ones
    assert same_day_only.isdisjoint(ds.columns)       # and never enter the forecasting table
    assert set(F.feature_set(cfg, "causal_city_aware")) <= set(ds.columns)


@pytest.mark.parametrize("held_out", ["Delhi", "Lucknow", "Chennai", "Hyderabad", "Bengaluru"])
def test_loco_never_trains_on_held_out_city(ds, cfg, held_out):
    cols = F.feature_set(cfg, "causal_common")
    train, test = loco_partition(ds, held_out, cfg, cols)
    assert held_out not in set(train["city"].astype(str))
    assert set(train["city"].astype(str)) == set(cfg["cities"]["core"]) - {held_out}
    assert set(test["city"].astype(str)) == {held_out}
    assert (train["split_main"] == "train").all() and (test["split_main"] == "test").all()
    assert test["eval_mask"].all()
    assert not set(train["city"].astype(str)) & set(cfg["cities"]["transfer"] + cfg["cities"]["transfer_optional"])


def test_loco_refuses_city_aware_features(ds, cfg):
    with pytest.raises(ValueError, match="city-ID-free"):
        loco_partition(ds, "Delhi", cfg, F.feature_set(cfg, "causal_city_aware"))


def test_encode_city_marks_unseen_cities_missing(ds, cfg):
    train_cities = [c for c in cfg["cities"]["core"] if c != "Delhi"]
    enc = F.encode_city(ds, train_cities)
    assert list(enc.cat.categories) == sorted(train_cities)
    unseen = ~ds["city"].astype(str).isin(train_cities)          # Delhi + transfer cities
    assert enc[unseen].isna().all() and (enc.cat.codes[unseen] == -1).all()
    assert enc[~unseen].notna().all()
    pd.testing.assert_series_equal(enc[~unseen].astype(str), ds.loc[~unseen, "city"].astype(str), check_names=False)
