from __future__ import annotations

import pandas as pd

ROLES = ("core", "transfer", "transfer_optional")


def select_cities(cfg: dict) -> dict[str, str]:
    roles = {}
    for role in ROLES:
        for city in cfg["cities"][role]:
            if city in roles:
                raise ValueError(f"{city} listed under two roles")
            roles[city] = role
    return roles


def city_daily_series(df: pd.DataFrame, city: str, cfg: dict) -> pd.Series:
    days = pd.date_range(cfg["data"]["date_min"], cfg["data"]["date_max"], freq="D", name="date")
    s = df.loc[df["City"] == city].set_index("Date")[cfg["data"]["target"]]
    if s.empty:
        raise ValueError(f"city {city!r} not in data")
    if s.index.has_duplicates:
        raise ValueError(f"duplicate dates for {city}")
    return s.reindex(days).astype("float64").rename("pm25")


def ffill_inputs(s: pd.Series, limit: int) -> pd.Series:
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("ffill limit must be a non-negative int")
    return s.ffill(limit=limit) if limit > 0 else s.copy()
