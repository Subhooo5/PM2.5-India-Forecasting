"""Missing-value handling for INPUT features (Day 1).

Frozen / confirmed rules (see CLAUDE.md decisions log):
- Each city's PM2.5 is placed on the full daily calendar; NaN means "not observed".
- Forward-fill at most `preprocessing.ffill_limit` (= 2) days, on historical LAG inputs only.
- Never fill the origin value y_t or the target. Never backfill or interpolate.
- Rolling statistics use raw observed values (no fill).
"""
from __future__ import annotations

import pandas as pd


def select_cities(cfg: dict) -> dict[str, str]:
    """Day 1 item 4: {city: role} for every city that enters the processed dataset."""
    roles = {}
    for role in ("core", "transfer", "transfer_optional"):
        for city in cfg["cities"][role]:
            if city in roles:
                raise ValueError(f"{city} listed under two roles")
            roles[city] = role
    return roles


def city_daily_series(df: pd.DataFrame, city: str, cfg: dict) -> pd.Series:
    """Raw PM2.5 of one city on the full daily calendar (date_min..date_max). Never filled."""
    days = pd.date_range(cfg["data"]["date_min"], cfg["data"]["date_max"], freq="D", name="date")
    s = df.loc[df["City"] == city].set_index("Date")[cfg["data"]["target"]]
    if s.empty:
        raise ValueError(f"city {city!r} not in data")
    if s.index.has_duplicates:
        raise ValueError(f"duplicate dates for {city}")
    return s.reindex(days).astype("float64").rename("pm25")


def ffill_inputs(s: pd.Series, limit: int) -> pd.Series:
    """Forward-fill an INPUT series for at most `limit` consecutive days (causal: uses past only)."""
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("ffill limit must be a non-negative int")
    return s.ffill(limit=limit) if limit > 0 else s.copy()
