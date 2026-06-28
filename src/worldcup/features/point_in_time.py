from __future__ import annotations

from datetime import datetime

import pandas as pd


def as_of_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def filter_as_of(df: pd.DataFrame, as_of_time: datetime, time_col: str) -> pd.DataFrame:
    """Return rows at or before the prediction cutoff."""
    ts = pd.to_datetime(df[time_col], utc=True)
    cutoff = as_of_timestamp(as_of_time)
    return df.loc[ts <= cutoff].copy()


def filter_before(df: pd.DataFrame, as_of_time: datetime, time_col: str) -> pd.DataFrame:
    """Return rows strictly before the prediction cutoff (for rolling history)."""
    ts = pd.to_datetime(df[time_col], utc=True)
    cutoff = as_of_timestamp(as_of_time)
    return df.loc[ts < cutoff].copy()
