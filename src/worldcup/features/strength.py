from __future__ import annotations

import pandas as pd


def latest_rating_before(
    ratings_df: pd.DataFrame,
    team_id: str,
    as_of_time: pd.Timestamp,
    date_col: str,
    value_cols: list[str],
) -> dict[str, float | int | None]:
    team_rows = ratings_df.loc[ratings_df["team_id"] == team_id].copy()
    if team_rows.empty:
        return {col: None for col in value_cols}

    dates = pd.to_datetime(team_rows[date_col]).dt.date
    cutoff_date = as_of_time.date()
    eligible = team_rows.loc[dates <= cutoff_date]
    if eligible.empty:
        return {col: None for col in value_cols}

    sort_dates = dates.loc[eligible.index]
    latest = eligible.assign(_sort_date=sort_dates).sort_values("_sort_date").iloc[-1]
    return {col: latest[col] for col in value_cols}
