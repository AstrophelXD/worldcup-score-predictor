"""Market odds summary columns for the unified match feature mart."""

from __future__ import annotations

import pandas as pd

from worldcup.features.point_in_time import filter_before
from worldcup.features.scoregen import odds_features_for_match

ODDS_SUMMARY_COLUMNS = [
    "odds_home_implied",
    "odds_draw_implied",
    "odds_away_implied",
    "odds_over25_implied",
    "odds_under25_implied",
    "odds_btts_implied",
    "odds_available",
]


def build_odds_summary_row(
    odds_df: pd.DataFrame,
    match_id: str,
    as_of_time: pd.Timestamp,
) -> dict[str, float]:
    odds, available = odds_features_for_match(odds_df, match_id, as_of_time)
    if not available:
        return {column: 0.0 for column in ODDS_SUMMARY_COLUMNS}

    return {
        "odds_home_implied": float(odds[0]),
        "odds_draw_implied": float(odds[1]),
        "odds_away_implied": float(odds[2]),
        "odds_over25_implied": float(odds[3]),
        "odds_under25_implied": float(odds[4]),
        "odds_btts_implied": float(odds[5]),
        "odds_available": 1.0,
    }


def load_odds_curated(curated_dir) -> pd.DataFrame:
    path = curated_dir / "odds.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
