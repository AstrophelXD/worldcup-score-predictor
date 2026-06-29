"""Load external bookmaker odds (separate from model-implied odds.csv)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from worldcup.features.point_in_time import filter_as_of
from worldcup.inference.market_odds import implied_probs_from_odds_row

MARKET_SUMMARY_COLUMNS = [
    "market_home_implied",
    "market_draw_implied",
    "market_away_implied",
    "market_over25_implied",
    "market_under25_implied",
    "market_btts_implied",
    "market_odds_available",
    "market_odds_source",
]


def load_market_odds_table(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        from worldcup.utils.paths import project_root

        root = project_root()
        candidates = [
            root / "data" / "samples" / "market_odds.csv",
            root / "data" / "curated" / "market_odds.parquet",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
    if path is None or not path.exists():
        return pd.DataFrame()

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_market_odds_summary_row(
    market_df: pd.DataFrame,
    match_id: str,
    as_of_time: pd.Timestamp,
) -> dict[str, float | str]:
    empty = {column: 0.0 for column in MARKET_SUMMARY_COLUMNS if column != "market_odds_source"}
    empty["market_odds_source"] = ""
    if market_df.empty:
        return empty

    scoped = market_df.loc[market_df["match_id"].astype(str) == str(match_id)]
    if scoped.empty:
        return empty

    scoped = filter_as_of(scoped, as_of_time, "snapshot_ts")
    if scoped.empty:
        return empty

    row = scoped.sort_values("snapshot_ts").iloc[-1].to_dict()
    implied = implied_probs_from_odds_row(row)
    return {
        "market_home_implied": float(implied["result_probs"]["home_win"]),
        "market_draw_implied": float(implied["result_probs"]["draw"]),
        "market_away_implied": float(implied["result_probs"]["away_win"]),
        "market_over25_implied": float(implied["ou25_probs"]["over_2_5"]),
        "market_under25_implied": float(implied["ou25_probs"]["under_2_5"]),
        "market_btts_implied": float(implied["btts_probs"]["yes"]),
        "market_odds_available": 1.0,
        "market_odds_source": str(row.get("source") or "market"),
    }
