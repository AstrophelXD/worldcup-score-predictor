from __future__ import annotations

import pandas as pd


def build_odds_curated(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for record in raw_df.to_dict(orient="records"):
        rows.append(
            {
                "match_id": str(record["match_id"]),
                "snapshot_ts": str(record["snapshot_ts"]),
                "home_odds": float(record["home_odds"]),
                "draw_odds": float(record["draw_odds"]),
                "away_odds": float(record["away_odds"]),
                "over25_odds": float(record.get("over25_odds", 2.05)),
                "under25_odds": float(record.get("under25_odds", 1.78)),
                "btts_yes_odds": float(record.get("btts_yes_odds", 1.90)),
            }
        )
    return pd.DataFrame(rows)
