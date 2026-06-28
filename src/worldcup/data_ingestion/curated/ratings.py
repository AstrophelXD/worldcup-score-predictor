from __future__ import annotations

import pandas as pd

from worldcup.data_ingestion.team_resolver import TeamResolver


def build_elo_curated(raw_df: pd.DataFrame, resolver: TeamResolver) -> pd.DataFrame:
    rows: list[dict] = []
    for idx, record in enumerate(raw_df.to_dict(orient="records")):
        team_id = resolver.resolve(str(record["team_name"]))
        rating_date = pd.to_datetime(record["rating_date"]).date()
        rows.append(
            {
                "elo_id": f"elo_{team_id}_{rating_date.isoformat()}",
                "team_id": team_id,
                "rating": float(record["rating"]),
                "rating_date": rating_date,
                "rating_system": record.get("rating_system", "elo"),
                "rank": int(record["rank"]) if pd.notna(record.get("rank")) else None,
                "source_system": record["source_system"],
                "source_record_id": record.get("source_record_id") or str(idx),
                "ingested_at": record["ingested_at"],
                "updated_at": record["updated_at"],
            }
        )
    return pd.DataFrame(rows)


def build_fifa_rankings_curated(raw_df: pd.DataFrame, resolver: TeamResolver) -> pd.DataFrame:
    rows: list[dict] = []
    for idx, record in enumerate(raw_df.to_dict(orient="records")):
        team_id = resolver.resolve(str(record["team_name"]))
        ranking_date = pd.to_datetime(record["ranking_date"]).date()
        rows.append(
            {
                "fifa_ranking_id": f"fifa_{team_id}_{ranking_date.isoformat()}",
                "team_id": team_id,
                "ranking_date": ranking_date,
                "rank": int(record["rank"]),
                "points": float(record["points"]),
                "source_system": record["source_system"],
                "source_record_id": record.get("source_record_id") or str(idx),
                "ingested_at": record["ingested_at"],
                "updated_at": record["updated_at"],
            }
        )
    return pd.DataFrame(rows)
