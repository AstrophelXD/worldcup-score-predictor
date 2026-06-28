from __future__ import annotations

import pandas as pd

from worldcup.data_ingestion.team_resolver import TeamResolver


def _nullable_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if pd.isna(value):
        return None
    return int(value)


def build_matches_curated(raw_df: pd.DataFrame, resolver: TeamResolver) -> pd.DataFrame:
    rows: list[dict] = []
    for record in raw_df.to_dict(orient="records"):
        home_name = str(record["home_team_name"])
        away_name = str(record["away_team_name"])
        rows.append(
            {
                "match_id": record["match_id"],
                "competition_name": record["competition_name"],
                "season_name": record.get("season_name"),
                "stage_name": record.get("stage_name"),
                "match_date": pd.to_datetime(record["match_date"]).date(),
                "kickoff_ts": pd.to_datetime(record["kickoff_ts"], utc=True),
                "venue": record.get("venue"),
                "city": record.get("city"),
                "country": record.get("country"),
                "home_team_id": resolver.resolve(home_name),
                "away_team_id": resolver.resolve(away_name),
                "home_score_ft": _nullable_int(record.get("home_score_ft")),
                "away_score_ft": _nullable_int(record.get("away_score_ft")),
                "home_score_ht": _nullable_int(record.get("home_score_ht")),
                "away_score_ht": _nullable_int(record.get("away_score_ht")),
                "aet_score_home": _nullable_int(record.get("aet_score_home")),
                "aet_score_away": _nullable_int(record.get("aet_score_away")),
                "pen_score_home": _nullable_int(record.get("pen_score_home")),
                "pen_score_away": _nullable_int(record.get("pen_score_away")),
                "status": record.get("status", "finished"),
                "is_world_cup": bool(record.get("is_world_cup", False)),
                "is_knockout": bool(record.get("is_knockout", False)),
                "source_system": record["source_system"],
                "source_record_id": record.get("source_record_id"),
                "ingested_at": record["ingested_at"],
                "updated_at": record["updated_at"],
            }
        )
    return pd.DataFrame(rows)


def build_teams_curated(matches_df: pd.DataFrame, resolver: TeamResolver) -> pd.DataFrame:
    team_ids = set(matches_df["home_team_id"]) | set(matches_df["away_team_id"])
    now = matches_df["ingested_at"].iloc[0] if len(matches_df) else pd.Timestamp.utcnow()
    rows = [
        {
            "team_id": team_id,
            "team_name": resolver.team_name(team_id),
            "country_code": None,
            "confederation": None,
            "fifa_team_id": None,
            "statsbomb_team_id": None,
            "is_national_team": True,
            "source_system": "curated_from_matches",
            "source_record_id": team_id,
            "ingested_at": now,
            "updated_at": now,
        }
        for team_id in sorted(team_ids)
    ]
    return pd.DataFrame(rows)
