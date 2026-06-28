from __future__ import annotations

import pandas as pd

from worldcup.data_ingestion.team_resolver import TeamResolver


def _nullable_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _resolve_team_id(resolver: TeamResolver, raw: str) -> str:
    value = str(raw).strip()
    if value.startswith("team_"):
        return value
    return resolver.resolve(value)


def build_players_curated(raw_df: pd.DataFrame, resolver: TeamResolver) -> pd.DataFrame:
    rows: list[dict] = []
    for record in raw_df.to_dict(orient="records"):
        team_raw = record.get("national_team_id") or record.get("national_team_name")
        team_id = _resolve_team_id(resolver, str(team_raw)) if team_raw else None
        rows.append(
            {
                "player_id": str(record["player_id"]),
                "full_name": str(record["full_name"]).strip(),
                "national_team_id": team_id,
                "primary_position": record.get("primary_position"),
                "market_value_eur": _nullable_float(record.get("market_value_eur")),
                "player_rating": _nullable_float(record.get("player_rating")),
                "source_system": record["source_system"],
                "source_record_id": record.get("source_record_id"),
                "ingested_at": record["ingested_at"],
                "updated_at": record["updated_at"],
            }
        )
    return pd.DataFrame(rows)


def build_lineups_curated(raw_df: pd.DataFrame, resolver: TeamResolver) -> pd.DataFrame:
    rows: list[dict] = []
    for record in raw_df.to_dict(orient="records"):
        team_id = _resolve_team_id(resolver, str(record["team_id"]))
        rows.append(
            {
                "lineup_id": str(record["lineup_id"]),
                "match_id": str(record["match_id"]),
                "team_id": team_id,
                "player_id": str(record["player_id"]),
                "is_starting": bool(record.get("is_starting", True)),
                "bench_order": record.get("bench_order"),
                "position_code": record.get("position_code"),
                "formation_slot": record.get("formation_slot"),
                "lineup_status": str(record.get("lineup_status", "historical")),
                "projection_prob": _nullable_float(record.get("projection_prob")),
                "source_system": record["source_system"],
                "source_record_id": record.get("source_record_id"),
                "ingested_at": record["ingested_at"],
                "updated_at": record["updated_at"],
            }
        )
    return pd.DataFrame(rows)


def build_player_match_stats_curated(raw_df: pd.DataFrame, resolver: TeamResolver) -> pd.DataFrame:
    rows: list[dict] = []
    for record in raw_df.to_dict(orient="records"):
        team_id = _resolve_team_id(resolver, str(record["team_id"]))
        rows.append(
            {
                "stat_id": str(record["stat_id"]),
                "match_id": str(record["match_id"]),
                "player_id": str(record["player_id"]),
                "team_id": team_id,
                "match_date": pd.to_datetime(record["match_date"]).date(),
                "minutes_played": int(record.get("minutes_played", 0)),
                "goals": int(record.get("goals", 0)),
                "assists": int(record.get("assists", 0)),
                "source_system": record["source_system"],
                "source_record_id": record.get("source_record_id"),
                "ingested_at": record["ingested_at"],
                "updated_at": record["updated_at"],
            }
        )
    return pd.DataFrame(rows)
