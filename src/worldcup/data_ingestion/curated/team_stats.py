from __future__ import annotations

import pandas as pd

from worldcup.data_ingestion.team_resolver import TeamResolver


def _nullable_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _nullable_int(value: object, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    return int(value)


def _resolve_team_id(resolver: TeamResolver, raw: str) -> str:
    value = str(raw).strip()
    if value.startswith("team_"):
        return value
    return resolver.resolve(value)


def build_team_match_stats_curated(
    raw_df: pd.DataFrame,
    resolver: TeamResolver,
) -> pd.DataFrame:
    rows: list[dict] = []
    for record in raw_df.to_dict(orient="records"):
        team_id = _resolve_team_id(resolver, str(record["team_id"]))
        rows.append(
            {
                "team_match_stat_id": str(record["team_match_stat_id"]),
                "match_id": str(record["match_id"]),
                "team_id": team_id,
                "match_date": pd.to_datetime(record["match_date"]).date(),
                "possession": _nullable_float(record.get("possession")),
                "shots": _nullable_int(record.get("shots")),
                "shots_on_target": _nullable_int(record.get("shots_on_target")),
                "xg": _nullable_float(record.get("xg")),
                "passes_completed": _nullable_int(record.get("passes_completed")),
                "corners": _nullable_int(record.get("corners")),
                "fouls": _nullable_int(record.get("fouls")),
                "yellow_cards": _nullable_int(record.get("yellow_cards")),
                "red_cards": _nullable_int(record.get("red_cards")),
                "cards": _nullable_int(record.get("cards")),
                "source_system": record["source_system"],
                "source_record_id": record.get("source_record_id"),
                "ingested_at": record["ingested_at"],
                "updated_at": record["updated_at"],
            }
        )
    return pd.DataFrame(rows)
