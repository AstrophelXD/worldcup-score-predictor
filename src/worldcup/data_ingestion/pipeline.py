from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from worldcup.data_ingestion.base import read_parquet, write_parquet
from worldcup.data_ingestion.csv_loader import load_csv_to_raw
from worldcup.data_ingestion.curated.matches import build_matches_curated, build_teams_curated
from worldcup.data_ingestion.curated.players import (
    build_injuries_curated,
    build_lineups_curated,
    build_player_match_stats_curated,
    build_players_curated,
)
from worldcup.data_ingestion.curated.ratings import build_elo_curated, build_fifa_rankings_curated
from worldcup.data_ingestion.team_resolver import TeamResolver
from worldcup.utils.paths import ensure_dir


@dataclass
class IngestResult:
    raw_paths: dict[str, Path]
    curated_paths: dict[str, Path]
    team_count: int
    match_count: int
    elo_count: int
    fifa_count: int
    player_count: int = 0
    lineup_count: int = 0
    player_stat_count: int = 0
    injury_count: int = 0


def run_ingest(
    *,
    raw_dir: Path,
    curated_dir: Path,
    mappings_dir: Path,
    matches_csv: Path | None,
    elo_csv: Path | None,
    fifa_csv: Path | None,
    players_csv: Path | None = None,
    lineups_csv: Path | None = None,
    player_stats_csv: Path | None = None,
    injuries_csv: Path | None = None,
    source_systems: dict[str, str],
) -> IngestResult:
    ensure_dir(raw_dir)
    ensure_dir(curated_dir)

    resolver = TeamResolver.from_csv(mappings_dir / "team_aliases.csv")
    raw_paths: dict[str, Path] = {}
    curated_paths: dict[str, Path] = {}

    matches_curated = pd.DataFrame()
    teams_curated = pd.DataFrame()
    if matches_csv and matches_csv.exists():
        raw_paths["matches"] = load_csv_to_raw(
            matches_csv,
            raw_dir,
            source_systems.get("matches", "matches_csv"),
            "matches",
        )
        matches_raw = read_parquet(str(raw_paths["matches"]))
        matches_curated = build_matches_curated(matches_raw, resolver)
        teams_curated = build_teams_curated(matches_curated, resolver)
        curated_paths["matches"] = curated_dir / "matches.parquet"
        curated_paths["teams"] = curated_dir / "teams.parquet"
        write_parquet(matches_curated, str(curated_paths["matches"]))
        write_parquet(teams_curated, str(curated_paths["teams"]))

    elo_curated = pd.DataFrame()
    if elo_csv and elo_csv.exists():
        raw_paths["elo"] = load_csv_to_raw(
            elo_csv,
            raw_dir,
            source_systems.get("elo", "elo_csv"),
            "elo",
        )
        elo_raw = read_parquet(str(raw_paths["elo"]))
        elo_curated = build_elo_curated(elo_raw, resolver)
        curated_paths["elo_ratings"] = curated_dir / "elo_ratings.parquet"
        write_parquet(elo_curated, str(curated_paths["elo_ratings"]))

    fifa_curated = pd.DataFrame()
    if fifa_csv and fifa_csv.exists():
        raw_paths["fifa_rankings"] = load_csv_to_raw(
            fifa_csv,
            raw_dir,
            source_systems.get("fifa_rankings", "fifa_csv"),
            "fifa_rankings",
        )
        fifa_raw = read_parquet(str(raw_paths["fifa_rankings"]))
        fifa_curated = build_fifa_rankings_curated(fifa_raw, resolver)
        curated_paths["fifa_rankings"] = curated_dir / "fifa_rankings.parquet"
        write_parquet(fifa_curated, str(curated_paths["fifa_rankings"]))

    players_curated = pd.DataFrame()
    if players_csv and players_csv.exists():
        raw_paths["players"] = load_csv_to_raw(
            players_csv,
            raw_dir,
            source_systems.get("players", "players_csv"),
            "players",
        )
        players_raw = read_parquet(str(raw_paths["players"]))
        players_curated = build_players_curated(players_raw, resolver)
        curated_paths["players"] = curated_dir / "players.parquet"
        write_parquet(players_curated, str(curated_paths["players"]))

    lineups_curated = pd.DataFrame()
    if lineups_csv and lineups_csv.exists():
        raw_paths["lineups"] = load_csv_to_raw(
            lineups_csv,
            raw_dir,
            source_systems.get("lineups", "lineups_csv"),
            "lineups",
        )
        lineups_raw = read_parquet(str(raw_paths["lineups"]))
        lineups_curated = build_lineups_curated(lineups_raw, resolver)
        curated_paths["lineups"] = curated_dir / "lineups.parquet"
        write_parquet(lineups_curated, str(curated_paths["lineups"]))

    player_stats_curated = pd.DataFrame()
    if player_stats_csv and player_stats_csv.exists():
        raw_paths["player_match_stats"] = load_csv_to_raw(
            player_stats_csv,
            raw_dir,
            source_systems.get("player_match_stats", "player_stats_csv"),
            "player_match_stats",
        )
        stats_raw = read_parquet(str(raw_paths["player_match_stats"]))
        player_stats_curated = build_player_match_stats_curated(stats_raw, resolver)
        curated_paths["player_match_stats"] = curated_dir / "player_match_stats.parquet"
        write_parquet(player_stats_curated, str(curated_paths["player_match_stats"]))

    injuries_curated = pd.DataFrame()
    if injuries_csv and injuries_csv.exists():
        raw_paths["injuries"] = load_csv_to_raw(
            injuries_csv,
            raw_dir,
            source_systems.get("injuries", "injuries_csv"),
            "injuries",
        )
        injuries_raw = read_parquet(str(raw_paths["injuries"]))
        injuries_curated = build_injuries_curated(injuries_raw, resolver)
        curated_paths["injuries"] = curated_dir / "injuries.parquet"
        write_parquet(injuries_curated, str(curated_paths["injuries"]))

    return IngestResult(
        raw_paths=raw_paths,
        curated_paths=curated_paths,
        team_count=len(teams_curated),
        match_count=len(matches_curated),
        elo_count=len(elo_curated),
        fifa_count=len(fifa_curated),
        player_count=len(players_curated),
        lineup_count=len(lineups_curated),
        player_stat_count=len(player_stats_curated),
        injury_count=len(injuries_curated),
    )
