from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from worldcup.data_ingestion.sources.transformers import (
    transform_elo_history,
    transform_eloratings_world_tsv,
    transform_fifa_rankings,
    transform_football_data_odds,
    transform_kaggle_international,
)
from worldcup.utils.paths import ensure_dir


class SourceFormat(StrEnum):
    KAGGLE_INTERNATIONAL = "kaggle_international"
    ELO_HISTORY = "elo_history"
    ELO_RATINGS_WORLD_TSV = "eloratings_world_tsv"
    FIFA_RANKINGS = "fifa_rankings"
    FOOTBALL_DATA_ODDS = "football_data_odds"
    CANONICAL = "canonical"


TRANSFORMERS = {
    SourceFormat.KAGGLE_INTERNATIONAL: transform_kaggle_international,
    SourceFormat.ELO_HISTORY: transform_elo_history,
    SourceFormat.ELO_RATINGS_WORLD_TSV: transform_eloratings_world_tsv,
    SourceFormat.FIFA_RANKINGS: transform_fifa_rankings,
    SourceFormat.FOOTBALL_DATA_ODDS: transform_football_data_odds,
}


@dataclass
class PreparedPaths:
    matches: Path | None
    elo: Path | None
    fifa_rankings: Path | None
    players: Path | None = None
    lineups: Path | None = None
    player_match_stats: Path | None = None
    injuries: Path | None = None
    odds: Path | None = None


def _read_source(path: Path, fmt: SourceFormat) -> pd.DataFrame:
    if fmt == SourceFormat.ELO_RATINGS_WORLD_TSV:
        return pd.read_csv(path, sep="\t", header=None)
    return pd.read_csv(path)


def _transform(path: Path, fmt: SourceFormat) -> pd.DataFrame:
    df = _read_source(path, fmt)
    if fmt == SourceFormat.CANONICAL:
        return df
    transformer = TRANSFORMERS.get(fmt)
    if transformer is None:
        raise ValueError(f"unsupported source format: {fmt}")
    return transformer(df)


def _align_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    columns = sorted(set().union(*(frame.columns for frame in frames)))
    aligned = [frame.reindex(columns=columns) for frame in frames]
    return pd.concat(aligned, ignore_index=True)


def _dedupe_matches(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_cols = [col for col in ["is_world_cup", "kickoff_ts", "match_id"] if col in df.columns]
    ordered = df.sort_values(sort_cols, ascending=[False, True, True]) if sort_cols else df
    subset = ["match_id"]
    if {"match_date", "home_team_name", "away_team_name"}.issubset(ordered.columns):
        ordered = ordered.drop_duplicates(
            subset=["match_date", "home_team_name", "away_team_name"],
            keep="first",
        )
    return ordered.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def _write_canonical(frames: list[pd.DataFrame], path: Path, subset: list[str] | None) -> Path:
    df = _align_concat(frames)
    if subset and not df.empty:
        df = df.drop_duplicates(subset=subset, keep="last")
    df.to_csv(path, index=False)
    return path


def prepare_external_sources(
    *,
    staging_dir: Path,
    match_sources: list[dict],
    elo_sources: list[dict],
    fifa_sources: list[dict],
    player_sources: list[dict] | None = None,
    lineup_sources: list[dict] | None = None,
    player_stat_sources: list[dict] | None = None,
    injury_sources: list[dict] | None = None,
    odds_sources: list[dict] | None = None,
    include_samples: bool,
    samples_dir: Path,
) -> PreparedPaths:
    ensure_dir(staging_dir)
    canonical_dir = staging_dir / "canonical"
    ensure_dir(canonical_dir)

    player_sources = player_sources or []
    lineup_sources = lineup_sources or []
    player_stat_sources = player_stat_sources or []
    injury_sources = injury_sources or []
    odds_sources = odds_sources or []

    match_frames: list[pd.DataFrame] = []
    if include_samples:
        sample_matches = samples_dir / "matches.csv"
        if sample_matches.exists():
            match_frames.append(_transform(sample_matches, SourceFormat.CANONICAL))

    for item in match_sources:
        path = Path(str(item["path"]))
        fmt = SourceFormat(str(item["format"]))
        if not path.exists():
            raise FileNotFoundError(f"match source not found: {path}")
        match_frames.append(_transform(path, fmt))

    match_frames = [frame for frame in match_frames if not frame.empty]
    matches_path = None
    if match_frames:
        matches_df = _dedupe_matches(_align_concat(match_frames))
        matches_path = canonical_dir / "matches.csv"
        matches_df.to_csv(matches_path, index=False)

    elo_frames: list[pd.DataFrame] = []
    if include_samples:
        sample_elo = samples_dir / "elo.csv"
        if sample_elo.exists():
            elo_frames.append(_transform(sample_elo, SourceFormat.CANONICAL))
    for item in elo_sources:
        path = Path(str(item["path"]))
        fmt = SourceFormat(str(item["format"]))
        if not path.exists():
            raise FileNotFoundError(f"elo source not found: {path}")
        elo_frames.append(_transform(path, fmt))
    elo_frames = [frame for frame in elo_frames if not frame.empty]
    elo_path = None
    if elo_frames:
        elo_path = _write_canonical(
            elo_frames,
            canonical_dir / "elo.csv",
            ["team_name", "rating_date"],
        )

    fifa_frames: list[pd.DataFrame] = []
    if include_samples:
        sample_fifa = samples_dir / "fifa_rankings.csv"
        if sample_fifa.exists():
            fifa_frames.append(_transform(sample_fifa, SourceFormat.CANONICAL))
    for item in fifa_sources:
        path = Path(str(item["path"]))
        fmt = SourceFormat(str(item["format"]))
        if not path.exists():
            raise FileNotFoundError(f"fifa source not found: {path}")
        fifa_frames.append(_transform(path, fmt))
    fifa_frames = [frame for frame in fifa_frames if not frame.empty]
    fifa_path = None
    if fifa_frames:
        fifa_path = _write_canonical(
            fifa_frames,
            canonical_dir / "fifa_rankings.csv",
            ["team_name", "ranking_date"],
        )

    def _prepare_player_table(
        sample_name: str,
        sources: list[dict],
        output_name: str,
        dedupe_cols: list[str],
    ) -> Path | None:
        frames: list[pd.DataFrame] = []
        sample_path = samples_dir / sample_name
        if include_samples and sample_path.exists():
            frames.append(_transform(sample_path, SourceFormat.CANONICAL))
        for item in sources:
            path = Path(str(item["path"]))
            fmt = SourceFormat(str(item["format"]))
            if not path.exists():
                raise FileNotFoundError(f"{output_name} source not found: {path}")
            frames.append(_transform(path, fmt))
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return None
        out = canonical_dir / output_name
        return _write_canonical(frames, out, dedupe_cols)

    players_path = _prepare_player_table(
        "players.csv", player_sources, "players.csv", ["player_id"]
    )
    lineups_path = _prepare_player_table(
        "lineups.csv", lineup_sources, "lineups.csv", ["lineup_id"]
    )
    stats_path = _prepare_player_table(
        "player_match_stats.csv",
        player_stat_sources,
        "player_match_stats.csv",
        ["stat_id"],
    )
    injuries_path = _prepare_player_table(
        "injuries.csv", injury_sources, "injuries.csv", ["injury_id"]
    )
    odds_path = _prepare_player_table("odds.csv", odds_sources, "odds.csv", ["match_id", "snapshot_ts"])

    return PreparedPaths(
        matches=matches_path,
        elo=elo_path,
        fifa_rankings=fifa_path,
        players=players_path,
        lineups=lineups_path,
        player_match_stats=stats_path,
        injuries=injuries_path,
        odds=odds_path,
    )
