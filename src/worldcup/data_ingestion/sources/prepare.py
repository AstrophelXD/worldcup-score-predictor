from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from worldcup.data_ingestion.sources.transformers import (
    transform_elo_history,
    transform_fifa_rankings,
    transform_kaggle_international,
)
from worldcup.utils.paths import ensure_dir


class SourceFormat(StrEnum):
    KAGGLE_INTERNATIONAL = "kaggle_international"
    ELO_HISTORY = "elo_history"
    FIFA_RANKINGS = "fifa_rankings"
    CANONICAL = "canonical"


TRANSFORMERS = {
    SourceFormat.KAGGLE_INTERNATIONAL: transform_kaggle_international,
    SourceFormat.ELO_HISTORY: transform_elo_history,
    SourceFormat.FIFA_RANKINGS: transform_fifa_rankings,
}


@dataclass
class PreparedPaths:
    matches: Path | None
    elo: Path | None
    fifa_rankings: Path | None


def _read_source(path: Path, fmt: SourceFormat) -> pd.DataFrame:
    if fmt == SourceFormat.CANONICAL:
        return pd.read_csv(path)
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
    sort_cols = [col for col in ["kickoff_ts", "match_id"] if col in df.columns]
    ordered = df.sort_values(sort_cols) if sort_cols else df
    return ordered.drop_duplicates(subset=["match_id"], keep="last").reset_index(drop=True)


def prepare_external_sources(
    *,
    staging_dir: Path,
    match_sources: list[dict],
    elo_sources: list[dict],
    fifa_sources: list[dict],
    include_samples: bool,
    samples_dir: Path,
) -> PreparedPaths:
    ensure_dir(staging_dir)
    canonical_dir = staging_dir / "canonical"
    ensure_dir(canonical_dir)

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
        elo_df = _align_concat(elo_frames).drop_duplicates(
            subset=["team_name", "rating_date"], keep="last"
        )
        elo_path = canonical_dir / "elo.csv"
        elo_df.to_csv(elo_path, index=False)

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
        fifa_df = _align_concat(fifa_frames).drop_duplicates(
            subset=["team_name", "ranking_date"], keep="last"
        )
        fifa_path = canonical_dir / "fifa_rankings.csv"
        fifa_df.to_csv(fifa_path, index=False)

    return PreparedPaths(matches=matches_path, elo=elo_path, fifa_rankings=fifa_path)
