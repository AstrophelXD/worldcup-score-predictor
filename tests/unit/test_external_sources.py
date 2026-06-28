import pandas as pd
import pytest

from worldcup.data_ingestion.sources.elo_from_matches import compute_elo_from_matches
from worldcup.data_ingestion.sources.prepare import prepare_external_sources
from worldcup.data_ingestion.sources.transformers import (
    transform_elo_history,
    transform_eloratings_world_tsv,
    transform_fifa_rankings,
    transform_football_data_odds,
    transform_kaggle_international,
)
from worldcup.utils.paths import project_root


@pytest.fixture
def examples_dir():
    return project_root() / "data" / "external" / "examples"


def test_transform_kaggle_international(examples_dir):
    df = pd.read_csv(examples_dir / "kaggle_results.csv")
    out = transform_kaggle_international(df)
    assert len(out) == len(df)
    assert out["is_world_cup"].sum() >= 2
    assert "match_id" in out.columns


def test_transform_elo_history(examples_dir):
    df = pd.read_csv(examples_dir / "elo_history.csv")
    out = transform_elo_history(df)
    assert len(out) == len(df)
    assert out.iloc[0]["team_name"] == "Brazil"


def test_transform_fifa_rankings(examples_dir):
    df = pd.read_csv(examples_dir / "fifa_rankings_history.csv")
    out = transform_fifa_rankings(df)
    assert len(out) == len(df)
    assert "points" in out.columns
    assert out["rank"].notna().all()


def test_compute_elo_from_matches(examples_dir):
    df = pd.read_csv(examples_dir / "kaggle_results.csv")
    matches = transform_kaggle_international(df)
    elo = compute_elo_from_matches(matches)
    assert len(elo) >= len(matches) * 2
    assert elo.iloc[0]["rating_system"] == "elo_match_derived"


def test_transform_eloratings_world_tsv():
    raw = pd.DataFrame([[1, 1, "Brazil", 2000.0, 0, 0, 0, 0]])
    out = transform_eloratings_world_tsv(raw)
    assert out.iloc[0]["team_name"] == "Brazil"
    assert out.iloc[0]["rating"] == 2000.0


def test_transform_football_data_odds():
    df = pd.DataFrame(
        [
            {
                "Date": "01/01/2020",
                "HomeTeam": "Brazil",
                "AwayTeam": "Argentina",
                "B365H": 2.1,
                "B365D": 3.2,
                "B365A": 3.5,
            }
        ]
    )
    out = transform_football_data_odds(df)
    assert len(out) == 1
    assert out.iloc[0]["home_odds"] == 2.1


def test_prepare_external_sources_merges_samples():
    import shutil

    staging_dir = project_root() / "tests" / "tmp" / "prepare_external"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    result = prepare_external_sources(
        staging_dir=staging_dir,
        match_sources=[
            {
                "path": str(project_root() / "data/external/examples/kaggle_results.csv"),
                "format": "kaggle_international",
            }
        ],
        elo_sources=[
            {
                "path": str(project_root() / "data/external/examples/elo_history.csv"),
                "format": "elo_history",
            }
        ],
        fifa_sources=[
            {
                "path": str(project_root() / "data/external/examples/fifa_rankings_history.csv"),
                "format": "fifa_rankings",
            }
        ],
        include_samples=True,
        samples_dir=project_root() / "data" / "samples",
    )
    assert result.matches and result.matches.exists()
    matches = pd.read_csv(result.matches)
    assert len(matches) > 5
def test_prepare_external_sources_includes_player_tables():
    import shutil

    staging_dir = project_root() / "tests" / "tmp" / "prepare_player"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    result = prepare_external_sources(
        staging_dir=staging_dir,
        match_sources=[],
        elo_sources=[],
        fifa_sources=[],
        include_samples=True,
        samples_dir=project_root() / "data" / "samples",
    )
    assert result.players and result.players.exists()
    assert result.lineups and result.lineups.exists()
    assert result.injuries and result.injuries.exists()
    injuries = pd.read_csv(result.injuries)
    assert len(injuries) >= 1
    shutil.rmtree(staging_dir, ignore_errors=True)
