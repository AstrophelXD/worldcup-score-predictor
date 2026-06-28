import pandas as pd

from worldcup.features.builder import build_match_feature_mart
from worldcup.features.player_match_features import (
    PLAYER_SUMMARY_COLUMNS,
    build_player_summary_row,
)
from worldcup.utils.paths import project_root


def test_build_player_summary_row():
    root = project_root()
    players = pd.read_csv(root / "data" / "samples" / "players.csv")
    lineups = pd.read_csv(root / "data" / "samples" / "lineups.csv")
    stats = pd.read_csv(root / "data" / "samples" / "player_match_stats.csv")
    injuries = pd.read_csv(root / "data" / "samples" / "injuries.csv")

    summary = build_player_summary_row(
        match_id="wc2022_arg_fra_final",
        home_team_id="team_arg",
        away_team_id="team_fra",
        as_of_time=pd.Timestamp("2022-12-18T15:00:00Z"),
        players=players,
        lineups=lineups,
        stats=stats,
        injuries=injuries,
    )
    assert summary["home_starter_count"] == 11.0
    assert summary["away_starter_count"] == 11.0
    assert all(column in summary for column in PLAYER_SUMMARY_COLUMNS)


def test_feature_mart_includes_player_and_odds_columns():
    import shutil

    from worldcup.data_ingestion.pipeline import run_ingest

    root = project_root()
    tmp_root = root / "tests" / "tmp" / "feature_mart_p1"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)
    curated_dir = tmp_root / "curated"
    feature_dir = tmp_root / "feature_mart"

    run_ingest(
        raw_dir=tmp_root / "raw",
        curated_dir=curated_dir,
        mappings_dir=root / "data" / "external_mappings",
        matches_csv=root / "data" / "samples" / "matches.csv",
        elo_csv=root / "data" / "samples" / "elo.csv",
        fifa_csv=root / "data" / "samples" / "fifa_rankings.csv",
        players_csv=root / "data" / "samples" / "players.csv",
        lineups_csv=root / "data" / "samples" / "lineups.csv",
        player_stats_csv=root / "data" / "samples" / "player_match_stats.csv",
        injuries_csv=root / "data" / "samples" / "injuries.csv",
        odds_csv=root / "data" / "samples" / "odds.csv",
        team_match_stats_csv=root / "data" / "samples" / "team_match_stats.csv",
        source_systems={"matches": "t", "elo": "t", "fifa_rankings": "t", "odds": "t"},
    )

    result = build_match_feature_mart(
        curated_dir=curated_dir,
        feature_mart_dir=feature_dir,
        form_windows=[3, 5],
    )
    features = pd.read_parquet(result.output_path)
    assert "home_starter_count" in features.columns
    assert "odds_available" in features.columns
    wc = features.loc[features["match_id"] == "wc2022_arg_fra_final"].iloc[0]
    assert wc["home_starter_count"] >= 8
    assert wc["odds_available"] == 1.0
    shutil.rmtree(tmp_root, ignore_errors=True)
