import pandas as pd

from worldcup.data_ingestion.sources.transformers import transform_statsbomb_team_match
from worldcup.features.event_features import (
    EVENT_SUMMARY_COLUMNS,
    build_event_summary_row,
    event_targets_for_match,
)
from worldcup.utils.paths import project_root


def test_transform_statsbomb_team_match():
    df = pd.DataFrame(
        [
            {
                "match_id": "wc2022_arg_fra_final",
                "team_name": "Argentina",
                "match_date": "2022-12-18",
                "xg": 1.9,
                "shots": 14,
                "shots_on_target": 6,
                "yellow_cards": 2,
            }
        ]
    )
    out = transform_statsbomb_team_match(df)
    assert out.iloc[0]["match_id"] == "wc2022_arg_fra_final"
    assert out.iloc[0]["xg"] == 1.9
    assert out.iloc[0]["cards"] == 2


def test_event_targets_for_match():
    root = project_root()
    team_stats = pd.read_csv(root / "data" / "samples" / "team_match_stats.csv")
    if team_stats.empty:
        return
    targets, available = event_targets_for_match(
        team_stats,
        "wc2022_arg_fra_final",
        "team_arg",
        "team_fra",
    )
    assert available
    assert len(targets) == 4
    assert targets[0] > 0


def test_build_event_summary_row():
    root = project_root()
    curated_matches = root / "data" / "curated" / "matches.parquet"
    team_stats = pd.read_csv(root / "data" / "samples" / "team_match_stats.csv")
    if team_stats.empty or not curated_matches.exists():
        return
    matches = pd.read_parquet(curated_matches)
    summary = build_event_summary_row(
        match_id="wc2022_arg_fra_final",
        home_team_id="team_arg",
        away_team_id="team_fra",
        as_of_time=pd.Timestamp("2022-12-18T15:00:00Z"),
        team_stats=team_stats,
        matches=matches,
    )
    assert all(column in summary for column in EVENT_SUMMARY_COLUMNS)
    assert summary["event_data_available"] == 1.0
