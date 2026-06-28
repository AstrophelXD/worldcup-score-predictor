import pandas as pd

from worldcup.features.matchup_graph import build_matchup_graph
from worldcup.features.player_state import (
    build_team_player_tensors,
    lineup_entries_for_match,
    rolling_player_form,
)
from worldcup.utils.paths import project_root


def test_lineup_entries_for_final():
    lineups = pd.read_csv(project_root() / "data" / "samples" / "lineups.csv")
    lineups["team_id"] = lineups["team_id"].astype(str)
    as_of = pd.Timestamp("2022-12-18T15:00:00Z", tz="UTC")
    home = lineup_entries_for_match(lineups, "wc2022_arg_fra_final", "team_arg", as_of)
    away = lineup_entries_for_match(lineups, "wc2022_arg_fra_final", "team_fra", as_of)
    assert len(home) == 11
    assert len(away) == 11


def test_player_form_is_point_in_time():
    stats = pd.read_csv(project_root() / "data" / "samples" / "player_match_stats.csv")
    before = rolling_player_form(
        stats,
        "pla_messi",
        pd.Timestamp("2022-12-10T00:00:00Z", tz="UTC"),
    )
    after = rolling_player_form(
        stats,
        "pla_messi",
        pd.Timestamp("2022-12-20T00:00:00Z", tz="UTC"),
    )
    assert before.goals_last5 <= after.goals_last5


def test_build_matchup_graph_for_final():
    players = pd.read_csv(project_root() / "data" / "samples" / "players.csv")
    lineups = pd.read_csv(project_root() / "data" / "samples" / "lineups.csv")
    stats = pd.read_csv(project_root() / "data" / "samples" / "player_match_stats.csv")
    injuries = pd.read_csv(project_root() / "data" / "samples" / "injuries.csv")
    as_of = pd.Timestamp("2022-12-18T15:00:00Z", tz="UTC")
    home_players, home_mask, home_positions = build_team_player_tensors(
        match_id="wc2022_arg_fra_final",
        team_id="team_arg",
        as_of_time=as_of,
        player_slots=11,
        players=players,
        lineups=lineups,
        stats=stats,
        injuries=injuries,
    )
    away_players, away_mask, away_positions = build_team_player_tensors(
        match_id="wc2022_arg_fra_final",
        team_id="team_fra",
        as_of_time=as_of,
        player_slots=11,
        players=players,
        lineups=lineups,
        stats=stats,
        injuries=injuries,
    )
    assert home_mask.sum() == 11
    assert away_mask.sum() == 11
    _, _, _, edge_mask = build_matchup_graph(
        home_players,
        home_mask,
        home_positions,
        away_players,
        away_mask,
        away_positions,
    )
    assert edge_mask.sum() > 0
