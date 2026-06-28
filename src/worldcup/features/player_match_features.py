"""PIT player / injury summary features for the unified match feature mart."""

from __future__ import annotations

import pandas as pd

from worldcup.features.player_state import (
    build_team_player_tensors,
    lineup_entries_for_match,
    player_availability,
)

PLAYER_SUMMARY_COLUMNS = [
    "home_starter_count",
    "away_starter_count",
    "home_avg_player_rating",
    "away_avg_player_rating",
    "home_squad_availability",
    "away_squad_availability",
    "home_injured_out_count",
    "away_injured_out_count",
    "home_avg_form_goals",
    "away_avg_form_goals",
    "home_lineup_projected_share",
    "away_lineup_projected_share",
]


def _team_player_summary(
    *,
    match_id: str,
    team_id: str,
    as_of_time: pd.Timestamp,
    players: pd.DataFrame,
    lineups: pd.DataFrame,
    stats: pd.DataFrame,
    injuries: pd.DataFrame,
    player_slots: int = 11,
) -> dict[str, float]:
    tensors, mask, _ = build_team_player_tensors(
        match_id=match_id,
        team_id=team_id,
        as_of_time=as_of_time,
        player_slots=player_slots,
        players=players,
        lineups=lineups,
        stats=stats,
        injuries=injuries,
    )
    starter_count = float(mask.sum())

    if mask.any():
        active = tensors[mask]
        avg_rating = float(active[:, 0].mean() * 100.0)
        avg_availability = float(active[:, 6].mean())
        avg_form_goals = float(active[:, 2].mean() * 5.0)
    else:
        avg_rating = 0.0
        avg_availability = 0.0
        avg_form_goals = 0.0

    entries = lineup_entries_for_match(lineups, match_id, team_id, as_of_time)
    projected_share = 0.0
    if not entries.empty:
        projected_share = float(
            entries["lineup_status"].str.lower().eq("projected").mean()
        )

    injured_out = 0.0
    if not injuries.empty and not players.empty:
        team_players = players.loc[players["national_team_id"] == team_id]
        for player_id in team_players["player_id"]:
            availability = player_availability(injuries, str(player_id), team_id, as_of_time)
            if availability <= 0.0:
                injured_out += 1.0

    return {
        "starter_count": starter_count,
        "avg_player_rating": avg_rating,
        "squad_availability": avg_availability,
        "injured_out_count": injured_out,
        "avg_form_goals": avg_form_goals,
        "lineup_projected_share": projected_share,
    }


def build_player_summary_row(
    *,
    match_id: str,
    home_team_id: str,
    away_team_id: str,
    as_of_time: pd.Timestamp,
    players: pd.DataFrame,
    lineups: pd.DataFrame,
    stats: pd.DataFrame,
    injuries: pd.DataFrame,
) -> dict[str, float]:
    home = _team_player_summary(
        match_id=match_id,
        team_id=home_team_id,
        as_of_time=as_of_time,
        players=players,
        lineups=lineups,
        stats=stats,
        injuries=injuries,
    )
    away = _team_player_summary(
        match_id=match_id,
        team_id=away_team_id,
        as_of_time=as_of_time,
        players=players,
        lineups=lineups,
        stats=stats,
        injuries=injuries,
    )
    return {
        "home_starter_count": home["starter_count"],
        "away_starter_count": away["starter_count"],
        "home_avg_player_rating": home["avg_player_rating"],
        "away_avg_player_rating": away["avg_player_rating"],
        "home_squad_availability": home["squad_availability"],
        "away_squad_availability": away["squad_availability"],
        "home_injured_out_count": home["injured_out_count"],
        "away_injured_out_count": away["injured_out_count"],
        "home_avg_form_goals": home["avg_form_goals"],
        "away_avg_form_goals": away["avg_form_goals"],
        "home_lineup_projected_share": home["lineup_projected_share"],
        "away_lineup_projected_share": away["lineup_projected_share"],
    }
