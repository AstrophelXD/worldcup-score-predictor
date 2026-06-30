from datetime import date

from worldcup.dashboard.world_cup_2026_schedule import WORLD_CUP_2026_SCHEDULE
from worldcup.data_ingestion.sources.world_cup_2026_knockout import (
    build_knockout_context,
    resolve_match_teams,
)
from worldcup.data_ingestion.sources.world_cup_2026_catalog import (
    is_predictable_schedule_match,
    world_cup_2026_match_rows,
)


def _match(num: int):
    return next(m for m in WORLD_CUP_2026_SCHEDULE if m.match_number == num)


def test_round_of_32_third_place_slots_resolved_from_official_standings():
    ctx = build_knockout_context(today=date(2026, 6, 30))
    assert resolve_match_teams(_match(77), ctx) == ("France", "Sweden")
    assert resolve_match_teams(_match(79), ctx) == ("Mexico", "Ecuador")


def test_resolved_r32_matches_are_predictable_and_exported():
    assert is_predictable_schedule_match(_match(77))
    assert is_predictable_schedule_match(_match(79))
    rows = {row["match_id"]: row for row in world_cup_2026_match_rows()}
    assert rows["wc2026_m077"]["home_team_name"] == "France"
    assert rows["wc2026_m077"]["away_team_name"] == "Sweden"
    assert rows["wc2026_m079"]["home_team_name"] == "Mexico"
    assert rows["wc2026_m079"]["away_team_name"] == "Ecuador"
