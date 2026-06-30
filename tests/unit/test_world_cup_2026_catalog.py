from worldcup.dashboard.world_cup_2026_schedule import WORLD_CUP_2026_SCHEDULE
from worldcup.data_ingestion.sources.world_cup_2026_catalog import (
    is_predictable_schedule_match,
    wc2026_match_id,
    world_cup_2026_match_rows,
)


def test_world_cup_2026_export_rows():
    rows = world_cup_2026_match_rows()
    assert len(rows) >= 72
    assert all(row["status"] in {"scheduled", "finished"} for row in rows)
    assert all(row["is_world_cup"] for row in rows)
    assert rows[0]["match_id"] == "wc2026_m001"


def test_wc2026_match_ids_are_unique():
    ids = [wc2026_match_id(m) for m in WORLD_CUP_2026_SCHEDULE]
    assert len(ids) == len(set(ids))


def test_predictable_matches_include_group_stage():
    predictable = [m for m in WORLD_CUP_2026_SCHEDULE if is_predictable_schedule_match(m)]
    group = [m for m in predictable if m.stage_name == "Group stage"]
    assert len(group) == 72


def test_germany_round_of_32_opponent_is_paraguay():
    germany = next(m for m in WORLD_CUP_2026_SCHEDULE if m.match_number == 74)
    assert germany.home_team == "Germany"
    assert germany.away_team == "Paraguay"
    assert is_predictable_schedule_match(germany)
