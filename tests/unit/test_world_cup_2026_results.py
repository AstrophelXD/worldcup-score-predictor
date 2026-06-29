from worldcup.data_ingestion.sources.world_cup_2026_catalog import schedule_to_match_row
from worldcup.data_ingestion.sources.world_cup_2026_results import (
    apply_results_to_row,
    load_wc2026_results,
)
from worldcup.dashboard.world_cup_2026_schedule import WORLD_CUP_2026_SCHEDULE


def test_load_wc2026_results_includes_iraq_norway():
    results = load_wc2026_results()
    assert "wc2026_m018" in results
    assert results["wc2026_m018"]["home_score_ft"] == 1
    assert results["wc2026_m018"]["away_score_ft"] == 4


def test_generate_wc2026_results_through_june_28():
    from worldcup.data_ingestion.sources.world_cup_2026_results import generate_wc2026_results_through

    rows = generate_wc2026_results_through("2026-06-28")
    assert len(rows) == 73
    m018 = next(r for r in rows if r["match_id"] == "wc2026_m018")
    assert m018["home_score_ft"] == 1
    assert m018["away_score_ft"] == 4


def test_apply_results_to_schedule_row():
    match = next(m for m in WORLD_CUP_2026_SCHEDULE if m.match_number == 18)
    results = load_wc2026_results()
    row = schedule_to_match_row(match, results=results)
    assert row["home_score_ft"] == 1
    assert row["away_score_ft"] == 4
    assert row["status"] == "finished"


def test_apply_results_to_row_noop_when_missing():
    row = {"match_id": "wc2026_m999", "home_score_ft": "", "away_score_ft": ""}
    updated = apply_results_to_row(row, None)
    assert updated == row
