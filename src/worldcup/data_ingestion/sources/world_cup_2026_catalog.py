"""Convert dashboard 2026 schedule into ingest-ready match records."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from worldcup.dashboard.flags import is_known_team
from worldcup.dashboard.world_cup_2026_schedule import (
    WORLD_CUP_2026_SCHEDULE,
    ScheduleMatch,
)

ET = ZoneInfo("America/New_York")
KNOCKOUT_STAGES = {
    "Round of 32",
    "Round of 16",
    "Quarter-finals",
    "Semi-finals",
    "Third place",
    "Final",
}


def wc2026_match_id(match: ScheduleMatch) -> str:
    return f"wc2026_m{match.match_number:03d}"


def et_to_utc_iso(match_date: str, kickoff_et: str) -> str:
    hour, minute = map(int, kickoff_et.split(":"))
    day = datetime.strptime(match_date, "%Y-%m-%d")
    if hour >= 24:
        hour -= 24
        day += timedelta(days=1)
    local = day.replace(hour=hour, minute=minute, tzinfo=ET)
    return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_predictable_schedule_match(match: ScheduleMatch) -> bool:
    return is_known_team(match.home_team) and is_known_team(match.away_team)


def schedule_to_match_row(match: ScheduleMatch) -> dict:
    is_ko = match.stage_name in KNOCKOUT_STAGES
    return {
        "match_id": wc2026_match_id(match),
        "competition_name": "FIFA World Cup",
        "season_name": "2026",
        "stage_name": match.stage_name,
        "match_date": match.match_date,
        "kickoff_ts": et_to_utc_iso(match.match_date, match.kickoff_et),
        "home_team_name": match.home_team,
        "away_team_name": match.away_team,
        "home_score_ft": "",
        "away_score_ft": "",
        "home_score_ht": "",
        "away_score_ht": "",
        "aet_score_home": "",
        "aet_score_away": "",
        "pen_score_home": "",
        "pen_score_away": "",
        "status": "scheduled",
        "is_world_cup": True,
        "is_knockout": is_ko,
        "venue": match.venue,
        "city": match.city,
        "country": "USA/Canada/Mexico",
    }


def world_cup_2026_match_rows(*, predictable_only: bool = True) -> list[dict]:
    rows: list[dict] = []
    for match in WORLD_CUP_2026_SCHEDULE:
        if predictable_only and not is_predictable_schedule_match(match):
            continue
        rows.append(schedule_to_match_row(match))
    return rows
