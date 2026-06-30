"""Resolve 2026 World Cup knockout placeholders from group-stage results."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date

from worldcup.dashboard.flags import is_known_team
from worldcup.dashboard.world_cup_2026_schedule import (
    WORLD_CUP_2026_GROUPS,
    WORLD_CUP_2026_SCHEDULE,
    ScheduleMatch,
)
from worldcup.data_ingestion.sources.annex_c import third_group_for_winner
from worldcup.data_ingestion.sources.world_cup_2026_results import load_wc2026_results

# Final group-stage tables (June 27, 2026) — authoritative once groups are done.
OFFICIAL_GROUP_STANDINGS: dict[str, tuple[str, str, str]] = {
    "A": ("Mexico", "South Africa", "Korea Republic"),
    "B": ("Switzerland", "Canada", "Bosnia and Herzegovina"),
    "C": ("Brazil", "Morocco", "Scotland"),
    "D": ("United States", "Australia", "Paraguay"),
    "E": ("Germany", "Ivory Coast", "Ecuador"),
    "F": ("Netherlands", "Japan", "Sweden"),
    "G": ("Belgium", "Egypt", "Iran"),
    "H": ("Spain", "Cape Verde", "Uruguay"),
    "I": ("France", "Norway", "Senegal"),
    "J": ("Argentina", "Austria", "Algeria"),
    "K": ("Colombia", "Portugal", "Congo DR"),
    "L": ("England", "Croatia", "Ghana"),
}

GROUP_STAGE_COMPLETE = date(2026, 6, 27)

# Best-eight third-place groups after the June 27, 2026 group stage (FIFA tiebreakers applied).
OFFICIAL_QUALIFYING_THIRD_GROUPS = frozenset({"B", "D", "E", "F", "I", "J", "K", "L"})

_PLACEHOLDER_RE = re.compile(
    r"^Group ([A-L]) (Winners|Runners Up|Runners-up)$",
    re.IGNORECASE,
)
_THIRD_PLACE_RE = re.compile(
    r"^Group ([A-L](?:/[A-L])+) 3rd Place$",
    re.IGNORECASE,
)
_MATCH_REF_RE = re.compile(r"^Match (\d+) (Winner|Loser)$", re.IGNORECASE)


def _wc2026_match_id(match: ScheduleMatch) -> str:
    return f"wc2026_m{match.match_number:03d}"


@dataclass
class KnockoutContext:
    winners: dict[str, str]
    runners: dict[str, str]
    thirds: dict[str, str]
    qualifying_third_groups: set[str]
    third_for_winner: dict[str, str]
    match_winners: dict[int, str]


def is_placeholder_team(name: str) -> bool:
    if not name or is_known_team(name):
        return False
    return bool(
        _PLACEHOLDER_RE.match(name)
        or _THIRD_PLACE_RE.match(name)
        or _MATCH_REF_RE.match(name)
    )


def _should_use_official_standings(*, today: date | None = None) -> bool:
    ref = today or date.today()
    return ref >= GROUP_STAGE_COMPLETE


def _standings_from_official() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    winners: dict[str, str] = {}
    runners: dict[str, str] = {}
    thirds: dict[str, str] = {}
    for group, (first, second, third) in OFFICIAL_GROUP_STANDINGS.items():
        winners[group] = first
        runners[group] = second
        thirds[group] = third
    return winners, runners, thirds


def _team_stats() -> dict[str, dict[str, int]]:
    from collections import defaultdict

    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"p": 0, "gf": 0, "ga": 0},
    )
    results = load_wc2026_results()
    for match in WORLD_CUP_2026_SCHEDULE:
        if match.stage_name != "Group stage":
            continue
        mid = _wc2026_match_id(match)
        if mid not in results:
            continue
        row = results[mid]
        hg, ag = int(row["home_score_ft"]), int(row["away_score_ft"])
        for team, gf, ga in (
            (match.home_team, hg, ag),
            (match.away_team, ag, hg),
        ):
            s = stats[team]
            s["p"] += 3 if gf > ag else (1 if gf == ag else 0)
            s["gf"] += gf
            s["ga"] += ga
    return stats


def _standings_from_results() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    stats = _team_stats()
    winners: dict[str, str] = {}
    runners: dict[str, str] = {}
    thirds: dict[str, str] = {}
    for group, teams in WORLD_CUP_2026_GROUPS.items():
        ranked = sorted(
            teams,
            key=lambda t: (
                -stats[t]["p"],
                -(stats[t]["gf"] - stats[t]["ga"]),
                -stats[t]["gf"],
                t,
            ),
        )
        winners[group] = ranked[0]
        runners[group] = ranked[1]
        thirds[group] = ranked[2]
    return winners, runners, thirds


def build_knockout_context(*, today: date | None = None) -> KnockoutContext:
    if _should_use_official_standings(today=today):
        winners, runners, thirds = _standings_from_official()
    else:
        winners, runners, thirds = _standings_from_results()

    stats = _team_stats()
    third_rows = sorted(
        (
            group,
            thirds[group],
            int(stats.get(thirds[group], {"p": 0})["p"]),
            int(stats.get(thirds[group], {"gf": 0, "ga": 0})["gf"])
            - int(stats.get(thirds[group], {"gf": 0, "ga": 0})["ga"]),
            int(stats.get(thirds[group], {"gf": 0})["gf"]),
        )
        for group in sorted(thirds)
    )
    if _should_use_official_standings(today=today):
        qualifying = set(OFFICIAL_QUALIFYING_THIRD_GROUPS)
    else:
        ranked = sorted(third_rows, key=lambda row: (-row[2], -row[3], -row[4], row[1]))
        qualifying = {row[0] for row in ranked[:8]}

    third_for_winner = {
        winner_group: thirds[third_group_for_winner(qualifying, winner_group)]
        for winner_group in ("A", "B", "D", "E", "G", "I", "K", "L")
    }
    return KnockoutContext(
        winners=winners,
        runners=runners,
        thirds=thirds,
        qualifying_third_groups=qualifying,
        third_for_winner=third_for_winner,
        match_winners={},
    )


def _third_for_slot(allowed_groups: str, fixed_winner_group: str, ctx: KnockoutContext) -> str:
    third_group = third_group_for_winner(ctx.qualifying_third_groups, fixed_winner_group)
    if third_group not in set(allowed_groups.split("/")):
        raise ValueError(
            f"Annex C assigns group {third_group} to 1{fixed_winner_group}, "
            f"but slot allows {allowed_groups}",
        )
    return ctx.thirds[third_group]


# Round-of-32 third-place slots: (allowed third groups, fixed winner group on other side).
R32_THIRD_SLOTS: dict[int, tuple[str, str]] = {
    77: ("C/D/F/G/H", "I"),
    79: ("C/E/F/H/I", "A"),
    80: ("E/H/I/J/K", "L"),
    82: ("A/E/H/I/J", "G"),
    85: ("E/F/G/I/J", "B"),
    87: ("D/E/I/J/L", "K"),
}


def _resolve_simple(name: str, ctx: KnockoutContext) -> str:
    if is_known_team(name):
        return name

    m = _PLACEHOLDER_RE.match(name)
    if m:
        group, slot = m.group(1), m.group(2).lower()
        if "winner" in slot:
            return ctx.winners[group]
        return ctx.runners[group]

    m = _MATCH_REF_RE.match(name)
    if m:
        match_no = int(m.group(1))
        resolved = ctx.match_winners.get(match_no)
        return resolved or name

    return name


def resolve_match_teams(
    match: ScheduleMatch,
    ctx: KnockoutContext | None = None,
) -> tuple[str, str]:
    context = ctx or build_knockout_context()
    home, away = match.home_team, match.away_team

    if match.match_number in R32_THIRD_SLOTS and match.stage_name == "Round of 32":
        allowed, fixed_winner = R32_THIRD_SLOTS[match.match_number]
        if _THIRD_PLACE_RE.match(home):
            home = _third_for_slot(allowed, fixed_winner, context)
        elif _PLACEHOLDER_RE.match(home):
            home = _resolve_simple(home, context)
        if _THIRD_PLACE_RE.match(away):
            away = _third_for_slot(allowed, fixed_winner, context)
        elif _PLACEHOLDER_RE.match(away):
            away = _resolve_simple(away, context)

    home = _resolve_simple(home, context)
    away = _resolve_simple(away, context)
    return home, away


def resolve_schedule_match(
    match: ScheduleMatch,
    ctx: KnockoutContext | None = None,
) -> ScheduleMatch:
    home, away = resolve_match_teams(match, ctx)
    if home == match.home_team and away == match.away_team:
        return match
    return replace(match, home_team=home, away_team=away)


def resolved_world_cup_2026_schedule(
    *,
    today: date | None = None,
) -> list[ScheduleMatch]:
    ctx = build_knockout_context(today=today)
    return [resolve_schedule_match(match, ctx) for match in WORLD_CUP_2026_SCHEDULE]


def is_resolved_predictable_match(
    match: ScheduleMatch,
    ctx: KnockoutContext | None = None,
) -> bool:
    home, away = resolve_match_teams(match, ctx)
    return is_known_team(home) and is_known_team(away)
