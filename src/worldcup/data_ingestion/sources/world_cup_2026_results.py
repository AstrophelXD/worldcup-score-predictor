"""Played-result overlay for FIFA World Cup 2026 (updated as tournament progresses)."""

from __future__ import annotations

import hashlib
import math
import struct
from datetime import date
from pathlib import Path

import pandas as pd

from worldcup.dashboard.flags import is_known_team
from worldcup.dashboard.world_cup_2026_schedule import WORLD_CUP_2026_SCHEDULE, ScheduleMatch
from worldcup.data_ingestion.sources.world_cup_2026_strength import strength_for
from worldcup.utils.paths import project_root

DEFAULT_RESULTS_PATH = project_root() / "data" / "samples" / "wc2026_results.csv"

# Confirmed real-world / storyline results (override simulation).
KNOWN_RESULTS: dict[str, tuple[int, int, str]] = {
    "wc2026_m018": (1, 4, "Iraq vs Norway — Haaland brace"),
}


def _rng_triplet(match_id: str) -> tuple[float, float, float]:
    digest = hashlib.sha256(match_id.encode()).digest()
    a, b, c = struct.unpack(">HHH", digest[:6])
    return a / 65535.0, b / 65535.0, c / 65535.0


def simulate_match_score(home_team: str, away_team: str, match_id: str) -> tuple[int, int]:
    """Deterministic plausible score from Elo (neutral site)."""
    home_elo = float(strength_for(home_team, default_elo=1500, default_rank=80)["elo"])
    away_elo = float(strength_for(away_team, default_elo=1500, default_rank=80)["elo"])
    diff = (home_elo - away_elo) / 400.0
    r1, r2, r3 = _rng_triplet(match_id)
    lam_home = max(0.35, 1.05 * math.exp(diff * 0.45) * (0.82 + 0.36 * r1))
    lam_away = max(0.35, 1.05 * math.exp(-diff * 0.45) * (0.82 + 0.36 * r2))
    home_goals = min(5, int(round(lam_home + r1 * 1.4 - 0.25)))
    away_goals = min(5, int(round(lam_away + r2 * 1.4 - 0.25)))
    if home_goals == 0 and away_goals == 0 and r3 > 0.25:
        if diff >= 0.15:
            home_goals = 1
        elif diff <= -0.15:
            away_goals = 1
        else:
            home_goals, away_goals = 1, 1
    return max(0, home_goals), max(0, away_goals)


def is_result_eligible_match(match: ScheduleMatch, *, cutoff: date) -> bool:
    if date.fromisoformat(match.match_date) > cutoff:
        return False
    return is_known_team(match.home_team) and is_known_team(match.away_team)


def generate_wc2026_results_through(
    cutoff: str | date = "2026-06-28",
    *,
    overrides: dict[str, tuple[int, int, str]] | None = None,
) -> list[dict[str, str | int]]:
    cutoff_day = date.fromisoformat(str(cutoff))
    merged_overrides = dict(KNOWN_RESULTS)
    if overrides:
        merged_overrides.update(overrides)

    rows: list[dict[str, str | int]] = []
    for match in WORLD_CUP_2026_SCHEDULE:
        if not is_result_eligible_match(match, cutoff=cutoff_day):
            continue
        match_id = f"wc2026_m{match.match_number:03d}"
        if match_id in merged_overrides:
            home_score, away_score, note = merged_overrides[match_id]
        else:
            home_score, away_score = simulate_match_score(
                match.home_team,
                match.away_team,
                match_id,
            )
            note = f"{match.home_team} vs {match.away_team}"
        rows.append(
            {
                "match_id": match_id,
                "home_score_ft": int(home_score),
                "away_score_ft": int(away_score),
                "home_score_ht": "",
                "away_score_ht": "",
                "notes": note,
            }
        )
    return rows


def write_wc2026_results_csv(
    path: Path | None = None,
    *,
    cutoff: str | date = "2026-06-28",
) -> int:
    target = path or DEFAULT_RESULTS_PATH
    rows = generate_wc2026_results_through(cutoff=cutoff)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)
    return len(rows)


def load_wc2026_results(path: Path | None = None) -> dict[str, dict[str, int | str]]:
    csv_path = path or DEFAULT_RESULTS_PATH
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    if df.empty or "match_id" not in df.columns:
        return {}

    results: dict[str, dict[str, int | str]] = {}
    for row in df.itertuples(index=False):
        match_id = str(getattr(row, "match_id", "") or "").strip()
        home = getattr(row, "home_score_ft", None)
        away = getattr(row, "away_score_ft", None)
        if not match_id or pd.isna(home) or pd.isna(away):
            continue
        payload: dict[str, int | str] = {
            "home_score_ft": int(home),
            "away_score_ft": int(away),
            "status": "finished",
        }
        home_ht = getattr(row, "home_score_ht", None)
        away_ht = getattr(row, "away_score_ht", None)
        if pd.notna(home_ht):
            payload["home_score_ht"] = int(home_ht)
        if pd.notna(away_ht):
            payload["away_score_ht"] = int(away_ht)
        notes = getattr(row, "notes", None)
        if pd.notna(notes) and str(notes).strip():
            payload["notes"] = str(notes).strip()
        results[match_id] = payload
    return results


def apply_results_to_row(row: dict, result: dict[str, int | str] | None) -> dict:
    if not result:
        return row
    updated = dict(row)
    for key in ("home_score_ft", "away_score_ft", "home_score_ht", "away_score_ht", "status"):
        if key in result:
            updated[key] = result[key]
    return updated
