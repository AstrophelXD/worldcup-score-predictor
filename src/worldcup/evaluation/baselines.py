"""Simple baselines for 1X2 and Top3 scoreline evaluation."""

from __future__ import annotations

from typing import Any

from worldcup.backtesting.metrics import actual_result

NAIVE_TOP3_SCORELINES: tuple[tuple[int, int], ...] = ((1, 1), (1, 0), (0, 1))
NAIVE_TOP6_SCORELINES: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 0),
    (0, 1),
    (2, 1),
    (1, 2),
    (0, 0),
)


def baseline_elo_favorite(home_elo: float, away_elo: float) -> str:
    if home_elo > away_elo + 20:
        return "home_win"
    if away_elo > home_elo + 20:
        return "away_win"
    return "draw"


def baseline_fifa_favorite(home_rank: float, away_rank: float) -> str:
    if home_rank < away_rank:
        return "home_win"
    if away_rank < home_rank:
        return "away_win"
    return "draw"


def baseline_market_favorite(row: dict[str, Any]) -> str | None:
    if not row.get("market_odds_available"):
        return None
    probs = {
        "home_win": float(row.get("market_home_implied") or 0),
        "draw": float(row.get("market_draw_implied") or 0),
        "away_win": float(row.get("market_away_implied") or 0),
    }
    if sum(probs.values()) <= 0:
        return None
    return max(probs, key=probs.get)


def hit_naive_top3(
    home_goals: int,
    away_goals: int,
    *,
    templates: tuple[tuple[int, int], ...] = NAIVE_TOP3_SCORELINES,
) -> bool:
    return (home_goals, away_goals) in templates


def baseline_hit_1x2(favorite: str | None, home_goals: int, away_goals: int) -> bool | None:
    if favorite is None:
        return None
    return favorite == actual_result(home_goals, away_goals)
