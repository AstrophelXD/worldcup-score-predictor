"""Offline evaluation helpers: baselines, audits, aggregate metrics."""

from worldcup.evaluation.baselines import (
    NAIVE_TOP3_SCORELINES,
    NAIVE_TOP6_SCORELINES,
    baseline_elo_favorite,
    baseline_fifa_favorite,
    baseline_market_favorite,
    hit_naive_top3,
)
from worldcup.evaluation.match_eval import (
    audit_prediction_consistency,
    evaluate_played_match,
    summarize_evaluations,
)

__all__ = [
    "NAIVE_TOP3_SCORELINES",
    "NAIVE_TOP6_SCORELINES",
    "audit_prediction_consistency",
    "baseline_elo_favorite",
    "baseline_fifa_favorite",
    "baseline_market_favorite",
    "evaluate_played_match",
    "hit_naive_top3",
    "summarize_evaluations",
]
