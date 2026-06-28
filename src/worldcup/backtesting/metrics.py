from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from worldcup.inference.decoder import PredictionOutput


@dataclass
class MatchMetrics:
    match_id: str
    score_nll: float
    top3_hit: bool
    brier_1x2: float
    brier_ou25: float
    brier_btts: float
    rps: float


@dataclass
class AggregateMetrics:
    score_nll: float
    top3_hit_rate: float
    brier_1x2: float
    brier_ou25: float
    brier_btts: float
    rps: float
    ece_1x2: float
    match_count: int


def actual_result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals == away_goals:
        return "draw"
    return "away_win"


def score_nll(
    home_goals: int,
    away_goals: int,
    matrix: np.ndarray,
    overflow_prob: float,
    eps: float = 1e-12,
) -> float:
    grid_max = matrix.shape[0] - 1
    if home_goals > grid_max or away_goals > grid_max:
        prob = max(overflow_prob, eps)
    else:
        prob = max(float(matrix[home_goals, away_goals]), eps)
    return -math.log(prob)


def top3_hit(home_goals: int, away_goals: int, output: PredictionOutput) -> bool:
    for scoreline in output.top3_scorelines:
        if scoreline.home_goals == home_goals and scoreline.away_goals == away_goals:
            return True
    return False


def brier_multiclass(probs: dict[str, float], actual_key: str) -> float:
    return sum((prob - (1.0 if key == actual_key else 0.0)) ** 2 for key, prob in probs.items())


def brier_binary(prob_positive: float, actual_positive: bool) -> float:
    target = 1.0 if actual_positive else 0.0
    return (prob_positive - target) ** 2


def ranked_probability_score(probs: dict[str, float], actual_key: str) -> float:
    order = ["home_win", "draw", "away_win"]
    cumulative = 0.0
    for key in order:
        cumulative += probs[key]
        if key == actual_key:
            return (cumulative - probs[key]) ** 2
    return 1.0


def evaluate_match(
    match_id: str,
    home_goals: int,
    away_goals: int,
    output: PredictionOutput,
) -> MatchMetrics:
    result_key = actual_result(home_goals, away_goals)
    total_goals = home_goals + away_goals
    btts = home_goals >= 1 and away_goals >= 1
    return MatchMetrics(
        match_id=match_id,
        score_nll=score_nll(home_goals, away_goals, output.matrix, output.overflow_prob),
        top3_hit=top3_hit(home_goals, away_goals, output),
        brier_1x2=brier_multiclass(output.result_probs, result_key),
        brier_ou25=brier_binary(output.ou25_probs["over_2_5"], total_goals >= 3),
        brier_btts=brier_binary(output.btts_probs["yes"], btts),
        rps=ranked_probability_score(output.result_probs, result_key),
    )


def expected_calibration_error(
    confidences: list[float],
    hits: list[bool],
    n_bins: int = 10,
) -> float:
    if not confidences:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidences)
    ece = 0.0
    for idx in range(n_bins):
        low, high = bins[idx], bins[idx + 1]
        mask = [(low <= c < high) if idx < n_bins - 1 else (low <= c <= high) for c in confidences]
        if not any(mask):
            continue
        bin_conf = np.mean([confidences[i] for i, m in enumerate(mask) if m])
        bin_acc = np.mean([float(hits[i]) for i, m in enumerate(mask) if m])
        weight = sum(mask) / total
        ece += weight * abs(bin_acc - bin_conf)
    return float(ece)


def aggregate_metrics(
    per_match: list[MatchMetrics],
    confidences: list[float],
    top1_hits: list[bool],
) -> AggregateMetrics:
    count = len(per_match)
    if count == 0:
        raise ValueError("no matches to aggregate")
    return AggregateMetrics(
        score_nll=float(np.mean([m.score_nll for m in per_match])),
        top3_hit_rate=float(np.mean([float(m.top3_hit) for m in per_match])),
        brier_1x2=float(np.mean([m.brier_1x2 for m in per_match])),
        brier_ou25=float(np.mean([m.brier_ou25 for m in per_match])),
        brier_btts=float(np.mean([m.brier_btts for m in per_match])),
        rps=float(np.mean([m.rps for m in per_match])),
        ece_1x2=expected_calibration_error(confidences, top1_hits),
        match_count=count,
    )
