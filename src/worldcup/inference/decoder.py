from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Scoreline:
    home_goals: int
    away_goals: int
    prob: float


@dataclass
class PredictionOutput:
    matrix: np.ndarray
    overflow_prob: float
    top3_scorelines: list[Scoreline]
    result_probs: dict[str, float]
    ou25_probs: dict[str, float]
    btts_probs: dict[str, float]
    expected_goals: dict[str, float]
    uncertainty: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "top3_scorelines": [
                {
                    "home_goals": s.home_goals,
                    "away_goals": s.away_goals,
                    "prob": s.prob,
                }
                for s in self.top3_scorelines
            ],
            "result_probs": self.result_probs,
            "ou25_probs": self.ou25_probs,
            "btts_probs": self.btts_probs,
            "expected_goals": self.expected_goals,
            "uncertainty": self.uncertainty,
            "overflow_prob": self.overflow_prob,
        }


def decode_score_matrix(matrix: np.ndarray, overflow_prob: float = 0.0) -> PredictionOutput:
    """Aggregate all serving outputs from a unified score probability matrix."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    grid_max = matrix.shape[0] - 1
    flat = matrix.ravel()
    top_indices = np.argsort(flat)[::-1][:3]
    top3: list[Scoreline] = []
    for idx in top_indices:
        home = idx // matrix.shape[1]
        away = idx % matrix.shape[1]
        top3.append(Scoreline(home_goals=int(home), away_goals=int(away), prob=float(flat[idx])))

    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    over_25 = 0.0
    btts_yes = 0.0
    exp_home = 0.0
    exp_away = 0.0

    for i in range(grid_max + 1):
        for j in range(grid_max + 1):
            p = float(matrix[i, j])
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p
            if i + j >= 3:
                over_25 += p
            if i >= 1 and j >= 1:
                btts_yes += p
            exp_home += i * p
            exp_away += j * p

    entropy = 0.0
    for p in flat:
        if p > 0:
            entropy -= p * np.log(p)
    max_entropy = np.log(len(flat))
    confidence = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)

    return PredictionOutput(
        matrix=matrix,
        overflow_prob=overflow_prob,
        top3_scorelines=top3,
        result_probs={"home_win": home_win, "draw": draw, "away_win": away_win},
        ou25_probs={"over_2_5": over_25, "under_2_5": 1.0 - over_25},
        btts_probs={"yes": btts_yes, "no": 1.0 - btts_yes},
        expected_goals={"home": exp_home, "away": exp_away, "total": exp_home + exp_away},
        uncertainty={
            "entropy": float(entropy),
            "ensemble_var": 0.0,
            "ood_score": 0.0,
            "confidence": float(confidence),
        },
    )
