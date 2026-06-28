from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationResult:
    temperature: float
    nll_before: float
    nll_after: float


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def apply_temperature(result_probs: dict[str, float], temperature: float) -> dict[str, float]:
    keys = ["home_win", "draw", "away_win"]
    logits = np.log(np.array([max(result_probs[k], 1e-12) for k in keys]))
    scaled = _softmax(logits / max(temperature, 1e-6))
    return {key: float(value) for key, value in zip(keys, scaled, strict=True)}


def fit_temperature(
    prob_rows: list[dict[str, float]],
    actual_keys: list[str],
) -> CalibrationResult:
    keys = ["home_win", "draw", "away_win"]
    logits = np.array(
        [[math.log(max(row[k], 1e-12)) for k in keys] for row in prob_rows]
    )

    def nll_for_temp(temp: float) -> float:
        total = 0.0
        for idx, actual in enumerate(actual_keys):
            probs = _softmax(logits[idx] / max(temp, 1e-6))
            actual_idx = keys.index(actual)
            total -= math.log(max(float(probs[actual_idx]), 1e-12))
        return total / len(actual_keys)

    candidates = np.linspace(0.5, 3.0, 26)
    nll_before = nll_for_temp(1.0)
    best_temp = 1.0
    best_nll = nll_before
    for temp in candidates:
        value = nll_for_temp(float(temp))
        if value < best_nll:
            best_nll = value
            best_temp = float(temp)

    return CalibrationResult(temperature=best_temp, nll_before=nll_before, nll_after=best_nll)
