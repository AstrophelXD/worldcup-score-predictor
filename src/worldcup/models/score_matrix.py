from __future__ import annotations

import math

import numpy as np


def poisson_pmf(k: int, lam: float) -> float:
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    if k < 0:
        return 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def independent_score_matrix(
    lambda_home: float,
    lambda_away: float,
    grid_max_goal: int = 7,
    max_support: int = 20,
) -> tuple[np.ndarray, float]:
    """Build independent Poisson score matrix truncated to grid_max_goal."""
    home_probs = np.array([poisson_pmf(i, lambda_home) for i in range(max_support + 1)])
    away_probs = np.array([poisson_pmf(j, lambda_away) for j in range(max_support + 1)])

    full = np.outer(home_probs, away_probs)
    grid = full[: grid_max_goal + 1, : grid_max_goal + 1]
    overflow_prob = float(1.0 - grid.sum())
    if overflow_prob < 0:
        overflow_prob = 0.0

    total = grid.sum() + overflow_prob
    if total <= 0:
        raise ValueError("invalid score matrix: total mass is zero")
    grid = grid / total
    overflow_prob = overflow_prob / total
    return grid, overflow_prob


def apply_dixon_coles_adjustment(
    matrix: np.ndarray,
    rho: float,
    lambda_home: float,
    lambda_away: float,
) -> np.ndarray:
    """Apply low-score correlation adjustment (Dixon-Coles)."""
    adjusted = matrix.copy()
    tau_map = {
        (0, 0): 1.0 - lambda_home * lambda_away * rho,
        (0, 1): 1.0 + lambda_home * rho,
        (1, 0): 1.0 + lambda_away * rho,
        (1, 1): 1.0 - rho,
    }
    for (i, j), tau in tau_map.items():
        if i < adjusted.shape[0] and j < adjusted.shape[1]:
            adjusted[i, j] *= tau
    total = adjusted.sum()
    if total <= 0:
        raise ValueError("Dixon-Coles adjustment produced non-positive mass")
    return adjusted / total
