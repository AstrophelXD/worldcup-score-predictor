from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from worldcup.models.score_matrix import (
    apply_dixon_coles_adjustment,
    independent_score_matrix,
)


@dataclass
class DixonColesParams:
    lambda_home: float
    lambda_away: float
    rho: float = -0.13
    home_advantage: float = 0.15
    grid_max_goal: int = 7


@dataclass
class DixonColesOutput:
    params: DixonColesParams
    matrix: np.ndarray
    overflow_prob: float


class DixonColesModel:
    """Baseline Poisson score model with Dixon-Coles low-score correction."""

    def __init__(
        self,
        home_advantage: float = 0.15,
        rho: float = -0.13,
        grid_max_goal: int = 7,
    ) -> None:
        self.home_advantage = home_advantage
        self.rho = rho
        self.grid_max_goal = grid_max_goal

    def predict(
        self,
        lambda_home: float,
        lambda_away: float,
        rho: float | None = None,
    ) -> DixonColesOutput:
        rho = self.rho if rho is None else rho
        matrix, overflow_prob = independent_score_matrix(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            grid_max_goal=self.grid_max_goal,
        )
        matrix = apply_dixon_coles_adjustment(matrix, rho, lambda_home, lambda_away)
        params = DixonColesParams(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            rho=rho,
            home_advantage=self.home_advantage,
            grid_max_goal=self.grid_max_goal,
        )
        return DixonColesOutput(params=params, matrix=matrix, overflow_prob=overflow_prob)
