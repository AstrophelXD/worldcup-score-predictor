from __future__ import annotations

import math

from worldcup.models.score_matrix import poisson_pmf


def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def match_log_likelihood(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    if lambda_home <= 0 or lambda_away <= 0:
        return -math.inf
    tau = dixon_coles_tau(home_goals, away_goals, lambda_home, lambda_away, rho)
    if tau <= 0:
        return -math.inf
    home_prob = poisson_pmf(home_goals, lambda_home)
    away_prob = poisson_pmf(away_goals, lambda_away)
    if home_prob <= 0 or away_prob <= 0:
        return -math.inf
    return math.log(home_prob) + math.log(away_prob) + math.log(tau)


def expected_lambdas(
    home_team_id: str,
    away_team_id: str,
    attack: dict[str, float],
    defense: dict[str, float],
    home_advantage: float,
) -> tuple[float, float]:
    log_lambda_home = (
        home_advantage
        + attack.get(home_team_id, 0.0)
        - defense.get(away_team_id, 0.0)
    )
    log_lambda_away = attack.get(away_team_id, 0.0) - defense.get(home_team_id, 0.0)
    log_lambda_home = max(min(log_lambda_home, 5.0), -5.0)
    log_lambda_away = max(min(log_lambda_away, 5.0), -5.0)
    return math.exp(log_lambda_home), math.exp(log_lambda_away)
