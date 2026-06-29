"""Post-inference adjustments: neutral venue, knockout pace, market calibration."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from worldcup.inference.decoder import PredictionOutput, decode_score_matrix
from worldcup.inference.market_odds import max_result_divergence, normalize_probs
from worldcup.inference.predictor import MatchPrediction
from worldcup.models.score_matrix import apply_dixon_coles_adjustment, independent_score_matrix

DEFAULT_HOME_ADVANTAGE_LOG = 0.15
DEFAULT_KNOCKOUT_LAMBDA_SCALE = 0.88
DEFAULT_MARKET_BLEND_WEIGHT = 0.45
DEFAULT_RHO = -0.13
DEFAULT_GRID_MAX = 7
DIVERGENCE_WARN_THRESHOLD = 0.20


def is_neutral_venue(row: pd.Series | dict[str, Any]) -> bool:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    return bool(row.get("is_world_cup")) and bool(row.get("is_knockout"))


def market_probs_from_row(row: pd.Series | dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    if not bool(row.get("market_odds_available")):
        return None
    result = normalize_probs(
        {
            "home_win": float(row["market_home_implied"]),
            "draw": float(row["market_draw_implied"]),
            "away_win": float(row["market_away_implied"]),
        }
    )
    ou = normalize_probs(
        {
            "over_2_5": float(row.get("market_over25_implied", 0.5)),
            "under_2_5": float(row.get("market_under25_implied", 0.5)),
        }
    )
    return {
        "result_probs": result,
        "ou25_probs": ou,
        "source": str(row.get("market_odds_source") or "market"),
    }


def _rebuild_matrix(
    lambda_home: float,
    lambda_away: float,
    *,
    rho: float = DEFAULT_RHO,
    grid_max_goal: int = DEFAULT_GRID_MAX,
) -> tuple[np.ndarray, float]:
    matrix, overflow = independent_score_matrix(lambda_home, lambda_away, grid_max_goal)
    matrix = apply_dixon_coles_adjustment(matrix, rho, lambda_home, lambda_away)
    return matrix, overflow


def _adjust_lambdas_for_context(
    lambda_home: float,
    lambda_away: float,
    row: pd.Series | dict[str, Any],
    *,
    home_advantage_log: float = DEFAULT_HOME_ADVANTAGE_LOG,
    knockout_scale: float = DEFAULT_KNOCKOUT_LAMBDA_SCALE,
) -> tuple[float, float, list[str]]:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    applied: list[str] = []
    log_home = math.log(max(lambda_home, 1e-6))
    log_away = math.log(max(lambda_away, 1e-6))

    if is_neutral_venue(row):
        shift = home_advantage_log / 2.0
        log_home -= shift
        log_away += shift
        applied.append("neutral_venue")

    lambda_home = math.exp(log_home)
    lambda_away = math.exp(log_away)

    if bool(row.get("is_knockout")):
        lambda_home *= knockout_scale
        lambda_away *= knockout_scale
        applied.append("knockout_pace")

    return lambda_home, lambda_away, applied


def _fit_lambdas_to_targets(
    home_win_t: float,
    draw_t: float,
    away_win_t: float,
    over25_t: float,
    *,
    rho: float = DEFAULT_RHO,
    grid_max_goal: int = DEFAULT_GRID_MAX,
) -> tuple[float, float]:
    best_loss = float("inf")
    best_pair = (1.2, 1.0)
    for lambda_home in np.linspace(0.35, 2.6, 24):
        for lambda_away in np.linspace(0.35, 2.6, 24):
            matrix, overflow = _rebuild_matrix(
                float(lambda_home),
                float(lambda_away),
                rho=rho,
                grid_max_goal=grid_max_goal,
            )
            decoded = decode_score_matrix(matrix, overflow)
            loss = (
                (decoded.result_probs["home_win"] - home_win_t) ** 2
                + (decoded.result_probs["draw"] - draw_t) ** 2
                + (decoded.result_probs["away_win"] - away_win_t) ** 2
                + 0.5 * (decoded.ou25_probs["over_2_5"] - over25_t) ** 2
            )
            if loss < best_loss:
                best_loss = loss
                best_pair = (float(lambda_home), float(lambda_away))
    return best_pair


def _blend_probs(
    model_probs: dict[str, float],
    market_probs: dict[str, float],
    *,
    weight: float,
) -> dict[str, float]:
    weight = max(0.0, min(1.0, weight))
    blended = {
        key: (1.0 - weight) * float(model_probs[key]) + weight * float(market_probs[key])
        for key in model_probs
    }
    return normalize_probs(blended)


def apply_prediction_adjustments(
    prediction: MatchPrediction,
    row: pd.Series | dict[str, Any],
    *,
    market: dict[str, Any] | None = None,
    market_blend_weight: float = DEFAULT_MARKET_BLEND_WEIGHT,
    rho: float = DEFAULT_RHO,
    grid_max_goal: int = DEFAULT_GRID_MAX,
) -> tuple[MatchPrediction, dict[str, Any]]:
    if isinstance(row, pd.Series):
        row = row.to_dict()

    lambda_home, lambda_away, applied = _adjust_lambdas_for_context(
        prediction.lambda_home,
        prediction.lambda_away,
        row,
    )
    matrix, overflow = _rebuild_matrix(
        lambda_home,
        lambda_away,
        rho=rho,
        grid_max_goal=grid_max_goal,
    )
    output = decode_score_matrix(matrix, overflow)

    comparison: dict[str, Any] = {
        "adjustments_applied": applied,
        "market_available": False,
        "market_source": None,
        "model_result_probs": dict(output.result_probs),
        "market_result_probs": None,
        "max_result_divergence": None,
        "market_warning": False,
        "market_blend_weight": 0.0,
    }

    market_payload = market if market is not None else market_probs_from_row(row)
    if market_payload:
        comparison["market_available"] = True
        comparison["market_source"] = market_payload.get("source", "market")
        comparison["market_result_probs"] = dict(market_payload["result_probs"])
        comparison["max_result_divergence"] = max_result_divergence(
            output.result_probs,
            market_payload["result_probs"],
        )
        comparison["market_warning"] = (
            comparison["max_result_divergence"] >= DIVERGENCE_WARN_THRESHOLD
        )

        blended_result = _blend_probs(
            output.result_probs,
            market_payload["result_probs"],
            weight=market_blend_weight,
        )
        blended_ou = _blend_probs(
            output.ou25_probs,
            market_payload["ou25_probs"],
            weight=market_blend_weight,
        )
        lambda_home, lambda_away = _fit_lambdas_to_targets(
            blended_result["home_win"],
            blended_result["draw"],
            blended_result["away_win"],
            blended_ou["over_2_5"],
            rho=rho,
            grid_max_goal=grid_max_goal,
        )
        matrix, overflow = _rebuild_matrix(
            lambda_home,
            lambda_away,
            rho=rho,
            grid_max_goal=grid_max_goal,
        )
        output = decode_score_matrix(matrix, overflow)
        applied.append("market_blend")
        comparison["adjustments_applied"] = applied
        comparison["market_blend_weight"] = market_blend_weight
        comparison["calibrated_result_probs"] = dict(output.result_probs)
        comparison["calibrated_ou25_probs"] = dict(output.ou25_probs)

    adjusted = replace(
        prediction,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        output=output,
    )
    return adjusted, comparison


def divergence_summary(comparison: dict[str, Any]) -> str | None:
    if not comparison.get("market_available"):
        return None
    div = comparison.get("max_result_divergence")
    if div is None:
        return None
    if comparison.get("market_warning"):
        return f"模型与赛前市场 1X2 最大偏差 {100.0 * div:.1f}%（≥20% 标红）"
    return f"模型与赛前市场 1X2 最大偏差 {100.0 * div:.1f}%"
