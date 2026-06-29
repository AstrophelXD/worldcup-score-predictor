"""Post-inference adjustments: neutral venue, knockout pace, market calibration."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from worldcup.inference.decoder import decode_score_matrix
from worldcup.inference.market_odds import max_result_divergence, normalize_probs
from worldcup.inference.predictor import MatchPrediction
from worldcup.models.score_matrix import apply_dixon_coles_adjustment, independent_score_matrix

DEFAULT_HOME_ADVANTAGE_LOG = 0.15
DEFAULT_KNOCKOUT_LAMBDA_SCALE = 0.82
DEFAULT_MARKET_BLEND_WEIGHT = 0.55
DEFAULT_OU_BLEND_WEIGHT = 0.62
DEFAULT_RHO = -0.13
KNOCKOUT_RHO = -0.20
DEFAULT_GRID_MAX = 7
DIVERGENCE_CAUTION_THRESHOLD = 0.15
DIVERGENCE_WARN_THRESHOLD = 0.20
DIVERGENCE_CRITICAL_THRESHOLD = 0.25
MAX_MARKET_BLEND_WEIGHT = 0.88
CRITICAL_BLEND_FLOOR = 0.82


def is_neutral_venue(row: pd.Series | dict[str, Any]) -> bool:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    return bool(row.get("is_world_cup"))


def is_knockout_row(row: pd.Series | dict[str, Any]) -> bool:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    return bool(row.get("is_knockout"))


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

    if is_knockout_row(row):
        lambda_home *= knockout_scale
        lambda_away *= knockout_scale
        applied.append("knockout_pace")

    return lambda_home, lambda_away, applied


def _context_rho(row: pd.Series | dict[str, Any]) -> float:
    return KNOCKOUT_RHO if is_knockout_row(row) else DEFAULT_RHO


def _adaptive_blend_weight(divergence: float, *, base: float = DEFAULT_MARKET_BLEND_WEIGHT) -> float:
    """Raise market weight when raw model diverges from bookmaker lines."""
    extra = max(0.0, divergence - 0.10) * 1.5
    weight = min(MAX_MARKET_BLEND_WEIGHT, base + extra)
    if divergence >= DIVERGENCE_CRITICAL_THRESHOLD:
        weight = max(weight, 0.78 + min(0.10, (divergence - DIVERGENCE_CRITICAL_THRESHOLD) * 0.35))
    return min(MAX_MARKET_BLEND_WEIGHT, weight)


def _adaptive_ou_blend_weight(divergence: float) -> float:
    extra = max(0.0, divergence - 0.08) * 1.0
    return min(0.75, DEFAULT_OU_BLEND_WEIGHT + extra)


def _nudge_knockout_draw(probs: dict[str, float]) -> dict[str, float]:
    boosted = dict(probs)
    boosted["draw"] = float(boosted["draw"]) + 0.025
    return normalize_probs(boosted)


def _fit_lambdas_to_targets(
    home_win_t: float,
    draw_t: float,
    away_win_t: float,
    over25_t: float,
    *,
    rho: float = DEFAULT_RHO,
    grid_max_goal: int = DEFAULT_GRID_MAX,
    draw_weight: float = 1.0,
) -> tuple[float, float]:
    best_loss = float("inf")
    best_pair = (1.2, 1.0)
    if away_win_t > home_win_t + 0.15:
        home_range = np.linspace(0.25, 1.25, 20)
        away_range = np.linspace(0.80, 3.50, 28)
    elif home_win_t > away_win_t + 0.15:
        home_range = np.linspace(0.80, 3.50, 28)
        away_range = np.linspace(0.25, 1.25, 20)
    else:
        home_range = np.linspace(0.30, 2.4, 28)
        away_range = np.linspace(0.30, 2.4, 28)
    for lambda_home in home_range:
        for lambda_away in away_range:
            matrix, overflow = _rebuild_matrix(
                float(lambda_home),
                float(lambda_away),
                rho=rho,
                grid_max_goal=grid_max_goal,
            )
            decoded = decode_score_matrix(matrix, overflow)
            loss = (
                (decoded.result_probs["home_win"] - home_win_t) ** 2
                + draw_weight * (decoded.result_probs["draw"] - draw_t) ** 2
                + (decoded.result_probs["away_win"] - away_win_t) ** 2
                + 0.65 * (decoded.ou25_probs["over_2_5"] - over25_t) ** 2
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


def _detect_alerts(
    raw_result: dict[str, float],
    market_result: dict[str, float] | None,
    row: dict[str, Any],
) -> list[str]:
    alerts: list[str] = []
    if market_result:
        if market_result.get("away_win", 0) > 0.70 and raw_result.get("away_win", 0) < 0.30:
            alerts.append("likely_team_mapping_bug")
        if market_result.get("home_win", 0) > 0.70 and raw_result.get("home_win", 0) < 0.30:
            alerts.append("likely_team_mapping_bug")
    hr = row.get("home_avg_player_rating")
    ar = row.get("away_avg_player_rating")
    if hr is not None and ar is not None:
        if abs(float(hr) - float(ar)) < 1.0 and 70.0 <= float(hr) <= 74.0:
            alerts.append("generic_player_data_suspected")
        if float(ar) >= 82.0 and raw_result.get("away_win", 0) < 0.35:
            if market_result and market_result.get("away_win", 0) > 0.65:
                alerts.append("star_player_signal_missing")
        if float(hr) >= 82.0 and raw_result.get("home_win", 0) < 0.35:
            if market_result and market_result.get("home_win", 0) > 0.65:
                alerts.append("star_player_signal_missing")
    return alerts


def _divergence_level(divergence: float | None) -> str:
    if divergence is None:
        return "none"
    if divergence >= DIVERGENCE_CRITICAL_THRESHOLD:
        return "critical"
    if divergence >= DIVERGENCE_WARN_THRESHOLD:
        return "warning"
    if divergence >= DIVERGENCE_CAUTION_THRESHOLD:
        return "caution"
    return "ok"


def apply_prediction_adjustments(
    prediction: MatchPrediction,
    row: pd.Series | dict[str, Any],
    *,
    market: dict[str, Any] | None = None,
    market_blend_weight: float | None = None,
    rho: float | None = None,
    grid_max_goal: int = DEFAULT_GRID_MAX,
) -> tuple[MatchPrediction, dict[str, Any]]:
    if isinstance(row, pd.Series):
        row = row.to_dict()

    raw_result_probs = dict(prediction.output.result_probs)
    context_rho = DEFAULT_RHO if rho is None else rho
    context_rho = _context_rho(row) if rho is None else rho

    lambda_home, lambda_away, applied = _adjust_lambdas_for_context(
        prediction.lambda_home,
        prediction.lambda_away,
        row,
    )
    matrix, overflow = _rebuild_matrix(
        lambda_home,
        lambda_away,
        rho=context_rho,
        grid_max_goal=grid_max_goal,
    )
    output = decode_score_matrix(matrix, overflow)

    comparison: dict[str, Any] = {
        "adjustments_applied": applied,
        "market_available": False,
        "market_source": None,
        "raw_result_probs": raw_result_probs,
        "pre_blend_result_probs": dict(output.result_probs),
        "model_result_probs": dict(raw_result_probs),
        "market_result_probs": None,
        "max_result_divergence": None,
        "market_caution": False,
        "market_warning": False,
        "market_critical": False,
        "alerts": [],
        "divergence_level": "none",
        "market_blend_weight": 0.0,
        "ou_blend_weight": 0.0,
    }

    market_payload = market if market is not None else market_probs_from_row(row)
    if market_payload:
        raw_div = max_result_divergence(raw_result_probs, market_payload["result_probs"])
        level = _divergence_level(raw_div)
        alerts = _detect_alerts(raw_result_probs, market_payload["result_probs"], row)
        if raw_div >= DIVERGENCE_CRITICAL_THRESHOLD:
            alerts.append("critical_market_divergence")
        blend_weight = (
            market_blend_weight
            if market_blend_weight is not None
            else _adaptive_blend_weight(raw_div)
        )
        ou_weight = _adaptive_ou_blend_weight(raw_div)

        comparison.update(
            {
                "market_available": True,
                "market_source": market_payload.get("source", "market"),
                "market_result_probs": dict(market_payload["result_probs"]),
                "max_result_divergence": raw_div,
                "market_caution": level in {"caution", "warning", "critical"},
                "market_warning": level in {"warning", "critical"},
                "market_critical": level == "critical",
                "divergence_level": level,
                "alerts": alerts,
            }
        )
        if "likely_team_mapping_bug" in alerts:
            blend_weight = min(MAX_MARKET_BLEND_WEIGHT, max(blend_weight, CRITICAL_BLEND_FLOOR))
        if "star_player_signal_missing" in alerts:
            blend_weight = min(MAX_MARKET_BLEND_WEIGHT, max(blend_weight, 0.80))

        blended_result = _blend_probs(
            output.result_probs,
            market_payload["result_probs"],
            weight=blend_weight,
        )
        if is_knockout_row(row):
            blended_result = _nudge_knockout_draw(blended_result)
        blended_ou = _blend_probs(
            output.ou25_probs,
            market_payload["ou25_probs"],
            weight=ou_weight,
        )
        draw_weight = 1.35 if is_knockout_row(row) else 1.0
        lambda_home, lambda_away = _fit_lambdas_to_targets(
            blended_result["home_win"],
            blended_result["draw"],
            blended_result["away_win"],
            blended_ou["over_2_5"],
            rho=context_rho,
            grid_max_goal=grid_max_goal,
            draw_weight=draw_weight,
        )
        matrix, overflow = _rebuild_matrix(
            lambda_home,
            lambda_away,
            rho=context_rho,
            grid_max_goal=grid_max_goal,
        )
        output = decode_score_matrix(matrix, overflow)
        applied.append("market_blend")
        comparison["adjustments_applied"] = applied
        comparison["market_blend_weight"] = blend_weight
        comparison["ou_blend_weight"] = ou_weight
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
    level = comparison.get("divergence_level", "ok")
    pct = 100.0 * float(div)
    if level == "critical":
        return f"模型与市场 1X2 最大偏差 {pct:.1f}%（≥25% 严重偏离，模型不可用于实战判断）"
    if level == "warning":
        return f"模型与市场 1X2 最大偏差 {pct:.1f}%（≥20% 标红，模型观点较激进）"
    if level == "caution":
        return f"模型与市场 1X2 最大偏差 {pct:.1f}%（≥15% 标黄，模型观点较激进）"
    return f"模型与市场 1X2 最大偏差 {pct:.1f}%"
