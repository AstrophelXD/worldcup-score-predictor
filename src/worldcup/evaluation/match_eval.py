"""Per-match evaluation and aggregate summaries."""

from __future__ import annotations

from typing import Any

from worldcup.backtesting.metrics import actual_result, brier_multiclass, ranked_probability_score
from worldcup.evaluation.baselines import (
    NAIVE_TOP3_SCORELINES,
    NAIVE_TOP6_SCORELINES,
    baseline_elo_favorite,
    baseline_fifa_favorite,
    baseline_hit_1x2,
    baseline_market_favorite,
    hit_naive_top3,
)


def _predicted_1x2(result_probs: dict[str, float]) -> str:
    return max(result_probs, key=result_probs.get)


def _top3_hit(top3: list[dict[str, Any]], home_goals: int, away_goals: int) -> bool:
    return any(
        int(s["home_goals"]) == home_goals and int(s["away_goals"]) == away_goals for s in top3
    )


def _top1_outcome(top3: list[dict[str, Any]]) -> str | None:
    if not top3:
        return None
    top = top3[0]
    return actual_result(int(top["home_goals"]), int(top["away_goals"]))


def slice_prediction_payload(payload: dict[str, Any], *, use_raw: bool) -> dict[str, Any]:
    """Select raw checkpoint output vs post-market adjusted output."""
    if not use_raw:
        return payload
    sliced = dict(payload)
    if payload.get("raw_result_probs"):
        sliced["result_probs"] = payload["raw_result_probs"]
    if payload.get("raw_top3_scorelines"):
        sliced["top3_scorelines"] = payload["raw_top3_scorelines"]
    if payload.get("raw_expected_goals"):
        sliced["expected_goals"] = payload["raw_expected_goals"]
    return sliced


def _model_metrics(
    prediction: dict[str, Any],
    *,
    use_raw: bool,
    actual: str,
    home_goals: int,
    away_goals: int,
) -> dict[str, Any]:
    pred = slice_prediction_payload(prediction, use_raw=use_raw)
    probs = pred["result_probs"]
    top3 = pred.get("top3_scorelines") or []
    predicted = _predicted_1x2(probs)
    consistency = _top1_outcome(top3) == predicted if top3 else None
    return {
        "predicted_outcome": predicted,
        "hit_1x2": predicted == actual,
        "hit_top3": _top3_hit(top3, home_goals, away_goals),
        "brier": round(brier_multiclass(probs, actual), 3),
        "rps": round(ranked_probability_score(probs, actual), 3),
        "top1_agrees_1x2": consistency,
    }


def audit_prediction_consistency(payload: dict[str, Any], *, use_raw: bool = False) -> dict[str, Any]:
    pred = slice_prediction_payload(payload, use_raw=use_raw)
    probs = pred.get("result_probs") or {}
    top3 = pred.get("top3_scorelines") or []
    if not probs or not top3:
        return {"ok": False, "reason": "missing_probs_or_top3"}

    predicted_1x2 = _predicted_1x2(probs)
    top1_outcome = _top1_outcome(top3)
    matrix_1x2_sum = round(
        float(probs.get("home_win", 0))
        + float(probs.get("draw", 0))
        + float(probs.get("away_win", 0)),
        4,
    )
    return {
        "ok": top1_outcome == predicted_1x2,
        "predicted_1x2": predicted_1x2,
        "top1_outcome": top1_outcome,
        "top1_score": f"{top3[0]['home_goals']}-{top3[0]['away_goals']}",
        "prob_sum": matrix_1x2_sum,
        "prob_sum_ok": 0.99 <= matrix_1x2_sum <= 1.01,
    }


def audit_pit_row(feature_row: dict[str, Any]) -> dict[str, Any]:
    as_of = str(feature_row.get("as_of_time") or "")
    kickoff = str(feature_row.get("kickoff_ts") or "")
    alerts: list[str] = []
    if as_of and kickoff and as_of > kickoff:
        alerts.append("as_of_after_kickoff")
    if feature_row.get("home_score_ft") is not None and feature_row.get("away_score_ft") is not None:
        alerts.append("feature_row_contains_final_score")
    return {
        "as_of_time": as_of,
        "kickoff_ts": kickoff,
        "pit_ok": not alerts,
        "alerts": alerts,
    }


def evaluate_played_match(
    *,
    feature_row: dict[str, Any],
    prediction: dict[str, Any],
    home_goals: int,
    away_goals: int,
) -> dict[str, Any]:
    actual = actual_result(home_goals, away_goals)
    home_elo = float(feature_row.get("home_elo") or 1500)
    away_elo = float(feature_row.get("away_elo") or 1500)
    home_rank = float(feature_row.get("home_fifa_rank") or 100)
    away_rank = float(feature_row.get("away_fifa_rank") or 100)
    pit = audit_pit_row(feature_row)

    return {
        "match_id": feature_row.get("match_id"),
        "actual_outcome": actual,
        "raw": _model_metrics(
            prediction,
            use_raw=True,
            actual=actual,
            home_goals=home_goals,
            away_goals=away_goals,
        ),
        "adjusted": _model_metrics(
            prediction,
            use_raw=False,
            actual=actual,
            home_goals=home_goals,
            away_goals=away_goals,
        ),
        "baselines": {
            "elo_favorite_hit": baseline_hit_1x2(
                baseline_elo_favorite(home_elo, away_elo),
                home_goals,
                away_goals,
            ),
            "fifa_favorite_hit": baseline_hit_1x2(
                baseline_fifa_favorite(home_rank, away_rank),
                home_goals,
                away_goals,
            ),
            "market_favorite_hit": baseline_hit_1x2(
                baseline_market_favorite(feature_row),
                home_goals,
                away_goals,
            ),
            "naive_top3_hit": hit_naive_top3(
                home_goals,
                away_goals,
                templates=NAIVE_TOP3_SCORELINES,
            ),
            "naive_top6_hit": hit_naive_top3(
                home_goals,
                away_goals,
                templates=NAIVE_TOP6_SCORELINES,
            ),
        },
        "pit_ok": pit["pit_ok"],
        "pit_alerts": pit["alerts"],
        "raw_consistency": audit_prediction_consistency(prediction, use_raw=True),
        "adjusted_consistency": audit_prediction_consistency(prediction, use_raw=False),
    }


def _mean_bool(values: list[bool | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(1 for v in clean if v) / len(clean)


def _summarize_branch(rows: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "hit_1x2": _mean_bool([r[branch]["hit_1x2"] for r in rows]),
        "hit_top3": _mean_bool([r[branch]["hit_top3"] for r in rows]),
        "top1_agrees_1x2": _mean_bool([r[branch].get("top1_agrees_1x2") for r in rows]),
        "avg_brier": sum(r[branch]["brier"] for r in rows) / len(rows),
    }


def summarize_evaluations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}

    baselines = {
        "elo_favorite_hit": _mean_bool([r["baselines"]["elo_favorite_hit"] for r in rows]),
        "fifa_favorite_hit": _mean_bool([r["baselines"]["fifa_favorite_hit"] for r in rows]),
        "market_favorite_hit": _mean_bool(
            [r["baselines"]["market_favorite_hit"] for r in rows]
        ),
        "naive_top3_hit": _mean_bool([r["baselines"]["naive_top3_hit"] for r in rows]),
        "naive_top6_hit": _mean_bool([r["baselines"]["naive_top6_hit"] for r in rows]),
    }
    return {
        "count": len(rows),
        "raw": _summarize_branch(rows, "raw"),
        "adjusted": _summarize_branch(rows, "adjusted"),
        "baselines": baselines,
        "pit_ok_rate": _mean_bool([r.get("pit_ok") for r in rows]),
        "raw_top1_consistent_rate": _mean_bool(
            [r.get("raw_consistency", {}).get("ok") for r in rows]
        ),
        "adjusted_top1_consistent_rate": _mean_bool(
            [r.get("adjusted_consistency", {}).get("ok") for r in rows]
        ),
    }
