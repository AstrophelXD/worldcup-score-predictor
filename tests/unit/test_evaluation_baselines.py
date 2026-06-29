from worldcup.evaluation.baselines import (
    baseline_elo_favorite,
    baseline_fifa_favorite,
    hit_naive_top3,
    NAIVE_TOP3_SCORELINES,
)
from worldcup.evaluation.match_eval import audit_prediction_consistency, slice_prediction_payload


def test_baseline_elo_favorite():
    assert baseline_elo_favorite(1900, 1600) == "home_win"
    assert baseline_elo_favorite(1600, 1900) == "away_win"
    assert baseline_elo_favorite(1700, 1710) == "draw"


def test_naive_top3_template():
    assert hit_naive_top3(1, 1, templates=NAIVE_TOP3_SCORELINES)
    assert not hit_naive_top3(4, 3, templates=NAIVE_TOP3_SCORELINES)


def test_raw_vs_adjusted_slice():
    payload = {
        "result_probs": {"home_win": 0.6, "draw": 0.2, "away_win": 0.2},
        "top3_scorelines": [{"home_goals": 2, "away_goals": 0, "prob": 0.2}],
        "raw_result_probs": {"home_win": 0.3, "draw": 0.3, "away_win": 0.4},
        "raw_top3_scorelines": [{"home_goals": 0, "away_goals": 1, "prob": 0.15}],
    }
    raw = slice_prediction_payload(payload, use_raw=True)
    assert raw["result_probs"]["away_win"] == 0.4
    assert raw["top3_scorelines"][0]["home_goals"] == 0


def test_top1_consistency_from_same_matrix():
    payload = {
        "result_probs": {"home_win": 0.5, "draw": 0.25, "away_win": 0.25},
        "top3_scorelines": [
            {"home_goals": 1, "away_goals": 0, "prob": 0.2},
            {"home_goals": 1, "away_goals": 1, "prob": 0.15},
            {"home_goals": 2, "away_goals": 0, "prob": 0.12},
        ],
    }
    audit = audit_prediction_consistency(payload, use_raw=False)
    assert audit["ok"] is True
    assert audit["prob_sum_ok"] is True
