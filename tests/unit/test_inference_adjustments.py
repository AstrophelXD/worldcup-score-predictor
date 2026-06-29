from worldcup.inference.market_odds import (
    american_to_decimal,
    implied_probs_from_odds_row,
    max_result_divergence,
    normalize_probs,
)
from worldcup.inference.postprocess import apply_prediction_adjustments, is_neutral_venue
from worldcup.features.tabular import enrich_tabular_features
import pandas as pd


def test_american_odds_to_implied_fox_sa_can():
    row = {
        "home_odds": 280,
        "draw_odds": 240,
        "away_odds": -140,
        "over25_odds": 138,
        "under25_odds": -158,
        "btts_yes_odds": 1.95,
        "btts_no_odds": 1.85,
        "odds_format": "american",
    }
    implied = implied_probs_from_odds_row(row)
    result = implied["result_probs"]
    assert result["away_win"] > result["home_win"]
    assert result["away_win"] > 0.45
    assert result["home_win"] < 0.30
    assert implied["ou25_probs"]["under_2_5"] > implied["ou25_probs"]["over_2_5"]


def test_fifa_rank_diff_direction():
    df = pd.DataFrame(
        [
            {
                "home_elo": 1800,
                "away_elo": 1700,
                "home_fifa_rank": 23,
                "away_fifa_rank": 32,
                "home_fifa_points": 1500,
                "away_fifa_points": 1400,
                "home_goals_for_last5": 6,
                "away_goals_for_last5": 4,
                "home_goals_against_last5": 3,
                "away_goals_against_last5": 5,
            }
        ]
    )
    enriched = enrich_tabular_features(df)
    assert enriched.loc[0, "fifa_rank_diff"] > 0


def test_postprocess_moves_knockout_toward_market():
    from worldcup.inference.decoder import PredictionOutput, Scoreline, decode_score_matrix
    from worldcup.inference.predictor import MatchPrediction
    from worldcup.models.score_matrix import apply_dixon_coles_adjustment, independent_score_matrix

    matrix, overflow = independent_score_matrix(1.85, 1.31, 7)
    matrix = apply_dixon_coles_adjustment(matrix, -0.13, 1.85, 1.31)
    output = decode_score_matrix(matrix, overflow)
    prediction = MatchPrediction(
        match_id="wc2026_m073",
        lambda_home=1.85,
        lambda_away=1.31,
        output=output,
    )
    row = pd.Series(
        {
            "is_world_cup": True,
            "is_knockout": True,
            "market_odds_available": 1.0,
            "market_home_implied": 0.23,
            "market_draw_implied": 0.26,
            "market_away_implied": 0.51,
            "market_over25_implied": 0.39,
            "market_under25_implied": 0.61,
            "market_odds_source": "fox",
        }
    )
    adjusted, comparison = apply_prediction_adjustments(prediction, row)
    assert is_neutral_venue(row)
    assert comparison["divergence_level"] in {"caution", "warning", "critical"}
    assert adjusted.output.result_probs["away_win"] > prediction.output.result_probs["away_win"]
    assert adjusted.output.ou25_probs["over_2_5"] < prediction.output.ou25_probs["over_2_5"]


def test_draftkings_netherlands_morocco_implied():
    row = {
        "home_odds": 115,
        "draw_odds": 230,
        "away_odds": 260,
        "over25_odds": 125,
        "under25_odds": -155,
        "btts_yes_odds": 1.90,
        "btts_no_odds": 1.90,
        "odds_format": "american",
    }
    implied = implied_probs_from_odds_row(row)
    result = implied["result_probs"]
    assert 0.42 <= result["home_win"] <= 0.48
    assert 0.24 <= result["away_win"] <= 0.30
    assert 0.26 <= result["draw"] <= 0.32
    assert implied["ou25_probs"]["under_2_5"] > implied["ou25_probs"]["over_2_5"]


def test_divergence_levels():
    from worldcup.inference.postprocess import _divergence_level

    assert _divergence_level(0.10) == "ok"
    assert _divergence_level(0.16) == "caution"
    assert _divergence_level(0.22) == "warning"


def test_netherlands_style_overconfidence_is_tempered():
    from worldcup.inference.decoder import decode_score_matrix
    from worldcup.inference.predictor import MatchPrediction
    from worldcup.models.score_matrix import apply_dixon_coles_adjustment, independent_score_matrix

    matrix, overflow = independent_score_matrix(1.99, 0.86, 7)
    matrix = apply_dixon_coles_adjustment(matrix, -0.13, 1.99, 0.86)
    output = decode_score_matrix(matrix, overflow)
    prediction = MatchPrediction(
        match_id="wc2026_m075",
        lambda_home=1.99,
        lambda_away=0.86,
        output=output,
    )
    row = pd.Series(
        {
            "is_world_cup": True,
            "is_knockout": True,
            "market_odds_available": 1.0,
            "market_home_implied": 0.445,
            "market_draw_implied": 0.290,
            "market_away_implied": 0.265,
            "market_over25_implied": 0.42,
            "market_under25_implied": 0.58,
            "market_odds_source": "draftkings",
        }
    )
    adjusted, comparison = apply_prediction_adjustments(prediction, row)
    result = adjusted.output.result_probs
    assert comparison["divergence_level"] in {"caution", "warning", "critical"}
    assert 0.44 <= result["home_win"] <= 0.56
    assert 0.20 <= result["away_win"] <= 0.28
    assert 0.25 <= result["draw"] <= 0.33
    assert adjusted.output.ou25_probs["under_2_5"] >= 0.50
    assert adjusted.output.expected_goals["total"] <= 2.65


def test_iraq_norway_market_favorite_not_inverted():
    from worldcup.inference.decoder import decode_score_matrix
    from worldcup.inference.predictor import MatchPrediction
    from worldcup.models.score_matrix import apply_dixon_coles_adjustment, independent_score_matrix

    # After strength fix Norway should be favored even before market blend
    matrix, overflow = independent_score_matrix(1.0, 2.2, 7)
    matrix = apply_dixon_coles_adjustment(matrix, -0.13, 1.0, 2.2)
    output = decode_score_matrix(matrix, overflow)
    prediction = MatchPrediction(
        match_id="wc2026_m018",
        lambda_home=1.0,
        lambda_away=2.2,
        output=output,
    )
    row = pd.Series(
        {
            "is_world_cup": True,
            "is_knockout": False,
            "home_elo": 1620,
            "away_elo": 1865,
            "home_fifa_rank": 58,
            "away_fifa_rank": 11,
            "home_avg_player_rating": 71.5,
            "away_avg_player_rating": 84.0,
            "market_odds_available": 1.0,
            "market_home_implied": 0.06,
            "market_draw_implied": 0.13,
            "market_away_implied": 0.81,
            "market_over25_implied": 0.42,
            "market_under25_implied": 0.58,
            "market_odds_source": "draftkings",
        }
    )
    adjusted, comparison = apply_prediction_adjustments(prediction, row)
    result = adjusted.output.result_probs
    assert comparison["divergence_level"] in {"caution", "warning", "critical"}
    assert "likely_team_mapping_bug" not in comparison.get("alerts", [])
    assert result["away_win"] > result["home_win"]
    assert result["away_win"] >= 0.72
    assert adjusted.output.expected_goals["away"] >= 2.0
    assert adjusted.output.expected_goals["away"] > adjusted.output.expected_goals["home"]


def test_iraq_norway_inverted_raw_model_calibrated_from_feature_mart():
    from pathlib import Path

    import pytest

    from worldcup.inference.factory import load_predictor
    from worldcup.inference.postprocess import apply_prediction_adjustments
    from worldcup.utils.paths import project_root

    mart = project_root() / "data" / "feature_mart" / "match_features.parquet"
    if not mart.exists():
        pytest.skip("feature mart not built")

    row = pd.read_parquet(mart)
    match = row.loc[row["match_id"] == "wc2026_m018"]
    if match.empty:
        pytest.skip("wc2026_m018 not in feature mart")

    r = match.iloc[0]
    assert float(r["away_elo"]) > float(r["home_elo"])
    predictor = load_predictor("scoregen_football")
    raw = predictor.predict_row(r)
    adjusted, comparison = apply_prediction_adjustments(raw, r)
    result = adjusted.output.result_probs

    assert comparison["divergence_level"] == "critical"
    assert "likely_team_mapping_bug" in comparison.get("alerts", [])
    assert result["away_win"] > result["home_win"]
    assert result["away_win"] >= 0.72
    assert adjusted.output.expected_goals["away"] >= 2.0
    assert comparison["market_blend_weight"] >= 0.80
