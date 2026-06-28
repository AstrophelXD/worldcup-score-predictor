from worldcup.inference.factory import load_predictor
from worldcup.inference.prediction_odds import prediction_to_odds_row
from worldcup.utils.paths import project_root
import pandas as pd


def test_prediction_to_odds_row_from_model():
    mart = project_root() / "data" / "feature_mart" / "match_features.parquet"
    if not mart.exists():
        return
    features = pd.read_parquet(mart)
    if "wc2026_m001" not in set(features["match_id"].astype(str)):
        return
    predictor = load_predictor()
    prediction = predictor.predict_match_id(features, "wc2026_m001")
    row = features.loc[features["match_id"] == "wc2026_m001"].iloc[0]
    odds = prediction_to_odds_row("wc2026_m001", row["kickoff_ts"], prediction)
    assert odds["match_id"] == "wc2026_m001"
    assert odds["home_odds"] >= 1.01
    assert odds["draw_odds"] >= 1.01
    assert odds["away_odds"] >= 1.01
