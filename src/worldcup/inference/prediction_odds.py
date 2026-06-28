"""Convert model predictions into market-style odds rows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from worldcup.inference.predictor import MatchPrediction


def _prob_to_decimal_odds(probability: float, *, floor: float = 0.001) -> float:
    return round(max(1.01, 1.0 / max(probability, floor)), 2)


def prediction_to_odds_row(
    match_id: str,
    kickoff_ts: str | pd.Timestamp,
    prediction: MatchPrediction,
) -> dict[str, Any]:
    payload = prediction.to_dict()
    kickoff = pd.Timestamp(kickoff_ts)
    if kickoff.tzinfo is None:
        kickoff = kickoff.tz_localize("UTC")
    snapshot = (kickoff - pd.Timedelta(hours=3)).isoformat()

    result = payload["result_probs"]
    ou = payload["ou25_probs"]
    btts = payload["btts_probs"]

    return {
        "match_id": match_id,
        "snapshot_ts": snapshot,
        "home_odds": _prob_to_decimal_odds(float(result["home_win"])),
        "draw_odds": _prob_to_decimal_odds(float(result["draw"])),
        "away_odds": _prob_to_decimal_odds(float(result["away_win"])),
        "over25_odds": _prob_to_decimal_odds(float(ou["over_2_5"])),
        "under25_odds": _prob_to_decimal_odds(float(ou["under_2_5"])),
        "btts_yes_odds": _prob_to_decimal_odds(float(btts["yes"])),
    }
