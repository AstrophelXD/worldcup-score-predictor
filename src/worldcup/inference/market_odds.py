"""Convert bookmaker odds to normalized implied probabilities."""

from __future__ import annotations


def american_to_decimal(odds: float) -> float:
    if odds >= 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def decimal_to_implied(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError(f"invalid decimal odds: {decimal_odds}")
    return 1.0 / decimal_odds


def implied_from_odds_value(odds: float, *, odds_format: str = "decimal") -> float:
    fmt = odds_format.strip().lower()
    if fmt in {"american", "us", "moneyline"}:
        return decimal_to_implied(american_to_decimal(odds))
    return decimal_to_implied(float(odds))


def normalize_probs(probs: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in probs.values())
    if total <= 0:
        raise ValueError("cannot normalize empty probability vector")
    return {key: max(0.0, float(value)) / total for key, value in probs.items()}


def implied_probs_from_odds_row(row: dict) -> dict[str, float]:
    odds_format = str(row.get("odds_format", "decimal")).lower()
    raw = {
        "home_win": implied_from_odds_value(float(row["home_odds"]), odds_format=odds_format),
        "draw": implied_from_odds_value(float(row["draw_odds"]), odds_format=odds_format),
        "away_win": implied_from_odds_value(float(row["away_odds"]), odds_format=odds_format),
    }
    result = normalize_probs(raw)
    over_raw = implied_from_odds_value(float(row.get("over25_odds", 2.0)), odds_format=odds_format)
    under_raw = implied_from_odds_value(float(row.get("under25_odds", 1.9)), odds_format=odds_format)
    ou = normalize_probs({"over_2_5": over_raw, "under_2_5": under_raw})
    btts_yes_raw = implied_from_odds_value(float(row.get("btts_yes_odds", 1.9)), odds_format=odds_format)
    btts_no_raw = implied_from_odds_value(float(row.get("btts_no_odds", 1.9)), odds_format=odds_format)
    btts = normalize_probs({"yes": btts_yes_raw, "no": btts_no_raw})
    return {
        "result_probs": result,
        "ou25_probs": ou,
        "btts_probs": btts,
    }


def max_result_divergence(model: dict[str, float], market: dict[str, float]) -> float:
    keys = ("home_win", "draw", "away_win")
    return max(abs(float(model[key]) - float(market[key])) for key in keys)
