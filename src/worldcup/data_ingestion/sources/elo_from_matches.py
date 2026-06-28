"""Derive national-team Elo time series from finished international matches."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute_elo_from_matches(
    matches: pd.DataFrame,
    *,
    initial_rating: float = 1500.0,
    k_factor: float = 20.0,
    home_advantage: float = 65.0,
) -> pd.DataFrame:
    """Walk chronological matches and emit PIT-friendly elo rows after each update."""
    if matches.empty:
        return pd.DataFrame(
            columns=["team_name", "rating", "rating_date", "rating_system", "rank"]
        )

    frame = matches.copy()
    frame["kickoff_ts"] = pd.to_datetime(frame["kickoff_ts"], utc=True, errors="coerce")
    frame = frame.sort_values("kickoff_ts").reset_index(drop=True)

    ratings: dict[str, float] = defaultdict(lambda: initial_rating)
    rows: list[dict] = []

    for record in frame.to_dict(orient="records"):
        home = str(record["home_team_name"]).strip()
        away = str(record["away_team_name"]).strip()
        home_score = record.get("home_score_ft")
        away_score = record.get("away_score_ft")
        if pd.isna(home_score) or pd.isna(away_score):
            continue

        hs = int(home_score)
        aws = int(away_score)
        if hs > aws:
            home_score_actual, away_score_actual = 1.0, 0.0
        elif hs < aws:
            home_score_actual, away_score_actual = 0.0, 1.0
        else:
            home_score_actual, away_score_actual = 0.5, 0.5

        home_rating = ratings[home]
        away_rating = ratings[away]
        home_effective = home_rating + home_advantage
        expected_home = _expected_score(home_effective, away_rating)
        expected_away = 1.0 - expected_home

        ratings[home] = home_rating + k_factor * (home_score_actual - expected_home)
        ratings[away] = away_rating + k_factor * (away_score_actual - expected_away)

        rating_date = pd.Timestamp(record["kickoff_ts"]).date().isoformat()
        for team_name, rating in ((home, ratings[home]), (away, ratings[away])):
            rows.append(
                {
                    "team_name": team_name,
                    "rating": round(rating, 1),
                    "rating_date": rating_date,
                    "rating_system": "elo_match_derived",
                    "rank": None,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["team_name", "rating", "rating_date", "rating_system", "rank"]
        )
    return pd.DataFrame(rows)
