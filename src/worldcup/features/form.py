from __future__ import annotations

import pandas as pd


def team_match_history(
    matches_df: pd.DataFrame,
    team_id: str,
    as_of_time: pd.Timestamp,
) -> pd.DataFrame:
    past = matches_df.loc[
        pd.to_datetime(matches_df["kickoff_ts"], utc=True) < as_of_time
    ].copy()
    home = past.loc[past["home_team_id"] == team_id].copy()
    away = past.loc[past["away_team_id"] == team_id].copy()

    home["goals_for"] = home["home_score_ft"]
    home["goals_against"] = home["away_score_ft"]
    away["goals_for"] = away["away_score_ft"]
    away["goals_against"] = away["home_score_ft"]

    history = pd.concat(
        [
            home[
                ["match_id", "kickoff_ts", "goals_for", "goals_against", "is_world_cup"]
            ],
            away[
                ["match_id", "kickoff_ts", "goals_for", "goals_against", "is_world_cup"]
            ],
        ],
        ignore_index=True,
    )
    return history.sort_values("kickoff_ts")


def rolling_form(history: pd.DataFrame, windows: list[int]) -> dict[str, float | int | None]:
    features: dict[str, float | int | None] = {}
    for window in windows:
        recent = history.tail(window)
        prefix = f"last{window}"
        if recent.empty:
            features[f"goals_for_{prefix}"] = None
            features[f"goals_against_{prefix}"] = None
            features[f"matches_{prefix}"] = 0
            continue
        features[f"goals_for_{prefix}"] = float(recent["goals_for"].sum())
        features[f"goals_against_{prefix}"] = float(recent["goals_against"].sum())
        features[f"matches_{prefix}"] = int(len(recent))
    return features


def rest_days(history: pd.DataFrame, as_of_time: pd.Timestamp) -> float | None:
    if history.empty:
        return None
    last_kickoff = pd.to_datetime(history.iloc[-1]["kickoff_ts"], utc=True)
    delta = as_of_time - last_kickoff
    return float(delta.total_seconds() / 86400.0)
