"""PIT rolling team event summaries (xG, shots, cards) for the feature mart."""

from __future__ import annotations

import pandas as pd

from worldcup.features.point_in_time import filter_before

EVENT_SUMMARY_COLUMNS = [
    "home_xg_for_last5",
    "away_xg_for_last5",
    "home_xg_against_last5",
    "away_xg_against_last5",
    "home_shots_for_last5",
    "away_shots_for_last5",
    "home_cards_last5",
    "away_cards_last5",
    "home_event_matches_last5",
    "away_event_matches_last5",
    "event_data_available",
]


def _team_event_history(
    team_stats: pd.DataFrame,
    matches: pd.DataFrame,
    team_id: str,
    as_of_time: pd.Timestamp,
) -> pd.DataFrame:
    if team_stats.empty:
        return pd.DataFrame()

    stats = team_stats.loc[team_stats["team_id"] == team_id].copy()
    if stats.empty:
        return pd.DataFrame()

    match_lookup = matches.set_index("match_id")
    stats["kickoff_ts"] = stats["match_id"].map(
        lambda mid: match_lookup.loc[mid, "kickoff_ts"] if mid in match_lookup.index else None
    )
    stats["kickoff_ts"] = pd.to_datetime(stats["kickoff_ts"], utc=True, errors="coerce")
    stats = stats.loc[stats["kickoff_ts"].notna()].copy()
    stats = filter_before(stats, as_of_time.to_pydatetime(), "kickoff_ts")
    if stats.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for record in stats.to_dict(orient="records"):
        match_id = str(record["match_id"])
        if match_id not in match_lookup.index:
            continue
        match = match_lookup.loc[match_id]
        is_home = str(match["home_team_id"]) == team_id
        goals_for = float(match["home_score_ft"] if is_home else match["away_score_ft"] or 0)
        goals_against = float(match["away_score_ft"] if is_home else match["home_score_ft"] or 0)
        xg = float(record.get("xg") or 0.0)
        opp_rows = team_stats.loc[
            (team_stats["match_id"] == match_id) & (team_stats["team_id"] != team_id)
        ]
        xg_against = float(opp_rows.iloc[0]["xg"]) if not opp_rows.empty else goals_against
        yellow = int(record.get("yellow_cards") or 0)
        red = int(record.get("red_cards") or 0)
        cards = int(record.get("cards") or yellow + red)
        rows.append(
            {
                "kickoff_ts": record["kickoff_ts"],
                "xg_for": xg,
                "xg_against": xg_against,
                "shots_for": int(record.get("shots") or 0),
                "cards": cards,
                "goals_for": goals_for,
                "goals_against": goals_against,
            }
        )
    return pd.DataFrame(rows).sort_values("kickoff_ts")


def _rolling_event_summary(history: pd.DataFrame, window: int = 5) -> dict[str, float]:
    if history.empty:
        return {
            "xg_for": 0.0,
            "xg_against": 0.0,
            "shots_for": 0.0,
            "cards": 0.0,
            "matches": 0.0,
        }
    recent = history.tail(window)
    return {
        "xg_for": float(recent["xg_for"].sum()),
        "xg_against": float(recent["xg_against"].sum()),
        "shots_for": float(recent["shots_for"].sum()),
        "cards": float(recent["cards"].sum()),
        "matches": float(len(recent)),
    }


def build_event_summary_row(
    *,
    match_id: str,
    home_team_id: str,
    away_team_id: str,
    as_of_time: pd.Timestamp,
    team_stats: pd.DataFrame,
    matches: pd.DataFrame,
    window: int = 5,
) -> dict[str, float]:
    if team_stats.empty:
        return {column: 0.0 for column in EVENT_SUMMARY_COLUMNS}

    home_hist = _team_event_history(team_stats, matches, home_team_id, as_of_time)
    away_hist = _team_event_history(team_stats, matches, away_team_id, as_of_time)
    home = _rolling_event_summary(home_hist, window)
    away = _rolling_event_summary(away_hist, window)

    has_current = bool(
        not team_stats.empty
        and (team_stats["match_id"] == match_id).any()
    )
    has_history = home["matches"] > 0 or away["matches"] > 0

    return {
        "home_xg_for_last5": home["xg_for"],
        "away_xg_for_last5": away["xg_for"],
        "home_xg_against_last5": home["xg_against"],
        "away_xg_against_last5": away["xg_against"],
        "home_shots_for_last5": home["shots_for"],
        "away_shots_for_last5": away["shots_for"],
        "home_cards_last5": home["cards"],
        "away_cards_last5": away["cards"],
        "home_event_matches_last5": home["matches"],
        "away_event_matches_last5": away["matches"],
        "event_data_available": 1.0 if (has_current or has_history) else 0.0,
    }


def event_targets_for_match(
    team_stats: pd.DataFrame,
    match_id: str,
    home_team_id: str,
    away_team_id: str,
) -> tuple[list[float], bool]:
    """Training labels: [home_xg, away_xg, home_shots, away_shots]."""
    rows = team_stats.loc[team_stats["match_id"] == match_id]
    if rows.empty:
        return [0.0, 0.0, 0.0, 0.0], False

    home_row = rows.loc[rows["team_id"] == home_team_id]
    away_row = rows.loc[rows["team_id"] == away_team_id]
    if home_row.empty or away_row.empty:
        return [0.0, 0.0, 0.0, 0.0], False

    home = home_row.iloc[0]
    away = away_row.iloc[0]
    return (
        [
            float(home.get("xg") or 0.0),
            float(away.get("xg") or 0.0),
            float(home.get("shots") or 0.0),
            float(away.get("shots") or 0.0),
        ],
        True,
    )
