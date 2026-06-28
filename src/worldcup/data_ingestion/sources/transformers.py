from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _infer_world_cup(tournament: str) -> bool:
    text = tournament.lower()
    return "world cup" in text or "fifa world cup" in text


def _infer_knockout(tournament: str, stage_name: str | None) -> bool:
    text = " ".join(filter(None, [tournament, stage_name])).lower()
    keywords = ("final", "semi", "quarter", "round of 16", "knockout", "last 16", "last 32")
    return any(key in text for key in keywords)


def transform_kaggle_international(df: pd.DataFrame) -> pd.DataFrame:
    """Transform Kaggle-style international results CSV to canonical matches schema."""
    date_col = _pick_column(df, ["date", "match_date"])
    home_col = _pick_column(df, ["home_team", "home_team_name", "home"])
    away_col = _pick_column(df, ["away_team", "away_team_name", "away"])
    home_score_col = _pick_column(df, ["home_score", "home_score_ft", "home_goals"])
    away_score_col = _pick_column(df, ["away_score", "away_score_ft", "away_goals"])
    tournament_col = _pick_column(df, ["tournament", "competition_name", "competition"])
    city_col = _pick_column(df, ["city"])
    country_col = _pick_column(df, ["country"])
    neutral_col = _pick_column(df, ["neutral", "is_neutral"])

    missing = [name for name, col in [
        ("date", date_col),
        ("home_team", home_col),
        ("away_team", away_col),
    ] if col is None]
    if missing:
        raise ValueError(f"kaggle_international missing required columns: {missing}")

    rows: list[dict] = []
    for idx, record in enumerate(df.to_dict(orient="records")):
        match_date = pd.to_datetime(record[date_col]).date()
        home = str(record[home_col]).strip()
        away = str(record[away_col]).strip()
        tournament = (
            str(record.get(tournament_col, "International"))
            if tournament_col
            else "International"
        )
        stage_name = str(record[tournament_col]) if tournament_col else tournament
        match_id = f"{_slug(tournament)}_{match_date.isoformat()}_{_slug(home)}_{_slug(away)}_{idx}"
        kickoff_ts = pd.Timestamp(match_date, tz="UTC").isoformat()
        is_world_cup = _infer_world_cup(tournament)
        is_knockout = _infer_knockout(tournament, stage_name)

        home_score = record.get(home_score_col) if home_score_col else None
        away_score = record.get(away_score_col) if away_score_col else None

        rows.append(
            {
                "match_id": match_id,
                "competition_name": tournament,
                "season_name": str(match_date.year),
                "stage_name": stage_name,
                "match_date": match_date.isoformat(),
                "kickoff_ts": kickoff_ts,
                "home_team_name": home,
                "away_team_name": away,
                "home_score_ft": home_score,
                "away_score_ft": away_score,
                "home_score_ht": None,
                "away_score_ht": None,
                "aet_score_home": None,
                "aet_score_away": None,
                "pen_score_home": None,
                "pen_score_away": None,
                "status": "finished",
                "is_world_cup": is_world_cup,
                "is_knockout": is_knockout,
                "venue": None,
                "city": record.get(city_col) if city_col else None,
                "country": record.get(country_col) if country_col else None,
            }
        )
        if neutral_col:
            rows[-1]["is_neutral"] = _parse_bool(record.get(neutral_col))

    return pd.DataFrame(rows)


def transform_elo_history(df: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick_column(df, ["rating_date", "date", "from"])
    team_col = _pick_column(df, ["team_name", "team", "country", "nation"])
    rating_col = _pick_column(df, ["rating", "elo", "rating_value"])
    rank_col = _pick_column(df, ["rank", "ranking"])
    system_col = _pick_column(df, ["rating_system", "system"])

    if not all([date_col, team_col, rating_col]):
        raise ValueError("elo_history missing required columns: date/team/rating")

    rows: list[dict] = []
    for record in df.to_dict(orient="records"):
        rows.append(
            {
                "team_name": str(record[team_col]).strip(),
                "rating": float(record[rating_col]),
                "rating_date": pd.to_datetime(record[date_col]).date().isoformat(),
                "rating_system": str(record.get(system_col, "elo")) if system_col else "elo",
                "rank": (
                    int(record[rank_col])
                    if rank_col and pd.notna(record.get(rank_col))
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def transform_fifa_rankings(df: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick_column(df, ["ranking_date", "rank_date", "date"])
    team_col = _pick_column(df, ["team_name", "country", "country_full", "nation"])
    rank_col = _pick_column(df, ["rank", "ranking"])
    points_col = _pick_column(df, ["points", "total_points", "avg_points"])

    if not all([date_col, team_col, rank_col, points_col]):
        raise ValueError("fifa_rankings missing required columns: date/team/rank/points")

    rows: list[dict] = []
    for record in df.to_dict(orient="records"):
        rows.append(
            {
                "team_name": str(record[team_col]).strip(),
                "ranking_date": pd.to_datetime(record[date_col]).date().isoformat(),
                "rank": int(record[rank_col]),
                "points": float(record[points_col]),
            }
        )
    return pd.DataFrame(rows)
