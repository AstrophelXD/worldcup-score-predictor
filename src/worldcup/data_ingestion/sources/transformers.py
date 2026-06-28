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
    team_col = _pick_column(df, ["team_name", "team", "country", "country_full", "nation"])
    rank_col = _pick_column(df, ["rank", "ranking"])
    points_col = _pick_column(df, ["points", "total_points", "avg_points"])

    if not all([date_col, team_col, points_col]):
        raise ValueError("fifa_rankings missing required columns: date/team/points")

    rows: list[dict] = []
    for record in df.to_dict(orient="records"):
        rows.append(
            {
                "team_name": str(record[team_col]).strip(),
                "ranking_date": pd.to_datetime(record[date_col]).date().isoformat(),
                "rank": (
                    int(record[rank_col])
                    if rank_col and pd.notna(record.get(rank_col))
                    else None
                ),
                "points": float(record[points_col]),
            }
        )
    out = pd.DataFrame(rows)
    out = out.loc[out["points"].notna()].copy()
    computed_rank = (
        out.groupby("ranking_date")["points"]
        .rank(method="min", ascending=False)
    )
    if rank_col:
        out["rank"] = pd.to_numeric(out["rank"], errors="coerce").fillna(computed_rank)
    else:
        out["rank"] = computed_rank
    out["rank"] = out["rank"].round().astype(int)
    return out


def transform_eloratings_world_tsv(df: pd.DataFrame) -> pd.DataFrame:
    """Parse headerless World.tsv from eloratings.net (rank, rank, code, elo, ...)."""
    if df.shape[1] < 4:
        raise ValueError("eloratings_world_tsv expects at least 4 columns")

    rating_date = pd.Timestamp.utcnow().date().isoformat()
    rows: list[dict] = []
    for record in df.itertuples(index=False):
        values = list(record)
        team_name = str(values[2]).strip()
        rating = float(values[3])
        rows.append(
            {
                "team_name": team_name,
                "rating": round(rating, 1),
                "rating_date": rating_date,
                "rating_system": "elo_eloratings_net",
                "rank": int(values[0]) if pd.notna(values[0]) else None,
            }
        )
    return pd.DataFrame(rows)


def transform_football_data_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Transform football-data.co.uk match odds export to canonical odds schema."""
    date_col = _pick_column(df, ["date", "match_date"])
    home_col = _pick_column(df, ["hometeam", "home_team", "home_team_name", "home"])
    away_col = _pick_column(df, ["awayteam", "away_team", "away_team_name", "away"])
    home_odds_col = _pick_column(df, ["b365h", "bwh", "home_odds", "avg_home"])
    draw_odds_col = _pick_column(df, ["b365d", "bwd", "draw_odds", "avg_draw"])
    away_odds_col = _pick_column(df, ["b365a", "bwa", "away_odds", "avg_away"])
    over_col = _pick_column(df, ["avg>2.5", "b365>2.5", "over25_odds"])
    under_col = _pick_column(df, ["avg<2.5", "b365<2.5", "under25_odds"])

    if not all([date_col, home_col, away_col, home_odds_col, draw_odds_col, away_odds_col]):
        raise ValueError(
            "football_data_odds missing required columns: date/home/away/home_odds/draw/away"
        )

    rows: list[dict] = []
    for idx, record in enumerate(df.to_dict(orient="records")):
        match_date = pd.to_datetime(record[date_col]).date()
        home = str(record[home_col]).strip()
        away = str(record[away_col]).strip()
        kickoff = pd.Timestamp(match_date, tz="UTC")
        match_id = f"fd_{match_date.isoformat()}_{_slug(home)}_{_slug(away)}_{idx}"
        rows.append(
            {
                "match_id": match_id,
                "snapshot_ts": (kickoff - pd.Timedelta(hours=3)).isoformat(),
                "home_odds": float(record[home_odds_col]),
                "draw_odds": float(record[draw_odds_col]),
                "away_odds": float(record[away_odds_col]),
                "over25_odds": float(record[over_col]) if over_col and pd.notna(record.get(over_col)) else 2.05,
                "under25_odds": float(record[under_col]) if under_col and pd.notna(record.get(under_col)) else 1.78,
                "btts_yes_odds": 1.90,
            }
        )
    return pd.DataFrame(rows)


def transform_statsbomb_team_match(df: pd.DataFrame) -> pd.DataFrame:
    """Transform StatsBomb-style team match summary CSV to canonical team_match_stats."""
    match_id_col = _pick_column(df, ["match_id"])
    team_col = _pick_column(df, ["team_name", "team", "team_id"])
    date_col = _pick_column(df, ["match_date", "date"])
    xg_col = _pick_column(df, ["xg", "xG", "team_xg"])
    shots_col = _pick_column(df, ["shots", "total_shots"])
    sot_col = _pick_column(df, ["shots_on_target", "on_target"])
    poss_col = _pick_column(df, ["possession", "poss"])
    yellow_col = _pick_column(df, ["yellow_cards", "yellows"])
    red_col = _pick_column(df, ["red_cards", "reds"])

    if not all([match_id_col, team_col]):
        raise ValueError("statsbomb_team_match missing required columns: match_id/team")

    rows: list[dict] = []
    for idx, record in enumerate(df.to_dict(orient="records")):
        match_id = str(record[match_id_col]).strip()
        team_id = str(record[team_col]).strip()
        match_date = (
            pd.to_datetime(record[date_col]).date().isoformat()
            if date_col and pd.notna(record.get(date_col))
            else pd.Timestamp.utcnow().date().isoformat()
        )
        yellow = int(record[yellow_col]) if yellow_col and pd.notna(record.get(yellow_col)) else 0
        red = int(record[red_col]) if red_col and pd.notna(record.get(red_col)) else 0
        rows.append(
            {
                "team_match_stat_id": f"sb_{match_id}_{_slug(team_id)}_{idx}",
                "match_id": match_id,
                "team_id": team_id,
                "match_date": match_date,
                "possession": float(record[poss_col]) if poss_col and pd.notna(record.get(poss_col)) else None,
                "shots": int(record[shots_col]) if shots_col and pd.notna(record.get(shots_col)) else 0,
                "shots_on_target": int(record[sot_col]) if sot_col and pd.notna(record.get(sot_col)) else 0,
                "xg": float(record[xg_col]) if xg_col and pd.notna(record.get(xg_col)) else None,
                "passes_completed": None,
                "corners": None,
                "fouls": None,
                "yellow_cards": yellow,
                "red_cards": red,
                "cards": yellow + red,
            }
        )
    return pd.DataFrame(rows)
