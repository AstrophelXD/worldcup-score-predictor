from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from worldcup.data_ingestion.base import read_parquet, write_parquet
from worldcup.features.form import rest_days, rolling_form, team_match_history
from worldcup.features.event_features import EVENT_SUMMARY_COLUMNS, build_event_summary_row
from worldcup.features.market_odds_features import build_market_odds_summary_row, load_market_odds_table
from worldcup.features.odds_features import build_odds_summary_row, load_odds_curated
from worldcup.features.player_match_features import PLAYER_SUMMARY_COLUMNS, build_player_summary_row
from worldcup.features.point_in_time import as_of_timestamp
from worldcup.features.strength import latest_rating_before
from worldcup.utils.paths import ensure_dir


@dataclass
class FeatureBuildResult:
    output_path: Path
    row_count: int


def _load_optional_parquet(curated_dir: Path, name: str) -> pd.DataFrame:
    path = curated_dir / name
    if not path.exists():
        return pd.DataFrame()
    return read_parquet(str(path))


def build_match_feature_mart(
    *,
    curated_dir: Path,
    feature_mart_dir: Path,
    form_windows: list[int],
) -> FeatureBuildResult:
    matches = read_parquet(str(curated_dir / "matches.parquet"))
    elo = read_parquet(str(curated_dir / "elo_ratings.parquet"))
    fifa = read_parquet(str(curated_dir / "fifa_rankings.parquet"))
    players = _load_optional_parquet(curated_dir, "players.parquet")
    lineups = _load_optional_parquet(curated_dir, "lineups.parquet")
    stats = _load_optional_parquet(curated_dir, "player_match_stats.parquet")
    injuries = _load_optional_parquet(curated_dir, "injuries.parquet")
    odds = load_odds_curated(curated_dir)
    market_odds = load_market_odds_table()
    team_stats = _load_optional_parquet(curated_dir, "team_match_stats.parquet")

    matches = matches.sort_values("kickoff_ts").reset_index(drop=True)
    lineup_match_ids: set[str] = set()
    if not lineups.empty:
        lineup_match_ids = set(lineups["match_id"].astype(str).unique())
    rows: list[dict] = []

    for match in matches.to_dict(orient="records"):
        as_of = as_of_timestamp(match["kickoff_ts"])
        home_id = match["home_team_id"]
        away_id = match["away_team_id"]

        home_history = team_match_history(matches, home_id, as_of)
        away_history = team_match_history(matches, away_id, as_of)
        home_form = rolling_form(home_history, form_windows)
        away_form = rolling_form(away_history, form_windows)

        home_elo = latest_rating_before(elo, home_id, as_of, "rating_date", ["rating", "rank"])
        away_elo = latest_rating_before(elo, away_id, as_of, "rating_date", ["rating", "rank"])
        home_fifa = latest_rating_before(
            fifa, home_id, as_of, "ranking_date", ["rank", "points"]
        )
        away_fifa = latest_rating_before(
            fifa, away_id, as_of, "ranking_date", ["rank", "points"]
        )

        row = {
            "match_id": match["match_id"],
            "as_of_time": as_of,
            "kickoff_ts": as_of,
            "competition_name": match["competition_name"],
            "stage_name": match.get("stage_name"),
            "is_world_cup": bool(match["is_world_cup"]),
            "is_knockout": bool(match["is_knockout"]),
            "neutral_site_flag": bool(match["is_world_cup"]) and bool(match["is_knockout"]),
            "must_win_flag": bool(match["is_knockout"]),
            "draw_acceptable_flag": not bool(match["is_knockout"]),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_elo": home_elo["rating"],
            "away_elo": away_elo["rating"],
            "home_elo_rank": home_elo["rank"],
            "away_elo_rank": away_elo["rank"],
            "home_fifa_rank": home_fifa["rank"],
            "away_fifa_rank": away_fifa["rank"],
            "home_fifa_points": home_fifa["points"],
            "away_fifa_points": away_fifa["points"],
            "home_rest_days": rest_days(home_history, as_of),
            "away_rest_days": rest_days(away_history, as_of),
            "home_score_ft": match.get("home_score_ft"),
            "away_score_ft": match.get("away_score_ft"),
        }

        for key, value in home_form.items():
            row[f"home_{key}"] = value
        for key, value in away_form.items():
            row[f"away_{key}"] = value

        if str(match["match_id"]) in lineup_match_ids and not players.empty:
            row.update(
                build_player_summary_row(
                    match_id=str(match["match_id"]),
                    home_team_id=str(home_id),
                    away_team_id=str(away_id),
                    as_of_time=pd.Timestamp(as_of),
                    players=players,
                    lineups=lineups,
                    stats=stats,
                    injuries=injuries,
                )
            )
        else:
            for column in PLAYER_SUMMARY_COLUMNS:
                row[column] = 0.0

        row.update(
            build_odds_summary_row(
                odds,
                str(match["match_id"]),
                pd.Timestamp(as_of),
            )
        )
        row.update(
            build_market_odds_summary_row(
                market_odds,
                str(match["match_id"]),
                pd.Timestamp(as_of),
            )
        )
        row.update(
            build_event_summary_row(
                match_id=str(match["match_id"]),
                home_team_id=str(home_id),
                away_team_id=str(away_id),
                as_of_time=pd.Timestamp(as_of),
                team_stats=team_stats,
                matches=matches,
            )
        )

        rows.append(row)

    feature_df = pd.DataFrame(rows)
    ensure_dir(feature_mart_dir)
    output_path = feature_mart_dir / "match_features.parquet"
    write_parquet(feature_df, str(output_path))
    return FeatureBuildResult(output_path=output_path, row_count=len(feature_df))
