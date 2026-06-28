from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException

from worldcup.data_ingestion.base import read_parquet
from worldcup.inference.factory import (
    checkpoint_metadata,
    default_model_name,
    load_predictor,
)
from worldcup.inference.predictor import MatchPrediction
from worldcup.models.registry import latest_checkpoint
from worldcup.utils.paths import project_root

FEATURE_MART = project_root() / "data" / "feature_mart" / "match_features.parquet"
CURATED_TEAMS = project_root() / "data" / "curated" / "teams.parquet"
CHECKPOINT_DIR = project_root() / "artifacts" / "checkpoints"
BACKTEST_DIR = project_root() / "artifacts" / "backtests"


def _json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


@lru_cache(maxsize=1)
def _team_names() -> dict[str, str]:
    if not CURATED_TEAMS.exists():
        return {}
    teams = read_parquet(str(CURATED_TEAMS))
    if not isinstance(teams, pd.DataFrame):
        teams = pd.DataFrame(teams)
    return dict(zip(teams["team_id"], teams["team_name"], strict=True))


def load_features() -> pd.DataFrame:
    if not FEATURE_MART.exists():
        raise HTTPException(status_code=503, detail="feature mart not built")
    df = read_parquet(str(FEATURE_MART))
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)


def get_predictor(model_name: str | None = None):
    try:
        return load_predictor(model_name=model_name, checkpoint_dir=CHECKPOINT_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def checkpoint_info(model_name: str | None = None) -> dict[str, Any]:
    name = model_name or default_model_name()
    return checkpoint_metadata(name, CHECKPOINT_DIR)


def list_matches(limit: int = 100, world_cup_only: bool = False) -> list[dict[str, Any]]:
    features = load_features()
    names = _team_names()
    if world_cup_only:
        features = features.loc[features["is_world_cup"].astype(bool)]
    items = features.sort_values("kickoff_ts", ascending=False).head(limit)
    payload: list[dict[str, Any]] = []
    for row in items.itertuples(index=False):
        payload.append(
            {
                "match_id": row.match_id,
                "kickoff_ts": row.kickoff_ts,
                "home_team_id": row.home_team_id,
                "away_team_id": row.away_team_id,
                "home_team_name": names.get(row.home_team_id, row.home_team_id),
                "away_team_name": names.get(row.away_team_id, row.away_team_id),
                "stage": row.stage_name,
                "is_world_cup": bool(row.is_world_cup),
                "is_knockout": bool(row.is_knockout),
            }
        )
    return payload


def get_match(match_id: str) -> dict[str, Any]:
    features = load_features()
    rows = features.loc[features["match_id"] == match_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"match not found: {match_id}")
    row = rows.iloc[0]
    names = _team_names()
    return {
        "match_id": match_id,
        "kickoff_ts": row["kickoff_ts"],
        "competition_name": row["competition_name"],
        "stage": row["stage_name"],
        "home_team_id": row["home_team_id"],
        "away_team_id": row["away_team_id"],
        "home_team_name": names.get(row["home_team_id"], row["home_team_id"]),
        "away_team_name": names.get(row["away_team_id"], row["away_team_id"]),
        "is_world_cup": bool(row["is_world_cup"]),
        "is_knockout": bool(row["is_knockout"]),
        "home_score_ft": None if pd.isna(row.get("home_score_ft")) else int(row["home_score_ft"]),
        "away_score_ft": None if pd.isna(row.get("away_score_ft")) else int(row["away_score_ft"]),
    }


def get_features(match_id: str) -> dict[str, Any]:
    features = load_features()
    rows = features.loc[features["match_id"] == match_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"match not found: {match_id}")
    row = rows.iloc[0]
    return {
        "match_id": match_id,
        "as_of_time": str(row["as_of_time"]),
        "team_strength": {
            "home_elo": _json_value(row.get("home_elo")),
            "away_elo": _json_value(row.get("away_elo")),
            "home_fifa_rank": _json_value(row.get("home_fifa_rank")),
            "away_fifa_rank": _json_value(row.get("away_fifa_rank")),
            "home_fifa_points": _json_value(row.get("home_fifa_points")),
            "away_fifa_points": _json_value(row.get("away_fifa_points")),
        },
        "recent_form": {
            "home_goals_for_last5": _json_value(row.get("home_goals_for_last5")),
            "away_goals_for_last5": _json_value(row.get("away_goals_for_last5")),
            "home_goals_against_last5": _json_value(row.get("home_goals_against_last5")),
            "away_goals_against_last5": _json_value(row.get("away_goals_against_last5")),
        },
        "rest_days": {
            "home_rest_days": _json_value(row.get("home_rest_days")),
            "away_rest_days": _json_value(row.get("away_rest_days")),
        },
    }


def predict_match(match_id: str, model_name: str | None = None) -> MatchPrediction:
    features = load_features()
    predictor = get_predictor(model_name)
    try:
        return predictor.predict_match_id(features, match_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def file_mtime(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return ts.isoformat()


def data_freshness(model_name: str | None = None) -> list[dict[str, Any]]:
    name = model_name or default_model_name()
    checkpoint_path = latest_checkpoint(CHECKPOINT_DIR, name)
    curated_matches = project_root() / "data" / "curated" / "matches.parquet"
    return [
        {
            "source": "feature_mart",
            "path": str(FEATURE_MART),
            "updated_at": file_mtime(FEATURE_MART),
        },
        {
            "source": "checkpoint",
            "model_name": name,
            "path": str(checkpoint_path) if checkpoint_path else None,
            "updated_at": file_mtime(checkpoint_path),
        },
        {
            "source": "curated_matches",
            "path": str(curated_matches),
            "updated_at": file_mtime(curated_matches),
        },
    ]


def list_backtest_runs() -> list[dict[str, Any]]:
    if not BACKTEST_DIR.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(BACKTEST_DIR.glob("backtest_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        items.append(
            {
                "run_id": path.stem,
                "test_scope": payload.get("test_scope"),
                "report_path": str(path),
                "score_nll": metrics.get("score_nll"),
                "top3_hit_rate": metrics.get("top3_hit_rate"),
                "match_count": metrics.get("match_count"),
                "generated_at": payload.get("generated_at"),
            }
        )
    return items


def get_backtest_run(run_id: str) -> dict[str, Any]:
    path = BACKTEST_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"backtest run not found: {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))
