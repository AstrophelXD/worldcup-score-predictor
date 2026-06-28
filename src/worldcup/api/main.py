from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from worldcup import __version__
from worldcup.data_ingestion.base import read_parquet
from worldcup.inference.predictor import BaselinePredictor
from worldcup.models.registry import latest_checkpoint
from worldcup.utils.paths import project_root

app = FastAPI(title="WorldCup Predictor API", version=__version__)

FEATURE_MART = project_root() / "data" / "feature_mart" / "match_features.parquet"
CHECKPOINT_DIR = project_root() / "artifacts" / "checkpoints"


class PredictRequest(BaseModel):
    match_id: str
    model_version: str | None = None


class ScorelineResponse(BaseModel):
    home_goals: int
    away_goals: int
    prob: float


class PredictResponse(BaseModel):
    match_id: str
    top3_scorelines: list[ScorelineResponse]
    result_probs: dict[str, float]
    ou25_probs: dict[str, float]
    btts_probs: dict[str, float]
    expected_goals: dict[str, float]
    uncertainty: dict[str, float]
    overflow_prob: float
    lambda_home: float
    lambda_away: float


def _load_features() -> pd.DataFrame:
    if not FEATURE_MART.exists():
        raise HTTPException(status_code=503, detail="feature mart not built")
    return read_parquet(str(FEATURE_MART))


def _load_predictor() -> BaselinePredictor:
    checkpoint = latest_checkpoint(CHECKPOINT_DIR, "baseline_dixon_coles")
    if checkpoint is None:
        raise HTTPException(status_code=503, detail="baseline checkpoint not found")
    return BaselinePredictor.from_path(checkpoint)


def _file_mtime(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return ts.isoformat()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/matches")
def list_matches(limit: int = 100) -> dict:
    features = _load_features()
    items = features.sort_values("kickoff_ts", ascending=False).head(limit)
    payload = []
    for row in items.itertuples(index=False):
        payload.append(
            {
                "match_id": row.match_id,
                "kickoff_ts": row.kickoff_ts,
                "home_team_id": row.home_team_id,
                "away_team_id": row.away_team_id,
                "stage": row.stage_name,
                "is_world_cup": bool(row.is_world_cup),
            }
        )
    return {"items": payload}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    features = _load_features()
    predictor = _load_predictor()
    try:
        prediction = predictor.predict_match_id(features, request.match_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    payload = prediction.to_dict()
    return PredictResponse(
        match_id=payload["match_id"],
        top3_scorelines=[ScorelineResponse(**item) for item in payload["top3_scorelines"]],
        result_probs=payload["result_probs"],
        ou25_probs=payload["ou25_probs"],
        btts_probs=payload["btts_probs"],
        expected_goals=payload["expected_goals"],
        uncertainty=payload["uncertainty"],
        overflow_prob=payload["overflow_prob"],
        lambda_home=payload["lambda_home"],
        lambda_away=payload["lambda_away"],
    )


@app.get("/predictions/{match_id}", response_model=PredictResponse)
def get_prediction(match_id: str) -> PredictResponse:
    return predict(PredictRequest(match_id=match_id))


@app.get("/score-matrix/{match_id}")
def score_matrix(match_id: str) -> dict:
    features = _load_features()
    predictor = _load_predictor()
    try:
        prediction = predictor.predict_match_id(features, match_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    matrix = prediction.output.matrix.tolist()
    return {
        "match_id": match_id,
        "grid_max_goal": prediction.output.matrix.shape[0] - 1,
        "overflow_prob": prediction.output.overflow_prob,
        "matrix": matrix,
    }


@app.get("/data/freshness")
def data_freshness() -> dict:
    checkpoint = latest_checkpoint(CHECKPOINT_DIR, "baseline_dixon_coles")
    curated_matches = project_root() / "data" / "curated" / "matches.parquet"
    return {
        "items": [
            {
                "source": "feature_mart",
                "path": str(FEATURE_MART),
                "updated_at": _file_mtime(FEATURE_MART),
            },
            {
                "source": "baseline_checkpoint",
                "path": str(checkpoint) if checkpoint else None,
                "updated_at": _file_mtime(checkpoint),
            },
            {
                "source": "curated_matches",
                "path": str(curated_matches),
                "updated_at": _file_mtime(curated_matches),
            },
        ]
    }
