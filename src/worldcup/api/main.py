from __future__ import annotations

from fastapi import FastAPI, Query
from pydantic import BaseModel

from worldcup import __version__
from worldcup.api import services as svc

app = FastAPI(title="WorldCup Predictor API", version=__version__)


class PredictRequest(BaseModel):
    match_id: str
    model_version: str | None = None
    model_name: str | None = None


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
    lambda_scale: float
    model_name: str | None = None
    model_type: str | None = None
    model_version: str | None = None
    prediction_source: str = "trained_checkpoint"


class BatchPredictRequest(BaseModel):
    match_ids: list[str]
    model_name: str | None = None


def _to_predict_response(prediction_payload: dict) -> PredictResponse:
    return PredictResponse(
        match_id=prediction_payload["match_id"],
        top3_scorelines=[
            ScorelineResponse(**item) for item in prediction_payload["top3_scorelines"]
        ],
        result_probs=prediction_payload["result_probs"],
        ou25_probs=prediction_payload["ou25_probs"],
        btts_probs=prediction_payload["btts_probs"],
        expected_goals=prediction_payload["expected_goals"],
        uncertainty=prediction_payload["uncertainty"],
        overflow_prob=prediction_payload["overflow_prob"],
        lambda_home=prediction_payload["lambda_home"],
        lambda_away=prediction_payload["lambda_away"],
        lambda_scale=prediction_payload["lambda_scale"],
        model_name=prediction_payload.get("model_name"),
        model_type=prediction_payload.get("model_type"),
        model_version=prediction_payload.get("model_version"),
        prediction_source=prediction_payload.get("prediction_source", "trained_checkpoint"),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/matches")
def list_matches(
    limit: int = Query(default=100, ge=1, le=500),
    world_cup_only: bool = False,
) -> dict:
    return {"items": svc.list_matches(limit=limit, world_cup_only=world_cup_only)}


@app.get("/matches/{match_id}")
def get_match(match_id: str) -> dict:
    return svc.get_match(match_id)


@app.get("/features/{match_id}")
def get_features(match_id: str) -> dict:
    return svc.get_features(match_id)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    return _to_predict_response(svc.prediction_payload(request.match_id, request.model_name))


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest) -> dict:
    return {
        "items": [
            _to_predict_response(item).model_dump()
            for item in svc.predict_matches(request.match_ids, request.model_name)
        ]
    }


@app.get("/predictions/{match_id}", response_model=PredictResponse)
def get_prediction(
    match_id: str,
    model_name: str | None = Query(default=None),
) -> PredictResponse:
    return _to_predict_response(svc.prediction_payload(match_id, model_name))


@app.get("/score-matrix/{match_id}")
def score_matrix(
    match_id: str,
    model_name: str | None = Query(default=None),
) -> dict:
    prediction = svc.predict_match(match_id, model_name)
    matrix = prediction.output.matrix.tolist()
    return {
        "match_id": match_id,
        "grid_max_goal": prediction.output.matrix.shape[0] - 1,
        "overflow_prob": prediction.output.overflow_prob,
        "matrix": matrix,
    }


@app.get("/data/freshness")
def data_freshness(model_name: str | None = Query(default=None)) -> dict:
    return {
        "items": svc.data_freshness(model_name),
        "model": svc.checkpoint_info(model_name),
    }


@app.get("/backtest/runs")
def backtest_runs() -> dict:
    return {"items": svc.list_backtest_runs()}


@app.get("/backtest/{run_id}")
def backtest_run(run_id: str) -> dict:
    return svc.get_backtest_run(run_id)
