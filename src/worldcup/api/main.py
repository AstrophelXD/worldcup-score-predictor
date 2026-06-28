from __future__ import annotations

from fastapi import FastAPI, Query
from pydantic import BaseModel

from worldcup import __version__
from worldcup.api import services as svc

app = FastAPI(title="WorldCup Predictor API", version=__version__)


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
    lambda_scale: float


def _to_predict_response(prediction) -> PredictResponse:
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
        lambda_scale=payload["lambda_scale"],
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
    return _to_predict_response(svc.predict_match(request.match_id))


@app.get("/predictions/{match_id}", response_model=PredictResponse)
def get_prediction(match_id: str) -> PredictResponse:
    return _to_predict_response(svc.predict_match(match_id))


@app.get("/score-matrix/{match_id}")
def score_matrix(match_id: str) -> dict:
    prediction = svc.predict_match(match_id)
    matrix = prediction.output.matrix.tolist()
    return {
        "match_id": match_id,
        "grid_max_goal": prediction.output.matrix.shape[0] - 1,
        "overflow_prob": prediction.output.overflow_prob,
        "matrix": matrix,
    }


@app.get("/data/freshness")
def data_freshness() -> dict:
    return {"items": svc.data_freshness(), "model": svc.checkpoint_info()}


@app.get("/backtest/runs")
def backtest_runs() -> dict:
    return {"items": svc.list_backtest_runs()}


@app.get("/backtest/{run_id}")
def backtest_run(run_id: str) -> dict:
    return svc.get_backtest_run(run_id)
