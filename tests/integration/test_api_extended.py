import pytest
from fastapi.testclient import TestClient

from worldcup.api.main import app
from worldcup.calibration.fit import fit_lambda_scale
from worldcup.models.registry import load_checkpoint
from worldcup.utils.paths import project_root


@pytest.fixture
def client():
    feature_mart = project_root() / "data" / "feature_mart" / "match_features.parquet"
    checkpoint_dir = project_root() / "artifacts" / "checkpoints"
    if not feature_mart.exists() or not any(checkpoint_dir.glob("baseline_dixon_coles_*.json")):
        pytest.skip("requires built feature mart and checkpoint")
    return TestClient(app)


def test_matches_include_team_names(client):
    response = client.get("/matches?limit=5")
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert "home_team_name" in item
    assert "away_team_name" in item


def test_match_detail_and_features(client):
    matches = client.get("/matches?limit=1").json()["items"]
    match_id = matches[0]["match_id"]
    detail = client.get(f"/matches/{match_id}")
    features = client.get(f"/features/{match_id}")
    assert detail.status_code == 200
    assert features.status_code == 200
    assert features.json()["team_strength"]


def test_backtest_runs_endpoint(client):
    response = client.get("/backtest/runs")
    assert response.status_code == 200
    assert "items" in response.json()


def test_predict_includes_lambda_scale(client):
    matches = client.get("/matches?limit=1").json()["items"]
    match_id = matches[0]["match_id"]
    response = client.post("/predict", json={"match_id": match_id})
    assert response.status_code == 200
    body = response.json()
    assert "lambda_scale" in body


def test_fit_lambda_scale_on_existing_checkpoint():
    checkpoint_dir = project_root() / "artifacts" / "checkpoints"
    feature_path = project_root() / "data" / "feature_mart" / "match_features.parquet"
    candidates = [
        p for p in checkpoint_dir.glob("baseline_dixon_coles_*.json") if "world_cup" not in p.name
    ]
    if not candidates or not feature_path.exists():
        pytest.skip("requires checkpoint and features")

    import pandas as pd

    checkpoint = load_checkpoint(sorted(candidates)[-1])
    features = pd.read_parquet(feature_path)
    scale, nll = fit_lambda_scale(checkpoint, features, val_ratio=0.15)
    assert 0.5 <= scale <= 2.0
    assert nll >= 0
