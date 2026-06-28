from fastapi.testclient import TestClient

from worldcup.api.main import app


def test_freshness_endpoint():
    client = TestClient(app)
    response = client.get("/data/freshness")
    assert response.status_code == 200
    body = response.json()
    assert "items" in body
