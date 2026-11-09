"""
tests/test_health.py
/health readiness reporting: 200 "ok" with the model loaded, 503 "degraded" without it.
"""

from fastapi.testclient import TestClient

from backend import config
from backend.main import app
from backend.model_utils import model_api


def test_health_check():
    with TestClient(app) as client:  # runs startup, which loads the model
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == config.APP_VERSION
    assert body["model_loaded"] is True
    assert body["model_version"] == model_api.metadata.get("model_version")  # version recorded in the model file
    assert body["uptime_seconds"] >= 0
    assert body["detail"] is None


def test_health_degraded_when_model_missing(monkeypatch):
    monkeypatch.setattr(model_api, "is_loaded", False)
    monkeypatch.setattr(model_api, "last_error", "model file not found")
    response = TestClient(app).get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["detail"] == "model file not found"
