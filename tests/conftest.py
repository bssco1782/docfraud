"""
tests/conftest.py
Shared pytest fixtures for the Document Fraud Detection API test suites.

Fixtures provided:
- test_api_key / auth_headers : deterministic API key patched into backend.config
- client                      : FastAPI TestClient (server exceptions -> HTTP responses)
- sample_image_bytes          : in-memory PNG generated with Pillow
- sample_image_path           : real fixture image on disk (tests/fixtures/clean_rgb.png)
- temp_file                   : factory that writes arbitrary bytes to a temp file
- mock_prediction / mock_model: patched model inference for fast, deterministic tests

All fixtures are opt-in (none are autouse), so existing test modules are unaffected.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, Dict, Iterator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import config
from backend.main import app

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Key used by the comprehensive API tests; patched over config.API_KEY so the
# suite never depends on the developer's .env or the "default-api-key" fallback.
TEST_API_KEY = "test-api-key-day6"

# Canned model output returned by the mock_model fixture.
MOCK_PREDICTION: Dict[str, Any] = {
    "tamper_risk_score": 0.12,
    "confidence": 0.88,
    "flagged_regions": [],
    "interpretation": (
        "Document verified successfully. Low risk of digital tampering detected "
        "(risk score: 0.12, confidence: 0.88)."
    ),
}


# -------------------------------------------------------------------------
# Authentication fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def test_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Patch backend.config.API_KEY with a known test key for the duration of a test."""
    # The auth middleware reads config.API_KEY on every request, so patching the
    # module attribute is sufficient; monkeypatch restores it afterwards.
    monkeypatch.setattr(config, "API_KEY", TEST_API_KEY)
    return TEST_API_KEY


@pytest.fixture
def auth_headers(test_api_key: str) -> Dict[str, str]:
    """Request headers carrying the valid (patched) API key."""
    return {"X-API-Key": test_api_key}


# -------------------------------------------------------------------------
# Client fixture
# -------------------------------------------------------------------------

@pytest.fixture
def client() -> Iterator[TestClient]:
    """
    FastAPI TestClient for the application.

    raise_server_exceptions=False makes unhandled exceptions surface as HTTP 500
    responses (as a real server would) instead of being re-raised in the test.
    Used as a context manager so the lifespan hook runs startup validation and
    loads the model, exactly like uvicorn does (/health reports 503 until then).
    """
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


# -------------------------------------------------------------------------
# File fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def sample_image_bytes() -> bytes:
    """A small, valid RGB PNG image generated in memory."""
    image = Image.new("RGB", (64, 64), color=(240, 240, 240))
    # Draw a dark block so the image is not perfectly uniform.
    for x in range(16, 48):
        for y in range(24, 40):
            image.putpixel((x, y), (20, 20, 20))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_image_path() -> Path:
    """Path to a real document-like image shipped with the test fixtures."""
    path = FIXTURES_DIR / "clean_rgb.png"
    assert path.exists(), f"Missing test fixture: {path}"
    return path


@pytest.fixture
def temp_file(tmp_path: Path) -> Callable[[str, bytes], Path]:
    """
    Factory fixture that writes bytes to a file under pytest's tmp_path.

    Usage:
        path = temp_file("upload.png", b"...")
    Files are removed automatically by pytest's tmp_path cleanup.
    """

    def _make(name: str, content: bytes) -> Path:
        path = tmp_path / name
        path.write_bytes(content)
        return path

    return _make


# -------------------------------------------------------------------------
# Model mocks
# -------------------------------------------------------------------------

@pytest.fixture
def mock_prediction() -> Dict[str, Any]:
    """A fresh copy of the canned model prediction."""
    return dict(MOCK_PREDICTION)


@pytest.fixture
def mock_model(mock_prediction: Dict[str, Any]) -> Iterator[MagicMock]:
    """
    Replace model inference in the /verify route with a MagicMock.

    Patches backend.main.analyze_document (the function the route offloads to a
    worker thread), so request validation, auth and error handling still run for
    real while feature extraction and the RandomForest are skipped.
    Tests may set `mock_model.side_effect` to simulate failures.
    """
    with patch("backend.main.analyze_document", return_value=mock_prediction) as mocked:
        yield mocked
