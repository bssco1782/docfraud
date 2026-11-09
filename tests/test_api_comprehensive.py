"""
tests/test_api_comprehensive.py
Comprehensive automated test suite for the Document Fraud Detection API (Day 6).

Coverage areas:
- Health endpoint          : status code, response fields
- Authentication           : missing / invalid / valid X-API-Key
- File validation          : empty, unsupported type, oversized, corrupted
- Verify endpoint          : prediction fields, value ranges, JSON schema
- Error handling           : internal exceptions, invalid requests, unknown routes

Fixtures come from tests/conftest.py. Most /verify tests use `mock_model` so they
are fast and deterministic; tests marked "real model" run the full forensic
pipeline against the trained model in data/models/.

Run:
    pytest tests/test_api_comprehensive.py -v
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from backend import config

VERIFY_URL = "/verify"
HEALTH_URL = "/health"

# Fields every error body must contain (see backend/error_handlers.py).
ERROR_FIELDS = {"success", "error", "code", "detail"}
# Fields every successful /verify body must contain (see VerifyResponse).
VERIFY_FIELDS = {"tamper_risk_score", "confidence", "flagged_regions", "interpretation"}


def upload(name: str, content: bytes, content_type: str = "image/png") -> Dict[str, Any]:
    """Build the multipart `files` argument for a /verify upload."""
    return {"file": (name, content, content_type)}


def assert_error_response(response, expected_status: int) -> Dict[str, Any]:
    """Assert the standardized error envelope and return the parsed body."""
    assert response.status_code == expected_status
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert ERROR_FIELDS <= body.keys()
    assert body["success"] is False
    assert body["code"] == expected_status
    assert isinstance(body["error"], str) and body["error"]
    return body


# =========================================================================
# TASK 2: Health endpoint
# =========================================================================

class TestHealthEndpoint:
    """GET /health is public and reports service status and version."""

    def test_health_returns_200(self, client):
        response = client.get(HEALTH_URL)
        assert response.status_code == 200

    def test_health_contains_status_field(self, client):
        body = client.get(HEALTH_URL).json()
        assert "status" in body
        assert body["status"] == "ok"

    def test_health_contains_version_field(self, client):
        body = client.get(HEALTH_URL).json()
        assert "version" in body
        assert body["version"] == client.app.version


# =========================================================================
# TASK 3: Authentication
# =========================================================================

class TestAuthentication:
    """X-API-Key enforcement on protected endpoints."""

    def test_missing_api_key_returns_401(self, client, test_api_key, mock_model, sample_image_bytes):
        response = client.post(VERIFY_URL, files=upload("doc.png", sample_image_bytes))
        body = assert_error_response(response, 401)
        assert "Missing X-API-Key" in body["error"]
        mock_model.assert_not_called()

    def test_invalid_api_key_returns_403(self, client, test_api_key, mock_model, sample_image_bytes):
        response = client.post(
            VERIFY_URL,
            files=upload("doc.png", sample_image_bytes),
            headers={"X-API-Key": "wrong-key"},
        )
        body = assert_error_response(response, 403)
        assert "Invalid API Key" in body["error"]
        mock_model.assert_not_called()

    def test_valid_api_key_succeeds(self, client, auth_headers, mock_model, sample_image_bytes):
        response = client.post(
            VERIFY_URL, files=upload("doc.png", sample_image_bytes), headers=auth_headers
        )
        assert response.status_code == 200
        mock_model.assert_called_once()

    def test_health_does_not_require_api_key(self, client, test_api_key):
        # Public endpoint must stay reachable with no header and with a bad header.
        assert client.get(HEALTH_URL).status_code == 200
        assert client.get(HEALTH_URL, headers={"X-API-Key": "wrong-key"}).status_code == 200


# =========================================================================
# TASK 4: File validation
# =========================================================================

class TestFileValidation:
    """Upload validation runs before the model is ever invoked."""

    def test_empty_file_rejected(self, client, auth_headers, mock_model):
        response = client.post(VERIFY_URL, files=upload("empty.png", b""), headers=auth_headers)
        body = assert_error_response(response, 400)
        assert "empty" in body["error"].lower()
        mock_model.assert_not_called()

    @pytest.mark.parametrize(
        "filename, content_type",
        [
            ("notes.txt", "text/plain"),
            ("archive.zip", "application/zip"),
            ("script.exe", "application/octet-stream"),
            ("no_extension", "image/png"),
        ],
    )
    def test_unsupported_file_type_rejected(
        self, client, auth_headers, mock_model, sample_image_bytes, filename, content_type
    ):
        response = client.post(
            VERIFY_URL,
            files=upload(filename, sample_image_bytes, content_type),
            headers=auth_headers,
        )
        body = assert_error_response(response, 400)
        assert "Invalid file format" in body["error"]
        mock_model.assert_not_called()

    def test_oversized_file_rejected(self, client, auth_headers, mock_model, monkeypatch):
        # Shrink the limit instead of uploading 10 MB; the route reads
        # config.MAX_FILE_SIZE at request time.
        monkeypatch.setattr(config, "MAX_FILE_SIZE", 1024)
        response = client.post(
            VERIFY_URL, files=upload("big.png", b"\x00" * 4096), headers=auth_headers
        )
        body = assert_error_response(response, 413)
        assert "Payload Too Large" in body["error"]
        mock_model.assert_not_called()

    def test_corrupted_file_rejected(self, client, auth_headers, temp_file):
        # Real model: bytes with an image extension that are not a decodable image.
        corrupt = temp_file("corrupt.jpg", b"\xff\xd8\xff\xe0 this is not a real jpeg")
        response = client.post(
            VERIFY_URL,
            files=upload(corrupt.name, corrupt.read_bytes(), "image/jpeg"),
            headers=auth_headers,
        )
        body = assert_error_response(response, 400)
        assert "corrupted" in body["error"].lower()

    def test_missing_file_field_rejected(self, client, auth_headers, mock_model):
        response = client.post(VERIFY_URL, headers=auth_headers)
        body = assert_error_response(response, 400)
        assert "file" in body["error"].lower()
        mock_model.assert_not_called()


# =========================================================================
# TASK 5: Verify endpoint
# =========================================================================

class TestVerifyEndpoint:
    """Successful /verify responses and their schema."""

    def test_valid_image_returns_prediction(self, client, auth_headers, sample_image_path):
        # Real model: full feature extraction + RandomForest inference.
        response = client.post(
            VERIFY_URL,
            files=upload(sample_image_path.name, sample_image_path.read_bytes()),
            headers=auth_headers,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert VERIFY_FIELDS <= body.keys()
        assert 0.0 <= body["tamper_risk_score"] <= 1.0

    def test_response_contains_tamper_risk_score(self, client, auth_headers, mock_model, sample_image_bytes):
        body = client.post(
            VERIFY_URL, files=upload("doc.png", sample_image_bytes), headers=auth_headers
        ).json()
        assert "tamper_risk_score" in body
        assert isinstance(body["tamper_risk_score"], float)
        assert 0.0 <= body["tamper_risk_score"] <= 1.0

    def test_response_contains_confidence(self, client, auth_headers, mock_model, sample_image_bytes):
        body = client.post(
            VERIFY_URL, files=upload("doc.png", sample_image_bytes), headers=auth_headers
        ).json()
        assert "confidence" in body
        assert isinstance(body["confidence"], float)
        assert 0.0 <= body["confidence"] <= 1.0

    def test_response_contains_interpretation(self, client, auth_headers, mock_model, sample_image_bytes):
        body = client.post(
            VERIFY_URL, files=upload("doc.png", sample_image_bytes), headers=auth_headers
        ).json()
        assert "interpretation" in body
        assert isinstance(body["interpretation"], str)
        assert body["interpretation"].strip()

    def test_response_json_format(self, client, auth_headers, mock_model, mock_prediction, sample_image_bytes):
        response = client.post(
            VERIFY_URL, files=upload("doc.png", sample_image_bytes), headers=auth_headers
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        # Exactly the documented fields, no leaks of internal data.
        assert set(body.keys()) == VERIFY_FIELDS
        assert isinstance(body["flagged_regions"], list)
        assert body == mock_prediction

    def test_model_receives_filename_and_bytes(self, client, auth_headers, mock_model, sample_image_bytes):
        client.post(VERIFY_URL, files=upload("scan.png", sample_image_bytes), headers=auth_headers)
        mock_model.assert_called_once_with("scan.png", sample_image_bytes)

    @pytest.mark.parametrize("filename", ["doc.jpg", "doc.jpeg", "doc.PNG", "doc.pdf"])
    def test_all_supported_extensions_accepted(
        self, client, auth_headers, mock_model, sample_image_bytes, filename
    ):
        response = client.post(
            VERIFY_URL, files=upload(filename, sample_image_bytes), headers=auth_headers
        )
        assert response.status_code == 200


# =========================================================================
# TASK 6: Error handling
# =========================================================================

class TestErrorHandling:
    """Centralized handlers return the standardized error envelope."""

    def test_internal_exception_returns_500(self, client, auth_headers, mock_model, sample_image_bytes):
        mock_model.side_effect = RuntimeError("simulated inference failure")
        response = client.post(
            VERIFY_URL, files=upload("doc.png", sample_image_bytes), headers=auth_headers
        )
        body = assert_error_response(response, 500)
        assert "Internal Server Error" in body["error"]

    def test_model_value_error_returns_400(self, client, auth_headers, mock_model, sample_image_bytes):
        mock_model.side_effect = ValueError("cannot identify image file")
        response = client.post(
            VERIFY_URL, files=upload("doc.png", sample_image_bytes), headers=auth_headers
        )
        body = assert_error_response(response, 400)
        assert "cannot identify image file" in body["error"]

    def test_invalid_request_returns_422(self, client, auth_headers, mock_model, sample_image_bytes):
        # threshold is constrained to [0.0, 1.0] by the route signature.
        response = client.post(
            VERIFY_URL,
            params={"threshold": 5},
            files=upload("doc.png", sample_image_bytes),
            headers=auth_headers,
        )
        body = assert_error_response(response, 422)
        assert "threshold" in body["error"]
        assert isinstance(body.get("details"), list)
        mock_model.assert_not_called()

    def test_endpoint_not_found_returns_404(self, client):
        response = client.get("/does-not-exist")
        body = assert_error_response(response, 404)
        assert "/does-not-exist" in body["error"]

    def test_unknown_endpoint_404_even_without_api_key(self, client, test_api_key):
        # Route existence is checked before auth, so unknown paths never leak 401/403.
        response = client.post("/admin/secret")
        assert_error_response(response, 404)

    def test_wrong_method_returns_405(self, client, auth_headers):
        response = client.get(VERIFY_URL, headers=auth_headers)
        assert_error_response(response, 405)
