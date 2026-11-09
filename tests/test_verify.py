"""
tests/test_verify.py
Integration tests for document verification endpoint, error handlers, and file validation.
"""

import io
import time
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import config
from backend.main import app

client = TestClient(app, raise_server_exceptions=False)
VALID_API_KEY = config.API_KEY


def test_health_public():
    """Verify health endpoint remains accessible without API Key."""
    with TestClient(app) as started:  # runs startup, which loads the model
        response = started.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"] == "0.1.0"


def test_verify_missing_api_key():
    """Verify missing X-API-Key header returns 401 Unauthorized."""
    file_content = b"sample document content"
    files = {"file": ("test.png", io.BytesIO(file_content), "image/png")}
    response = client.post("/verify", files=files)
    assert response.status_code == 401
    data = response.json()
    assert data["success"] is False
    assert data["code"] == 401
    assert "error" in data
    assert "detail" in data
    assert "Unauthorized" in data["error"]


def test_verify_invalid_api_key():
    """Verify invalid X-API-Key header returns 403 Forbidden."""
    file_content = b"sample document content"
    files = {"file": ("test.png", io.BytesIO(file_content), "image/png")}
    headers = {"X-API-Key": "incorrect-secret-key"}
    response = client.post("/verify", files=files, headers=headers)
    assert response.status_code == 403
    data = response.json()
    assert data["success"] is False
    assert data["code"] == 403
    assert "error" in data
    assert "detail" in data
    assert "Forbidden" in data["error"]


def test_verify_success():
    """Verify valid file upload with correct API key returns 200 and expected schema."""
    file_content = (Path(__file__).parent / "fixtures" / "clean_rgb.png").read_bytes()
    files = {"file": ("sample_doc.png", io.BytesIO(file_content), "image/png")}
    headers = {"X-API-Key": VALID_API_KEY}
    response = client.post("/verify", files=files, headers=headers)
    assert response.status_code == 200
    data = response.json()

    # Verify all expected keys are present
    assert "tamper_risk_score" in data
    assert "confidence" in data
    assert "flagged_regions" in data
    assert "interpretation" in data

    assert isinstance(data["tamper_risk_score"], (int, float))
    assert isinstance(data["confidence"], (int, float))
    assert isinstance(data["flagged_regions"], list)
    assert isinstance(data["interpretation"], str)


def test_verify_empty_file():
    """Verify empty file upload returns 400 Bad Request."""
    files = {"file": ("empty.png", io.BytesIO(b""), "image/png")}
    headers = {"X-API-Key": VALID_API_KEY}
    response = client.post("/verify", files=files, headers=headers)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["code"] == 400
    assert "error" in data
    assert "detail" in data
    assert "empty" in data["error"].lower()


def test_verify_missing_file_payload():
    """Verify request with missing file payload returns 400 Bad Request."""
    headers = {"X-API-Key": VALID_API_KEY}
    response = client.post("/verify", headers=headers)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["code"] == 400
    assert "error" in data
    assert "detail" in data


def test_verify_unsupported_file_extension():
    """Verify rejection of unsupported file extensions (e.g., .txt, .docx, .exe)."""
    for bad_ext in ["document.txt", "resume.docx", "malware.exe"]:
        files = {"file": (bad_ext, io.BytesIO(b"sample non-image text"), "text/plain")}
        headers = {"X-API-Key": VALID_API_KEY}
        response = client.post("/verify", files=files, headers=headers)
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["code"] == 400
        assert "Invalid file format" in data["error"]
        assert "Allowed formats: jpg, jpeg, png, pdf" in data["error"]


def test_verify_empty_filename():
    """Verify upload with missing or whitespace-only filename returns 400 Bad Request."""
    files = {"file": ("", io.BytesIO(b"some content"), "image/png")}
    headers = {"X-API-Key": VALID_API_KEY}
    response = client.post("/verify", files=files, headers=headers)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["code"] == 400
    assert "detail" in data


def test_verify_file_too_large_content_length():
    """Verify pre-flight rejection via Content-Length header exceeding MAX_FILE_SIZE plus multipart framing."""
    headers = {
        "X-API-Key": VALID_API_KEY,
        "Content-Length": str(config.MAX_FILE_SIZE + config.MULTIPART_OVERHEAD_BYTES + 1),
    }
    files = {"file": ("large.png", io.BytesIO(b"small body"), "image/png")}
    response = client.post("/verify", files=files, headers=headers)
    assert response.status_code == 413
    data = response.json()
    assert data["success"] is False
    assert data["code"] == 413
    assert "Payload Too Large" in data["error"]


def test_verify_file_too_large_actual_bytes():
    """Verify file exceeding MAX_FILE_SIZE returns 413 Payload Too Large."""
    with patch.object(config, "MAX_FILE_SIZE", 50):
        large_content = b"X" * 100
        files = {"file": ("large.png", io.BytesIO(large_content), "image/png")}
        headers = {"X-API-Key": VALID_API_KEY}
        response = client.post("/verify", files=files, headers=headers)
        assert response.status_code == 413
        data = response.json()
        assert data["success"] is False
        assert data["code"] == 413
        assert "Payload Too Large" in data["error"]


def test_verify_internal_server_error():
    """Verify unexpected processing exception returns 500 Internal Server Error."""
    file_content = b"sample document content"
    files = {"file": ("error.png", io.BytesIO(file_content), "image/png")}
    headers = {"X-API-Key": VALID_API_KEY}

    with patch("backend.main.analyze_document", side_effect=RuntimeError("Model inference failed")):
        response = client.post("/verify", files=files, headers=headers)
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert data["code"] == 500
        assert "Internal Server Error" in data["error"]


def test_verify_invalid_image_bytes():
    """Verify invalid or corrupted image bytes return 400 Bad Request."""
    corrupted_bytes = b"NOT_AN_IMAGE_DATA_CORRUPT"
    files = {"file": ("corrupted.png", io.BytesIO(corrupted_bytes), "image/png")}
    headers = {"X-API-Key": VALID_API_KEY}
    response = client.post("/verify", files=files, headers=headers)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["code"] == 400
    assert "Bad Request" in data["detail"]


def test_not_found_endpoint_404():
    """Verify requests to non-existent endpoints return structured 404 Not Found response."""
    response = client.get("/non-existent-endpoint")
    assert response.status_code == 404
    data = response.json()
    assert data["success"] is False
    assert data["code"] == 404
    assert "Not Found" in data["error"]


def test_validation_error_422():
    """Verify invalid query parameter returns structured 422 Unprocessable Entity response."""
    file_content = b"not validated: the threshold check fails first"
    files = {"file": ("sample.png", io.BytesIO(file_content), "image/png")}
    headers = {"X-API-Key": VALID_API_KEY}
    # threshold must be between 0.0 and 1.0; 999 is invalid
    response = client.post("/verify?threshold=999", files=files, headers=headers)
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["code"] == 422
    assert "Validation error" in data["error"]
    assert "details" in data


def test_verify_synthetic_documents_end_to_end():
    """
    Test end-to-end: upload image -> get risk score.
    Runs verification on 5 synthetic images and asserts response time and valid scores.
    """
    synthetic_dir = Path(__file__).parent.parent / "data" / "synthetic"
    image_paths = [
        synthetic_dir / "genuine" / "payslip_001.png",
        synthetic_dir / "genuine" / "payslip_002.png",
        synthetic_dir / "genuine" / "bank_stmt_011.png",
        synthetic_dir / "genuine" / "id_proof_019.png",
        synthetic_dir / "tampered" / "payslip_004.png",
    ]

    # Fallback to fixtures if synthetic files aren't found
    if not all(p.exists() for p in image_paths):
        fixtures_dir = Path(__file__).parent / "fixtures"
        image_paths = [
            fixtures_dir / "clean_rgb.png",
            fixtures_dir / "clean_rgb.jpg",
            fixtures_dir / "grayscale.jpg",
            fixtures_dir / "small_image.jpg",
            fixtures_dir / "tampered_patch.jpg",
        ]

    latencies = []
    for img_path in image_paths:
        file_bytes = img_path.read_bytes()
        files = {"file": (img_path.name, io.BytesIO(file_bytes), "image/png")}
        headers = {"X-API-Key": VALID_API_KEY}

        start_time = time.perf_counter()
        response = client.post("/verify", files=files, headers=headers)
        elapsed = time.perf_counter() - start_time
        latencies.append(elapsed)

        assert response.status_code == 200, f"Failed for {img_path.name}: {response.text}"
        data = response.json()

        assert "tamper_risk_score" in data
        assert "confidence" in data
        assert "flagged_regions" in data
        assert "interpretation" in data

        assert isinstance(data["tamper_risk_score"], (int, float))
        assert 0.0 <= data["tamper_risk_score"] <= 1.0
        assert isinstance(data["confidence"], (int, float))
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["flagged_regions"], list)
        assert isinstance(data["interpretation"], str)
        assert len(data["interpretation"]) > 0

    avg_latency = sum(latencies) / len(latencies)
    print(f"\nTested {len(image_paths)} synthetic images:")
    for path, lat in zip(image_paths, latencies):
        print(f"  {path.name:25s} -> {lat:.3f}s")
    print(f"Average response time: {avg_latency:.3f}s per image")
