"""
tests/test_hardening.py
Day 9 production-hardening tests: upload limits, image-dimension guards, error-message
hygiene, CORS preflight, request IDs, model availability / recovery and concurrency limits.
"""

from __future__ import annotations

import io
import pickle
import tempfile
import threading
import time
import zlib
from struct import pack
from typing import Iterator

import numpy as np
import pymupdf
import pytest
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

import backend.main as main_module
from backend import config
from backend.model_utils import ModelAPI, model_api


def png_bytes(width: int, height: int) -> bytes:
    """A plain white PNG of the given size."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def png_bomb(width: int, height: int) -> bytes:
    """A small, valid grayscale PNG whose header declares width x height pixels."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return pack(">I", len(data)) + kind + data + pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    compressor = zlib.compressobj(9)
    row = b"\x00" * (width + 1)
    data = b"".join(compressor.compress(row) for _ in range(height)) + compressor.flush()
    header = pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", data) + chunk(b"IEND", b"")


def pdf_bytes(width_pt: float, height_pt: float, encrypt: bool = False) -> bytes:
    """A one-page PDF with a line of text."""
    doc = pymupdf.open()
    doc.new_page(width=width_pt, height=height_pt).insert_text((20, 40), "Payslip")
    options = {"encryption": pymupdf.PDF_ENCRYPT_AES_256, "user_pw": "u", "owner_pw": "o"} if encrypt else {}
    return doc.tobytes(**options)


def upload(name: str, content: bytes, content_type: str = "image/png") -> dict:
    """Multipart `files` argument for /verify."""
    return {"file": (name, content, content_type)}


# =========================================================================
# Upload size enforcement before parsing
# =========================================================================


class TestUploadLimits:
    def test_oversized_content_length_rejected_before_parsing(self, client, auth_headers, mock_model):
        limit = config.MAX_FILE_SIZE + config.MULTIPART_OVERHEAD_BYTES
        response = client.post(
            "/verify",
            files=upload("doc.png", b"tiny"),
            headers={**auth_headers, "Content-Length": str(limit + 1)},
        )
        assert response.status_code == 413
        assert response.json()["code"] == 413
        mock_model.assert_not_called()

    def test_chunked_upload_without_content_length_is_counted(self, client, auth_headers, mock_model, monkeypatch):
        monkeypatch.setattr(config, "MAX_FILE_SIZE", 1024)
        monkeypatch.setattr(config, "MULTIPART_OVERHEAD_BYTES", 1024)
        boundary = "hardeningboundary"
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="big.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()

        def body() -> Iterator[bytes]:  # a generator body is sent chunked, with no Content-Length
            yield head
            for _ in range(64):
                yield b"\x00" * 1024
            yield f"\r\n--{boundary}--\r\n".encode()

        response = client.post(
            "/verify",
            content=body(),
            headers={**auth_headers, "Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        assert response.status_code == 413
        assert "Payload Too Large" in response.json()["error"]
        mock_model.assert_not_called()

    def test_unauthenticated_upload_rejected_before_size_check(self, client, test_api_key):
        response = client.post(
            "/verify",
            files=upload("doc.png", b"tiny"),
            headers={"Content-Length": str(10**9)},
        )
        assert response.status_code == 401

    def test_file_just_under_limit_is_not_rejected_for_framing(
        self, client, auth_headers, mock_model, monkeypatch, sample_image_bytes
    ):
        # Old behaviour compared the whole multipart body to MAX_FILE_SIZE, so a file
        # slightly under the limit was rejected because of the form framing around it.
        monkeypatch.setattr(config, "MAX_FILE_SIZE", len(sample_image_bytes))
        response = client.post("/verify", files=upload("doc.png", sample_image_bytes), headers=auth_headers)
        assert response.status_code == 200


# =========================================================================
# Image dimension guards (real model, no mock)
# =========================================================================


class TestImageGuards:
    @pytest.mark.parametrize("width, height", [(1, 1), (8, 8), (20000, 10)])
    def test_too_small_or_narrow_images_rejected(self, client, auth_headers, width, height):
        response = client.post("/verify", files=upload("tiny.png", png_bytes(width, height)), headers=auth_headers)
        assert response.status_code == 400
        assert "too small or too narrow" in response.json()["error"]

    def test_decompression_bomb_rejected_quickly(self, client, auth_headers):
        bomb = png_bomb(10000, 5000)  # 50 MP declared, ~50 KB on disk
        assert len(bomb) < 200_000
        start = time.perf_counter()
        response = client.post("/verify", files=upload("bomb.png", bomb), headers=auth_headers)
        assert response.status_code == 400
        assert "pixel limit" in response.json()["error"]
        assert time.perf_counter() - start < 2.0

    def test_oversized_pdf_page_rejected(self, client, auth_headers):
        response = client.post(
            "/verify", files=upload("poster.pdf", pdf_bytes(5000, 5000), "application/pdf"), headers=auth_headers
        )
        assert response.status_code == 400
        assert "pixel limit" in response.json()["error"]

    def test_encrypted_pdf_rejected(self, client, auth_headers):
        response = client.post(
            "/verify",
            files=upload("locked.pdf", pdf_bytes(595, 842, encrypt=True), "application/pdf"),
            headers=auth_headers,
        )
        assert response.status_code == 400
        assert "Password-protected" in response.json()["error"]

    def test_a4_pdf_still_accepted(self, client, auth_headers):
        response = client.post(
            "/verify", files=upload("a4.pdf", pdf_bytes(595, 842), "application/pdf"), headers=auth_headers
        )
        assert response.status_code == 200

    @pytest.mark.parametrize("width, height", [(16, 16), (80, 60)])
    def test_small_but_valid_images_accepted(self, client, auth_headers, width, height):
        response = client.post("/verify", files=upload("small.png", png_bytes(width, height)), headers=auth_headers)
        assert response.status_code == 200


# =========================================================================
# Error-message hygiene and path handling
# =========================================================================


class TestErrorHygiene:
    def test_corrupted_upload_error_does_not_leak_temp_path(self, client, auth_headers):
        broken = png_bytes(100, 100)[:200]
        response = client.post("/verify", files=upload("scan.png", broken), headers=auth_headers)
        assert response.status_code == 400
        error = response.json()["error"]
        assert tempfile.gettempdir() not in error
        assert "tmp" not in error.lower().replace("tampering", "")

    def test_unexpected_error_returns_generic_500_with_request_id(self, client, auth_headers, mock_model):
        mock_model.side_effect = RuntimeError("secret internal detail /srv/app/models")
        response = client.post(
            "/verify", files=upload("doc.png", png_bytes(100, 100)), headers={**auth_headers, "X-Request-ID": "abc123"}
        )
        assert response.status_code == 500
        assert "secret internal detail" not in response.json()["error"]
        assert "abc123" in response.json()["error"]

    def test_path_traversal_filename_only_affects_extension(self, client, auth_headers, mock_model):
        response = client.post(
            "/verify", files=upload("../../../../etc/passwd.png", png_bytes(100, 100)), headers=auth_headers
        )
        assert response.status_code == 200
        # The analyzer only ever sees the base name, never a path supplied by the client
        assert mock_model.call_args.args[0] == "passwd.png"


# =========================================================================
# Authentication, CORS and request IDs
# =========================================================================


class TestTransport:
    def test_cors_preflight_is_not_blocked_by_api_key(self, client):
        response = client.options(
            "/verify",
            headers={
                "Origin": "https://frontend.example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-api-key",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") in {"*", "https://frontend.example.com"}

    def test_wildcard_cors_does_not_allow_credentials(self, client):
        response = client.get("/health", headers={"Origin": "https://evil.example.com", "Cookie": "a=b"})
        assert response.headers.get("access-control-allow-credentials") != "true"

    def test_same_length_wrong_key_rejected(self, client, test_api_key, mock_model):
        wrong = "x" * len(test_api_key)
        response = client.post("/verify", files=upload("d.png", png_bytes(50, 50)), headers={"X-API-Key": wrong})
        assert response.status_code == 403

    def test_request_id_generated_and_echoed(self, client):
        generated = client.get("/health").headers.get("x-request-id")
        assert generated and len(generated) >= 8
        assert client.get("/health", headers={"X-Request-ID": "client-id-42"}).headers["x-request-id"] == "client-id-42"

    def test_unsafe_request_id_is_replaced(self, client):
        response = client.get("/health", headers={"X-Request-ID": "bad id\\n<script>"})
        assert response.headers["x-request-id"] != "bad id\\n<script>"


# =========================================================================
# Model availability, startup validation and recovery
# =========================================================================


class TestModelAvailability:
    def test_verify_returns_503_while_model_missing_then_recovers(self, client, auth_headers, tmp_path, monkeypatch):
        original = model_api.model_path
        monkeypatch.setattr(model_api, "model", None)
        monkeypatch.setattr(model_api, "is_loaded", False)
        monkeypatch.setattr(model_api, "model_path", tmp_path / "missing.pkl")

        response = client.post("/verify", files=upload("d.png", png_bytes(100, 100)), headers=auth_headers)
        assert response.status_code == 503
        assert client.get("/health").status_code == 503

        model_api.model_path = original  # "fix" the deployment; no restart needed
        response = client.post("/verify", files=upload("d.png", png_bytes(100, 100)), headers=auth_headers)
        assert response.status_code == 200
        assert client.get("/health").json()["status"] == "ok"

    def test_incompatible_model_is_refused_at_load(self, tmp_path):
        model = RandomForestClassifier(n_estimators=2, random_state=0).fit(np.zeros((4, 5)), [0, 1, 0, 1])
        path = tmp_path / "five_features.pkl"
        path.write_bytes(pickle.dumps({"model": model, "metadata": {"model_version": "test"}}))
        api = ModelAPI(path)
        with pytest.raises(RuntimeError, match="expects 5 features"):
            api.load_model()
        assert api.status()["loaded"] is False
        assert "5 features" in api.status()["error"]

    def test_startup_warns_about_default_key_and_open_cors(self, monkeypatch):
        monkeypatch.setattr(config, "API_KEY", config.DEFAULT_API_KEY)
        monkeypatch.setattr(config, "CORS_ORIGINS", ["*"])
        warnings = main_module.validate_startup_configuration()
        assert any("API_KEY" in w for w in warnings)
        assert any("CORS" in w for w in warnings)

    def test_startup_quiet_with_safe_configuration(self, monkeypatch):
        monkeypatch.setattr(config, "API_KEY", "a-long-random-secret-value-123")
        monkeypatch.setattr(config, "CORS_ORIGINS", ["https://app.example.com"])
        assert main_module.validate_startup_configuration() == []


# =========================================================================
# Concurrency limit
# =========================================================================


def test_concurrent_analyses_are_capped(client, auth_headers, monkeypatch):
    monkeypatch.setattr(main_module, "_analysis_slots", threading.BoundedSemaphore(1))
    active, peak, lock = [0], [0], threading.Lock()

    def slow_analysis(filename, content):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        with lock:
            active[0] -= 1
        return {"tamper_risk_score": 0.1, "confidence": 0.9, "flagged_regions": [], "interpretation": "ok"}

    monkeypatch.setattr(main_module, "analyze_document", slow_analysis)
    image = png_bytes(50, 50)
    statuses = []
    threads = [
        threading.Thread(
            target=lambda: statuses.append(
                client.post("/verify", files=upload("d.png", image), headers=auth_headers).status_code
            )
        )
        for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert statuses == [200, 200, 200]
    assert peak[0] == 1


def test_upload_limit_stops_reading_streamed_body_early():
    """The middleware aborts a streamed body at the limit instead of letting the app read it all."""
    import asyncio

    from backend.middleware import UploadSizeLimitMiddleware

    chunks_sent = [0]
    bytes_seen_by_app = [0]
    sent_messages = []

    async def app(scope, receive, send):
        while True:  # an app that reads the whole body, like multipart parsing does
            message = await receive()
            bytes_seen_by_app[0] += len(message.get("body", b""))
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        chunks_sent[0] += 1
        return {"type": "http.request", "body": b"x" * 1000, "more_body": chunks_sent[0] < 1000}

    async def send(message):
        sent_messages.append(message)

    middleware = UploadSizeLimitMiddleware(app, max_body_bytes=lambda: 5000, message=lambda: "too large")
    scope = {"type": "http", "method": "POST", "path": "/verify", "headers": []}
    asyncio.run(middleware(scope, receive, send))

    assert sent_messages[0]["status"] == 413
    assert chunks_sent[0] <= 6  # stopped just past 5000 bytes, not after 1,000,000
    assert bytes_seen_by_app[0] <= 5000


def test_filename_control_characters_are_escaped_in_logs(client, auth_headers, mock_model, caplog):
    """A filename with raw control characters (e.g. ANSI escapes) is logged escaped, not raw."""
    import logging

    boundary = "logboundary"
    evil_name = "scan\x1b[2KFAKE LOG LINE.png"
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{evil_name}"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()
        + png_bytes(50, 50)
        + f"\r\n--{boundary}--\r\n".encode()
    )
    with caplog.at_level(logging.INFO, logger="backend.main"):
        response = client.post(
            "/verify",
            content=body,
            headers={**auth_headers, "Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
    assert response.status_code == 200
    assert "\x1b" in mock_model.call_args.args[0]  # the raw control character reached the server
    verified = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Verified")]
    assert verified
    assert all("\x1b" not in m and "\r" not in m for m in verified)
    assert "\\x1b" in verified[0]  # logged in escaped form
