"""
tests/test_error_handlers.py
Unit tests for centralized exceptions, error handlers, and file validation functions.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient

from backend.error_handlers import (
    register_error_handlers,
    validate_file_content,
    validate_file_metadata,
)
from backend.exceptions import (
    AuthenticationException,
    BadRequestException,
    EmptyFileException,
    FileSizeExceededException,
    FileValidationException,
    ForbiddenException,
    InternalServerException,
    NotFoundException,
    RequestValidationException,
)


def test_exception_classes_attributes():
    """Verify status codes and default messages of custom API exceptions."""
    bad_req = BadRequestException()
    assert bad_req.status_code == 400
    assert bad_req.message == "Bad Request"

    file_val = FileValidationException()
    assert file_val.status_code == 400
    assert file_val.message == "Invalid file format"

    empty_err = EmptyFileException()
    assert empty_err.status_code == 400
    assert empty_err.message == "Uploaded file is empty"

    auth_err = AuthenticationException()
    assert auth_err.status_code == 401
    assert auth_err.message == "Unauthorized: Missing X-API-Key header"

    forbid_err = ForbiddenException()
    assert forbid_err.status_code == 403
    assert forbid_err.message == "Forbidden: Invalid API Key"

    not_found = NotFoundException()
    assert not_found.status_code == 404
    assert not_found.message == "Resource not found"

    size_err = FileSizeExceededException()
    assert size_err.status_code == 413
    assert "Payload Too Large" in size_err.message

    val_err = RequestValidationException()
    assert val_err.status_code == 422
    assert "Validation error" in val_err.message

    server_err = InternalServerException()
    assert server_err.status_code == 500
    assert "Internal Server Error" in server_err.message


def test_validate_file_metadata_valid_extensions():
    """Verify that allowed extensions pass metadata validation."""
    for filename in ["doc.jpg", "image.jpeg", "scan.png", "statement.pdf", "PAYS_LIP.PNG", "PHOTO.JPG"]:
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = filename
        validate_file_metadata(mock_file)  # Should not raise


def test_validate_file_metadata_invalid_extensions():
    """Verify that unsupported extensions raise FileValidationException."""
    for bad_name in ["doc.txt", "data.csv", "archive.zip", "run.exe", "file.gif", "image.bmp", "doc"]:
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = bad_name
        with pytest.raises(FileValidationException) as exc_info:
            validate_file_metadata(mock_file)
        assert exc_info.value.status_code == 400
        assert "Invalid file format" in exc_info.value.message


def test_validate_file_metadata_missing_filename():
    """Verify that None, empty, or whitespace-only filename raises FileValidationException."""
    for empty_name in [None, "", "   "]:
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = empty_name
        with pytest.raises(FileValidationException) as exc_info:
            validate_file_metadata(mock_file)
        assert exc_info.value.status_code == 400


def test_validate_file_content_empty_bytes():
    """Verify that 0-byte content raises EmptyFileException."""
    with pytest.raises(EmptyFileException) as exc_info:
        validate_file_content(b"")
    assert exc_info.value.status_code == 400
    assert "empty" in exc_info.value.message.lower()


def test_validate_file_content_oversized():
    """Verify that bytes exceeding max_file_size raise FileSizeExceededException."""
    with pytest.raises(FileSizeExceededException) as exc_info:
        validate_file_content(b"1234567890", max_file_size=5)
    assert exc_info.value.status_code == 413
    assert "Payload Too Large" in exc_info.value.message


def test_validate_file_content_valid():
    """Verify valid content length passes without exception."""
    validate_file_content(b"valid content", max_file_size=100)


def test_isolated_app_error_handlers():
    """Test error handlers on an isolated FastAPI application to verify response formats."""
    test_app = FastAPI()
    register_error_handlers(test_app)

    @test_app.get("/trigger-api-exception")
    def trigger_api_exception():
        raise AuthenticationException("Custom auth failure")

    @test_app.get("/trigger-value-error")
    def trigger_value_error():
        raise ValueError("Invalid pixel dimensions")

    @test_app.get("/trigger-unhandled")
    def trigger_unhandled():
        raise RuntimeError("Unexpected pipeline crash")

    test_client = TestClient(test_app, raise_server_exceptions=False)

    # 1. APIException -> 401
    resp = test_client.get("/trigger-api-exception")
    assert resp.status_code == 401
    assert resp.json() == {
        "success": False,
        "error": "Custom auth failure",
        "code": 401,
        "detail": "Custom auth failure",
    }

    # 2. ValueError -> 400
    resp = test_client.get("/trigger-value-error")
    assert resp.status_code == 400
    assert resp.json()["success"] is False
    assert resp.json()["code"] == 400
    assert "Invalid pixel dimensions" in resp.json()["error"]

    # 3. Unhandled Exception -> 500
    resp = test_client.get("/trigger-unhandled")
    assert resp.status_code == 500
    assert resp.json()["success"] is False
    assert resp.json()["code"] == 500
    # The exception text is logged but never sent to the client (it can contain paths or internals)
    assert "Unexpected pipeline crash" not in resp.json()["error"]
    assert "Internal Server Error" in resp.json()["error"]
