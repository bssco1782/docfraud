"""
backend/error_handlers.py
Centralized error handling and file validation module for Document Fraud Detection API.
Provides uniform JSON error responses across all HTTP status codes (400, 401, 403, 404, 413, 422, 500).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional

from fastapi import FastAPI, Request, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend import config
from backend.exceptions import (
    APIException,
    EmptyFileException,
    FileSizeExceededException,
    FileValidationException,
)
from backend.logging_config import request_id_var

logger = logging.getLogger(__name__)

# Allowed file extensions for document fraud analysis (defined in config)
ALLOWED_EXTENSIONS: FrozenSet[str] = config.ALLOWED_EXTENSIONS


def create_error_response(
    status_code: int,
    error: str,
    details: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    """
    Construct standardized JSON error response body.

    Format:
    {
        "success": False,
        "error": "Error description",
        "code": 400,
        "detail": "Error description",  # Kept for backward compatibility
        "details": [...]                # Optional structured details
    }
    """
    content: Dict[str, Any] = {
        "success": False,
        "error": error,
        "code": status_code,
        "detail": error,
    }
    if details is not None:
        content["details"] = jsonable_encoder(details)

    return JSONResponse(status_code=status_code, content=content, headers=headers)


def validate_file_metadata(file: Optional[UploadFile]) -> None:
    """
    Validate uploaded file presence, filename, and extension against allowed types.

    Raises:
        FileValidationException: If filename is missing or extension is not allowed.
    """
    if file is None or not file.filename or not file.filename.strip():
        raise FileValidationException("Bad Request: Missing or empty filename")

    suffix = Path(file.filename).suffix.lower()
    if not suffix or suffix not in ALLOWED_EXTENSIONS:
        raise FileValidationException(
            f"Invalid file format: '{suffix or 'unspecified'}' is not supported. Allowed formats: jpg, jpeg, png, pdf"
        )


def file_size_exceeded_message(max_file_size: int) -> str:
    """Build the standard 413 message for an upload larger than max_file_size bytes."""
    max_mb = max_file_size // (1024 * 1024)
    size_label = f" ({max_mb}MB)" if max_mb > 0 else ""
    return f"Payload Too Large: File size exceeds maximum allowed limit of {max_file_size} bytes{size_label}"


def validate_file_content(content: bytes, max_file_size: Optional[int] = None) -> None:
    """
    Validate raw byte payload: non-empty and within MAX_FILE_SIZE.

    Raises:
        EmptyFileException: If payload length is 0 bytes.
        FileSizeExceededException: If payload exceeds max_file_size.
    """
    effective_max_size = max_file_size if max_file_size is not None else config.MAX_FILE_SIZE

    if len(content) == 0:
        raise EmptyFileException("Uploaded file is empty")

    if len(content) > effective_max_size:
        raise FileSizeExceededException(file_size_exceeded_message(effective_max_size))


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """Handle custom domain exceptions derived from APIException."""
    logger.warning("APIException on %s %s: [%d] %s", request.method, request.url.path, exc.status_code, exc.message)
    return create_error_response(
        status_code=exc.status_code,
        error=exc.message,
        details=exc.details,
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle Starlette and FastAPI HTTPExceptions (400, 401, 403, 404, 405, 413, etc.).
    """
    detail_msg = exc.detail
    if isinstance(detail_msg, dict) and "error" in detail_msg:
        detail_msg = detail_msg["error"]
    elif exc.status_code == status.HTTP_404_NOT_FOUND and detail_msg == "Not Found":
        detail_msg = f"Not Found: Endpoint '{request.url.path}' does not exist"

    logger.warning("HTTPException on %s %s: [%d] %s", request.method, request.url.path, exc.status_code, detail_msg)
    return create_error_response(
        status_code=exc.status_code,
        error=str(detail_msg),
        headers=getattr(exc, "headers", None),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle FastAPI / Pydantic RequestValidationError.
    Formats field-level errors into human-readable summary.
    If the error relates to missing or empty file uploads, returns HTTP 400 Bad Request.
    Otherwise returns HTTP 422 Unprocessable Entity.
    """
    errors = exc.errors()
    error_messages = []
    is_file_error = False

    for err in errors:
        loc = err.get("loc", ())
        if "file" in loc:
            is_file_error = True
        loc_str = " -> ".join(str(part) for part in loc)
        msg = err.get("msg", "Invalid value")
        error_messages.append(f"{loc_str}: {msg}" if loc_str else msg)

    summary = "; ".join(error_messages) if error_messages else "Request validation failed"
    logger.warning("RequestValidationError on %s %s: %s", request.method, request.url.path, summary)

    safe_details = jsonable_encoder(errors)

    if is_file_error:
        return create_error_response(
            status_code=status.HTTP_400_BAD_REQUEST,
            error=f"Bad Request: Invalid or missing file upload ({summary})",
            details=safe_details,
        )

    return create_error_response(
        status_code=422,  # Unprocessable Content (constant name differs across Starlette versions)
        error=f"Validation error: {summary}",
        details=safe_details,
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle Python ValueErrors (e.g., corrupted image or invalid parsing) as HTTP 400."""
    logger.warning("ValueError on %s %s: %s", request.method, request.url.path, exc)
    return create_error_response(
        status_code=status.HTTP_400_BAD_REQUEST,
        error=f"Bad Request: Invalid image format or corrupted file: {str(exc)}",
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all unhandled server exceptions (HTTP 500).
    Logs the full stack trace; the client gets a generic message with the request ID,
    never the exception text (which can contain file paths or library internals).
    """
    logger.exception("Unhandled server exception on %s %s: %s", request.method, request.url.path, exc)
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error=f"Internal Server Error: unexpected error (request id {request_id_var.get()})",
    )


def register_error_handlers(app: FastAPI) -> None:
    """
    Register all centralized exception handlers with the FastAPI application.
    """
    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
