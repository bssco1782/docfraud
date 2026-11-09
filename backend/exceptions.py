"""
backend/exceptions.py
Centralized custom exceptions hierarchy for Document Fraud Detection API.
Provides typed domain exceptions for consistent HTTP status codes and error messages.
"""

from typing import Any, Optional


class APIException(Exception):
    """
    Base exception class for all custom API-level exceptions.

    Attributes:
        message (str): Human-readable error explanation.
        status_code (int): Associated HTTP status code.
        details (Optional[Any]): Optional structured metadata or validation diagnostics.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Any] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


class BadRequestException(APIException):
    """HTTP 400 Bad Request — General client-side invalid request."""

    def __init__(self, message: str = "Bad Request", details: Optional[Any] = None):
        super().__init__(message=message, status_code=400, details=details)


class FileValidationException(BadRequestException):
    """
    HTTP 400 Bad Request — Raised when uploaded document fails format,
    extension, or structural validation.
    """

    def __init__(
        self,
        message: str = "Invalid file format",
        details: Optional[Any] = None,
    ):
        super().__init__(message=message, details=details)


class EmptyFileException(BadRequestException):
    """HTTP 400 Bad Request — Raised when uploaded file payload is 0 bytes."""

    def __init__(
        self,
        message: str = "Uploaded file is empty",
        details: Optional[Any] = None,
    ):
        super().__init__(message=message, details=details)


class AuthenticationException(APIException):
    """
    HTTP 401 Unauthorized — Raised when X-API-Key header is absent
    or unauthenticated.
    """

    def __init__(
        self,
        message: str = "Unauthorized: Missing X-API-Key header",
        details: Optional[Any] = None,
    ):
        super().__init__(message=message, status_code=401, details=details)


class ForbiddenException(APIException):
    """
    HTTP 403 Forbidden — Raised when provided X-API-Key is invalid
    or does not grant required permissions.
    """

    def __init__(
        self,
        message: str = "Forbidden: Invalid API Key",
        details: Optional[Any] = None,
    ):
        super().__init__(message=message, status_code=403, details=details)


class NotFoundException(APIException):
    """HTTP 404 Not Found — Raised when requested resource or endpoint is not found."""

    def __init__(
        self,
        message: str = "Resource not found",
        details: Optional[Any] = None,
    ):
        super().__init__(message=message, status_code=404, details=details)


class FileSizeExceededException(APIException):
    """
    HTTP 413 Payload Too Large — Raised when uploaded file exceeds
    the configured MAX_FILE_SIZE limit.
    """

    def __init__(
        self,
        message: str = "Payload Too Large: File size exceeds maximum allowed limit",
        details: Optional[Any] = None,
    ):
        super().__init__(message=message, status_code=413, details=details)


class RequestValidationException(APIException):
    """
    HTTP 422 Unprocessable Entity — Raised when schema validation fails
    on request parameters or body.
    """

    def __init__(
        self,
        message: str = "Validation error: Invalid request parameters",
        details: Optional[Any] = None,
    ):
        super().__init__(message=message, status_code=422, details=details)


class InternalServerException(APIException):
    """HTTP 500 Internal Server Error — Raised on unexpected server or pipeline errors."""

    def __init__(
        self,
        message: str = "Internal Server Error: Unexpected error occurred",
        details: Optional[Any] = None,
    ):
        super().__init__(message=message, status_code=500, details=details)


class ServiceUnavailableException(APIException):
    """HTTP 503 Service Unavailable — Raised when the fraud model is not loaded or cannot be loaded."""

    def __init__(
        self,
        message: str = "Service Unavailable: Fraud detection model is not loaded",
        details: Optional[Any] = None,
    ):
        super().__init__(message=message, status_code=503, details=details)
