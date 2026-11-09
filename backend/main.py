"""
backend/main.py
FastAPI application for Document Fraud Detection Backend.
Provides /health and /verify endpoints with centralized error handling,
file validation, and comprehensive OpenAPI documentation.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from fastapi import FastAPI, File, Query, Request, Response, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

from backend import config
from backend.error_handlers import (
    create_error_response,
    file_size_exceeded_message,
    register_error_handlers,
    validate_file_content,
    validate_file_metadata,
)
from backend.exceptions import (
    APIException,
    BadRequestException,
    FileValidationException,
    InternalServerException,
    ServiceUnavailableException,
)
from backend.logging_config import configure_logging, request_id_var
from backend.middleware import RequestContextMiddleware, UploadSizeLimitMiddleware
from backend.model_utils import ImageValidationError, ModelUnavailableError, model_api

configure_logging(config.LOG_LEVEL)
logger = logging.getLogger(__name__)

STARTED_AT = time.monotonic()
# Bounds how many CPU-heavy analyses run at once. A thread-level semaphore is acquired inside
# the worker thread, so it works regardless of which event loop dispatched the request.
_analysis_slots = threading.BoundedSemaphore(config.MAX_CONCURRENT_ANALYSES)


def validate_startup_configuration() -> List[str]:
    """Log the effective configuration and return warnings about unsafe settings."""
    warnings: List[str] = []
    if config.API_KEY == config.DEFAULT_API_KEY:
        warnings.append("API_KEY is the public default 'default-api-key'; set API_KEY before exposing the service")
    elif len(config.API_KEY) < 16:
        warnings.append("API_KEY is shorter than 16 characters; use a long random secret")
    if "*" in config.CORS_ORIGINS:
        warnings.append("CORS_ORIGINS allows any origin ('*'); restrict it to known frontends in deployment")
    if not 0.0 <= config.TAMPER_THRESHOLD <= 1.0:
        warnings.append(f"TAMPER_THRESHOLD={config.TAMPER_THRESHOLD} is outside [0, 1]")
    logger.info(
        "Configuration: version=%s model=%s max_file_size=%d max_image_pixels=%d "
        "max_concurrent_analyses=%d tamper_threshold=%s cors_origins=%s log_level=%s",
        config.APP_VERSION,
        config.MODEL_PATH,
        config.MAX_FILE_SIZE,
        config.MAX_IMAGE_PIXELS,
        config.MAX_CONCURRENT_ANALYSES,
        config.TAMPER_THRESHOLD,
        ",".join(config.CORS_ORIGINS),
        config.LOG_LEVEL,
    )
    for warning in warnings:
        logger.warning("Startup check: %s", warning)
    return warnings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Validate configuration and load the model at startup.

    A missing or broken model does not stop the server: it starts in a degraded
    state, /health reports it (HTTP 503) and /verify returns 503 while retrying
    the load on each request, so fixing the model file recovers without a restart.
    """
    validate_startup_configuration()
    try:
        if not model_api.load_model():
            logger.error("Starting in DEGRADED mode: %s", model_api.last_error)
    except RuntimeError:
        logger.error("Starting in DEGRADED mode: %s", model_api.last_error)
    yield
    logger.info("Shutting down")


# OpenAPI metadata tags
TAGS_METADATA = [
    {
        "name": "Health",
        "description": "Public system health check and service readiness endpoint.",
    },
    {
        "name": "Verification",
        "description": "Protected document forensic verification and tamper risk analysis endpoints.",
    },
]

# OpenAPI description in rich Markdown
APP_DESCRIPTION = """
## Document Fraud Detection Backend API

Welcome to the **Document Fraud Detection API**, an automated forensic verification platform
designed to detect digital tampering, metadata inconsistencies, and localized forgery in uploaded documents.

---

### Core Capabilities
* **Local Tamper Analysis**: Looks for one part of the page behaving differently from the rest of the same page
  (a noise patch, a local colour cast, a blurred region, inserted text with a different ink level).
* **29 Forensic Features** (model v2, schema `local-tamper-v2`): sizes of the largest anomalous regions plus a
  page-level processing fingerprint (JPEG grid, flatness, colour and ink variety).
* **Machine Learning Classification**: Pre-trained Random Forest classifier returns a **tamper risk score** (0.0 to 1.0) and confidence.
  The score is the forest's vote fraction for the "tampered" class; it is **not calibrated** and is not a fraud probability.
* **Flagged Regions**: Reserved field; currently always an empty list (edits are not localised).

---

### Authentication
All verification endpoints require an API key passed in the request header:
* **Header Name**: `X-API-Key`
* **Default Value**: Configured via the `API_KEY` environment variable.
* **Public Endpoints**: `/health`, `/docs`, `/redoc`, and `/openapi.json` are accessible without authentication.

---

### File Upload Specifications
* **Supported Formats**: JPEG (`.jpg`, `.jpeg`), PNG (`.png`), PDF (`.pdf` — page 1 rendered at 200 DPI).
* **Maximum File Size**: 10 MB (`10,485,760 bytes`), configurable via `MAX_FILE_SIZE`.
* **Empty Files**: 0-byte uploads are strictly rejected.

---

### Standardized Error Format
All errors return consistent JSON responses:
```json
{
  "success": false,
  "error": "Human-readable error description",
  "code": 400,
  "detail": "Human-readable error description"
}
```
"""

app = FastAPI(
    title="Document Fraud Detection API",
    version=config.APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan,
    openapi_tags=TAGS_METADATA,
    contact={
        "name": "Document Fraud Detection Engineering Team",
        "email": "support@documentfraud.internal",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Register centralized error handlers
register_error_handlers(app)

# API Key security scheme for Swagger UI documentation
api_key_header = APIKeyHeader(
    name="X-API-Key", auto_error=False, description="API Key Header required for protected endpoints."
)

PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


async def api_key_validation_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Middleware to validate X-API-Key header for protected endpoints.
    Allows public access to health check and documentation endpoints.
    Returns:
        404 if the requested route does not exist.
        401 if the X-API-Key header is missing.
        403 if the X-API-Key header is invalid.
    """
    path = request.url.path.rstrip("/")
    if path in PUBLIC_PATHS or path == "":
        return await call_next(request)

    # Check if the endpoint exists in registered routes; return 404 if not found
    is_known = False
    for route in request.app.routes:
        match, _ = route.matches(request.scope)
        if match != Match.NONE:
            is_known = True
            break

    if not is_known:
        return create_error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            error=f"Not Found: Endpoint '{request.url.path}' does not exist",
        )

    # Validate API Key
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return create_error_response(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error="Unauthorized: Missing X-API-Key header",
        )

    # Constant-time comparison so response timing does not reveal how much of the key matched
    if not hmac.compare_digest(api_key.encode(), config.API_KEY.encode()):
        return create_error_response(
            status_code=status.HTTP_403_FORBIDDEN,
            error="Forbidden: Invalid API Key",
        )

    return await call_next(request)


# Middleware stack. Starlette runs the LAST added middleware first, so they are added
# innermost-first. Request order: RequestContext -> CORS -> API key -> upload limit -> app.
#  - CORS sits outside the API-key check so browser preflight (OPTIONS, sent without the
#    key) is answered instead of rejected with 401.
#  - The API-key check runs before the upload limit, so unauthenticated uploads are
#    refused without reading their body.
app.add_middleware(
    UploadSizeLimitMiddleware,
    # config is read per request so the limit follows MAX_FILE_SIZE changes (tests patch it)
    max_body_bytes=lambda: config.MAX_FILE_SIZE + config.MULTIPART_OVERHEAD_BYTES,
    message=lambda: file_size_exceeded_message(config.MAX_FILE_SIZE),
)
app.add_middleware(BaseHTTPMiddleware, dispatch=api_key_validation_middleware)
# Authentication uses a header, not cookies, so credentials are only allowed for an
# explicit origin list, never together with "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials="*" not in config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
app.add_middleware(RequestContextMiddleware)


# -------------------------------------------------------------------------
# Pydantic Schemas for Swagger / OpenAPI Documentation
# -------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standardized error response payload schema."""

    success: bool = Field(False, description="Indicates request failure", json_schema_extra={"example": False})
    error: str = Field(
        ..., description="Human-readable error description", json_schema_extra={"example": "Invalid file format"}
    )
    code: int = Field(
        ..., description="HTTP status code corresponding to the error", json_schema_extra={"example": 400}
    )
    detail: Optional[str] = Field(
        None, description="Detailed error explanation", json_schema_extra={"example": "Invalid file format"}
    )
    details: Optional[Any] = Field(None, description="Optional structured validation error diagnostics")


class FlaggedRegion(BaseModel):
    """Bounding box of a suspicious region, in pixels of the uploaded image (PDF: page 1 at 200 DPI)."""

    x: int = Field(..., description="X-coordinate of region top-left corner", json_schema_extra={"example": 160})
    y: int = Field(..., description="Y-coordinate of region top-left corner", json_schema_extra={"example": 296})
    width: int = Field(..., description="Width of flagged region in pixels", json_schema_extra={"example": 152})
    height: int = Field(..., description="Height of flagged region in pixels", json_schema_extra={"example": 24})
    area: int = Field(
        ...,
        description="Pixel area of the anomalous tiles in the region (for an ink run: the run's box)",
        json_schema_extra={"example": 2880},
    )
    type: str = Field(
        "colour",
        description=(
            "Kind of anomaly: 'noise', 'colour', 'softness' or 'ink'; several joined by '+' when regions "
            "of different kinds overlap"
        ),
        json_schema_extra={"example": "colour"},
    )


class VerifyResponse(BaseModel):
    """Document verification response schema containing risk assessment and diagnostics."""

    tamper_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Uncalibrated Random Forest vote fraction for the tampered class (0.0 = genuine, 1.0 = tampered)",
        json_schema_extra={"example": 0.08},
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of model classification (0.0 to 1.0)",
        json_schema_extra={"example": 0.94},
    )
    flagged_regions: List[FlaggedRegion] = Field(
        default_factory=list,
        description=(
            "Suspicious regions, largest first (at most 5), returned only when the score reaches "
            "TAMPER_THRESHOLD and the loaded model can localise edits (model v2); otherwise an empty list"
        ),
    )
    interpretation: str = Field(
        ...,
        description="Human-readable forensic summary explaining model verdict and anomaly counts",
        json_schema_extra={
            "example": "Document verified successfully. Low risk of digital tampering detected (risk score: 0.08, confidence: 0.94)."
        },
    )


class HealthResponse(BaseModel):
    """Service health and operational status response schema."""

    status: str = Field(
        "ok",
        description='"ok" when the model is loaded and requests can be served, "degraded" otherwise',
        json_schema_extra={"example": "ok"},
    )
    version: str = Field(
        config.APP_VERSION,
        description="Current backend service version",
        json_schema_extra={"example": config.APP_VERSION},
    )
    model_loaded: bool = Field(
        ..., description="Whether the fraud model is loaded", json_schema_extra={"example": True}
    )
    model_version: Optional[str] = Field(
        None, description="Loaded model version", json_schema_extra={"example": "2.0.0"}
    )
    uptime_seconds: float = Field(
        ..., description="Seconds since the process started", json_schema_extra={"example": 42.0}
    )
    detail: Optional[str] = Field(None, description="Reason for a degraded status")


# -------------------------------------------------------------------------
# Core Helper Routines
# -------------------------------------------------------------------------


def analyze_document(filename: str, content: bytes) -> Dict[str, Any]:
    """
    Document verification analysis routine.
    Interfaces with model_api to run real model inference on uploaded file content.
    """
    return model_api.predict(content, filename=filename)


def _analyze_with_slot(filename: str, content: bytes) -> Dict[str, Any]:
    """Run analyze_document() while holding one of the MAX_CONCURRENT_ANALYSES slots."""
    with _analysis_slots:
        return analyze_document(filename, content)


# -------------------------------------------------------------------------
# Route Handlers
# -------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="System Health & Readiness Check",
    description=(
        "Public endpoint for load balancers and monitoring. Returns 200 when the model is loaded and "
        "/verify can serve requests, 503 with status 'degraded' otherwise. Does not require authentication."
    ),
    responses={
        200: {
            "model": HealthResponse,
            "description": "API service is healthy and the model is loaded.",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "version": config.APP_VERSION,
                        "model_loaded": True,
                        "model_version": "2.0.0",
                        "uptime_seconds": 42.0,
                        "detail": None,
                    }
                }
            },
        },
        503: {"model": HealthResponse, "description": "Degraded: the model is missing or failed to load."},
    },
)
def health() -> JSONResponse:
    """Report service status, model readiness and uptime (200 ok / 503 degraded)."""
    model = model_api.status()
    body = HealthResponse(
        status="ok" if model["loaded"] else "degraded",
        version=config.APP_VERSION,
        model_loaded=model["loaded"],
        model_version=model["version"],
        uptime_seconds=round(time.monotonic() - STARTED_AT, 1),
        detail=None if model["loaded"] else (model["error"] or "model not loaded yet"),
    )
    return JSONResponse(status_code=200 if model["loaded"] else 503, content=body.model_dump())


@app.post(
    "/verify",
    response_model=VerifyResponse,
    tags=["Verification"],
    summary="Verify uploaded document for tampering and fraud",
    description=(
        "Protected endpoint for document forensic analysis. Uploads an image or document (JPG, JPEG, PNG, PDF), "
        "extracts the model's forensic features (29 local tamper features for model v2), and predicts tamper risk "
        "using a pre-trained Random Forest classifier. Enforces strict file validation (allowed extensions, empty file "
        "checks, and MAX_FILE_SIZE limits)."
    ),
    dependencies=[Security(api_key_header)],
    responses={
        200: {
            "model": VerifyResponse,
            "description": "Document analysis completed successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "tamper_risk_score": 0.08,
                        "confidence": 0.94,
                        "flagged_regions": [],
                        "interpretation": "Document verified successfully. Low risk of digital tampering detected (risk score: 0.08, confidence: 0.94).",
                    }
                }
            },
        },
        400: {
            "model": ErrorResponse,
            "description": "Bad Request — Invalid file format, unsupported extension, missing filename, empty file, or corrupted bytes.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": "Invalid file format: '.txt' is not supported. Allowed formats: jpg, jpeg, png, pdf",
                        "code": 400,
                        "detail": "Invalid file format: '.txt' is not supported. Allowed formats: jpg, jpeg, png, pdf",
                    }
                }
            },
        },
        401: {
            "model": ErrorResponse,
            "description": "Unauthorized — Missing X-API-Key header.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": "Unauthorized: Missing X-API-Key header",
                        "code": 401,
                        "detail": "Unauthorized: Missing X-API-Key header",
                    }
                }
            },
        },
        403: {
            "model": ErrorResponse,
            "description": "Forbidden — Invalid or unrecognized API Key.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": "Forbidden: Invalid API Key",
                        "code": 403,
                        "detail": "Forbidden: Invalid API Key",
                    }
                }
            },
        },
        413: {
            "model": ErrorResponse,
            "description": "Payload Too Large — Uploaded file exceeds configured MAX_FILE_SIZE limit.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": "Payload Too Large: File size exceeds maximum allowed limit of 10485760 bytes (10MB)",
                        "code": 413,
                        "detail": "Payload Too Large: File size exceeds maximum allowed limit of 10485760 bytes (10MB)",
                    }
                }
            },
        },
        422: {
            "model": ErrorResponse,
            "description": "Unprocessable Entity — Request schema or parameter validation failure.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": "Validation error: threshold -> Value must be less than or equal to 1.0",
                        "code": 422,
                        "detail": "threshold -> Value must be less than or equal to 1.0",
                    }
                }
            },
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal Server Error — Unexpected failure during model inference or server processing.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": "Internal Server Error: Unexpected model inference error",
                        "code": 500,
                        "detail": "Internal Server Error: Unexpected model inference error",
                    }
                }
            },
        },
    },
)
async def verify_document(
    request: Request,
    file: Optional[UploadFile] = File(None, description="Document or image file to verify (jpg, jpeg, png, pdf)"),
    threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Reserved for a custom risk threshold (0.0 to 1.0). Validated but currently ignored; the server uses TAMPER_THRESHOLD.",
    ),
) -> Dict[str, Any]:
    """
    Verify uploaded document for tampering.

    Performs comprehensive multi-stage validation:
    1. Checks presence of uploaded file.
    2. Validates filename and supported extension (jpg, jpeg, png, pdf).
    3. Enforces MAX_FILE_SIZE via Content-Length header and streamed byte checks.
    4. Rejects empty (0-byte) payloads.
    5. Executes forensic feature extraction and ML model inference.
    """
    # 400 Bad Request: Check file payload presence
    if file is None:
        raise FileValidationException("Bad Request: Missing required 'file' parameter")

    # 400 Bad Request: Validate filename and extension
    validate_file_metadata(file)

    # Oversized request bodies were already rejected by UploadSizeLimitMiddleware before parsing.
    # Read at most one byte beyond MAX_FILE_SIZE to detect an oversized file part exactly.
    try:
        content = await file.read(config.MAX_FILE_SIZE + 1)
    except Exception as exc:
        logger.warning("Failed to read upload: %r", exc)
        raise BadRequestException("Bad Request: Failed to read file content")

    # 400 / 413 Validation: empty file and exact size limit
    validate_file_content(content, config.MAX_FILE_SIZE)

    # Model inference, offloaded to a worker thread so the event loop stays responsive, with at
    # most MAX_CONCURRENT_ANALYSES running at once (extra requests wait for a slot).
    safe_name = Path(file.filename).name
    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(_analyze_with_slot, safe_name, content)
    except ModelUnavailableError as exc:
        logger.error("Verification unavailable: %s", exc)
        raise ServiceUnavailableException(f"Service Unavailable: Fraud detection model is not available ({exc})")
    except ImageValidationError as exc:
        raise FileValidationException(f"Bad Request: {exc}")
    except ValueError as exc:
        raise BadRequestException(f"Bad Request: Invalid image format or corrupted file: {exc}")
    except APIException:
        raise
    except Exception:
        # Full details go to the log only; clients get a generic message and the request ID.
        logger.exception("Unexpected error while analysing %r (%d bytes)", safe_name, len(content))
        raise InternalServerException(
            f"Internal Server Error: unexpected error during verification (request id {request_id_var.get()})"
        )
    logger.info(
        "Verified %r (%d bytes): risk=%.4f confidence=%.4f in %.0f ms",
        safe_name,
        len(content),
        result["tamper_risk_score"],
        result["confidence"],
        (time.perf_counter() - started) * 1000,
    )
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
