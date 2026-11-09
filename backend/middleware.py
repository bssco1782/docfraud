"""
backend/middleware.py
Pure-ASGI middleware for request tracing and upload-size enforcement.

RequestContextMiddleware
    Assigns each request an ID (the client's X-Request-ID if it is short and safe,
    otherwise a new one), exposes it to logging, returns it in the X-Request-ID
    response header and writes one access-log line per request.

UploadSizeLimitMiddleware
    Rejects oversized request bodies *before* they are parsed. FastAPI parses the
    multipart form (and spools the upload to disk) before the /verify handler runs,
    so a size check inside the handler only happens after the whole body has been
    received. This middleware refuses a too-large Content-Length immediately, and
    counts bytes for chunked uploads that send no Content-Length.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Awaitable, Callable, Dict, List, MutableMapping, Optional

from backend.logging_config import request_id_var

logger = logging.getLogger("backend.access")

Scope = MutableMapping[str, object]
Message = MutableMapping[str, object]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _header(scope: Scope, name: bytes) -> Optional[str]:
    """Return a request header value (latin-1 decoded) or None."""
    for key, value in scope.get("headers", []):  # type: ignore[union-attr]
        if key == name:
            return value.decode("latin-1")
    return None


def _json_error(status: int, message: str) -> tuple[Message, Message]:
    """Build ASGI start/body messages for the standard error envelope."""
    body = json.dumps(
        {"success": False, "error": message, "code": status, "detail": message}, separators=(",", ":")
    ).encode()
    start: Message = {
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
    }
    return start, {"type": "http.response.body", "body": body}


class RequestContextMiddleware:
    """Request ID propagation and one access-log line per HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the downstream ASGI app."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Set the request ID, time the request and log its outcome."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = _header(scope, b"x-request-id")
        request_id = incoming if incoming and _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        status_holder: Dict[str, int] = {"status": 500}
        start = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = int(message["status"])  # type: ignore[arg-type]
                headers: List = list(message.get("headers", []))  # type: ignore[arg-type]
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            status = status_holder["status"]
            level = logging.ERROR if status >= 500 else logging.WARNING if status >= 400 else logging.INFO
            logger.log(level, "%s %s -> %d (%.0f ms)", scope.get("method"), scope.get("path"), status, elapsed_ms)
            request_id_var.reset(token)


class UploadSizeLimitMiddleware:
    """Reject request bodies larger than a limit before they are parsed."""

    def __init__(self, app: ASGIApp, max_body_bytes: Callable[[], int], message: Callable[[], str]) -> None:
        """
        Args:
            app: downstream ASGI app.
            max_body_bytes: returns the current body limit (read per request so tests can patch config).
            message: returns the 413 error message.
        """
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.message = message

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Enforce the limit from Content-Length, or by counting streamed bytes."""
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        limit = self.max_body_bytes()
        declared = _header(scope, b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await self._reject(send)
            return

        state = {"received": 0, "exceeded": False, "replaced": False}

        async def counting_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                state["received"] += len(message.get("body", b""))  # type: ignore[arg-type]
                if state["received"] > limit:
                    state["exceeded"] = True
                    raise _BodyTooLarge()
            return message

        async def guarded_send(message: Message) -> None:
            # Once the limit is hit, replace whatever the app answers (it sees a parse error) with a 413.
            if state["exceeded"]:
                if not state["replaced"] and message["type"] == "http.response.start":
                    state["replaced"] = True
                    await self._reject(send)
                return
            await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except _BodyTooLarge:
            if not state["replaced"]:
                await self._reject(send)

    async def _reject(self, send: Send) -> None:
        """Send the standard 413 error response."""
        start, body = _json_error(413, self.message())
        await send(start)
        await send(body)


class _BodyTooLarge(Exception):
    """Internal signal: the streamed request body exceeded the limit."""
