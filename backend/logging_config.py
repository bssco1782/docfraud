"""
backend/logging_config.py
Logging setup for the API: one consistent line format with a per-request ID.

Every log record gets a `request_id` attribute from a context variable set by
RequestContextMiddleware, so all lines written while handling one request can be
correlated (the same ID is returned to the client in the X-Request-ID header).
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [req=%(request_id)s] %(message)s"
# Loggers whose records use LOG_FORMAT; uvicorn keeps its own format for its own loggers.
APP_LOGGERS = ("backend", "core")


class RequestIdFilter(logging.Filter):
    """Attach the current request ID (or '-') to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add `record.request_id` and always keep the record."""
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Attach a single formatted stream handler to the backend and core loggers (idempotent)."""
    for name in APP_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if not any(getattr(h, "_docfraud", False) for h in logger.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            handler.addFilter(RequestIdFilter())
            handler._docfraud = True  # type: ignore[attr-defined]
            logger.addHandler(handler)
