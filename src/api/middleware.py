"""Request logging middleware."""
from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.monitoring.logging import get_logger
from src.monitoring.metrics import API_REQUEST_DURATION_SECONDS, API_REQUESTS_TOTAL

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with timing, correlation ID, and status code."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        request_id = str(uuid.uuid4())[:8]
        t0 = time.monotonic()

        response = await call_next(request)

        duration = time.monotonic() - t0
        logger.info(
            "HTTP request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration * 1000, 1),
            request_id=request_id,
        )

        # Metrics
        API_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(response.status_code),
        ).inc()
        API_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        response.headers["X-Request-ID"] = request_id
        return response


# Import needed for the type hint in dispatch
from typing import Any  # noqa: E402
