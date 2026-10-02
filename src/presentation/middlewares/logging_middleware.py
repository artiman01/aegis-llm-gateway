"""Structured logging middleware tracking incoming HTTP requests and response statuses."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("aegisllm.access")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logs incoming API requests with client IP, path, duration, and status."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start_time = time.monotonic()
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path

        try:
            response = await call_next(request)
            duration_ms = (time.monotonic() - start_time) * 1000.0
            logger.info(
                "%s %s - %d (%s) in %.2fms",
                method,
                path,
                response.status_code,
                client_ip,
                duration_ms,
            )
            return response
        except Exception:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            logger.error(
                "%s %s - EXCEPTION (%s) in %.2fms",
                method,
                path,
                client_ip,
                duration_ms,
            )
            raise
