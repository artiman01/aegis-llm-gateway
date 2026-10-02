"""Global error handling middleware translating domain errors to OpenAI canonical JSON format."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import orjson
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from domain.exceptions import (
    AegisLLMError,
    CircuitBreakerOpenError,
    InvalidRequestError,
    NoAvailableProviderError,
    ProviderAuthenticationError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)


def format_openai_error_response(
    status_code: int,
    message: str,
    error_type: str,
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> Response:
    """Format an error into standard OpenAI REST API error schema."""
    body: dict[str, Any] = {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code or str(status_code),
        }
    }
    content = orjson.dumps(body)
    resp_headers = headers or {}
    resp_headers["content-type"] = "application/json"
    return Response(content=content, status_code=status_code, headers=resp_headers)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Catches domain exceptions and serializes them into OpenAI-compliant error responses."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        try:
            return await call_next(request)

        except CircuitBreakerOpenError as exc:
            logger.warning("CircuitBreakerOpen caught in middleware: %s", exc)
            headers = {"Retry-After": str(max(1, int(exc.retry_after_seconds)))}
            return format_openai_error_response(
                status_code=503,
                message=exc.message,
                error_type="service_unavailable",
                code="circuit_breaker_open",
                headers=headers,
            )

        except NoAvailableProviderError as exc:
            logger.error("All providers exhausted for request: %s", exc)
            return format_openai_error_response(
                status_code=503,
                message=exc.message,
                error_type="service_unavailable",
                code="all_providers_unavailable",
            )

        except ProviderRateLimitError as exc:
            logger.warning("Upstream rate limit reached: %s", exc)
            headers = {}
            if exc.retry_after is not None:
                headers["Retry-After"] = str(max(1, int(exc.retry_after)))
            return format_openai_error_response(
                status_code=429,
                message=exc.message,
                error_type="rate_limit_error",
                code="rate_limit_exceeded",
                headers=headers,
            )

        except ProviderAuthenticationError as exc:
            logger.error("Upstream authentication failed: %s", exc)
            return format_openai_error_response(
                status_code=401,
                message=exc.message,
                error_type="authentication_error",
                code="invalid_api_key",
            )

        except (ProviderBadRequestError, InvalidRequestError) as exc:
            logger.info("Bad request from client: %s", exc)
            return format_openai_error_response(
                status_code=400,
                message=str(exc),
                error_type="invalid_request_error",
                code="bad_request",
            )

        except ProviderTimeoutError as exc:
            logger.error("Upstream provider timeout: %s", exc)
            return format_openai_error_response(
                status_code=504,
                message=exc.message,
                error_type="timeout_error",
                code="gateway_timeout",
            )

        except ProviderUnavailableError as exc:
            logger.error("Upstream provider unavailable: %s", exc)
            return format_openai_error_response(
                status_code=503,
                message=exc.message,
                error_type="service_unavailable",
                code="provider_unavailable",
            )

        except AegisLLMError as exc:
            logger.exception("Domain error in gateway: %s", exc)
            return format_openai_error_response(
                status_code=500,
                message="Internal server error",
                error_type="api_error",
                code="internal_error",
            )

        except Exception as exc:
            if isinstance(exc, RequestValidationError):
                logger.info("Request validation failed: %s", exc)
                return format_openai_error_response(
                    status_code=400,
                    message=str(exc),
                    error_type="invalid_request_error",
                    code="bad_request",
                )

            if isinstance(exc, StarletteHTTPException):
                return format_openai_error_response(
                    status_code=exc.status_code,
                    message=str(exc.detail),
                    error_type="invalid_request_error",
                    code=str(exc.status_code),
                )

            logger.exception("Unhandled internal exception in gateway: %s", exc)
            return format_openai_error_response(
                status_code=500,
                message="Internal server error",
                error_type="api_error",
                code="internal_error",
            )
