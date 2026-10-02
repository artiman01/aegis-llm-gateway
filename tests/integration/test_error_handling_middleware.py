"""Integration tests for ErrorHandlingMiddleware domain exception translation."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

from domain.exceptions import (
    CircuitBreakerOpenError,
    NoAvailableProviderError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from presentation.middlewares.error_handling_middleware import ErrorHandlingMiddleware


def create_error_triggering_app() -> FastAPI:
    """Create test application with routes throwing specific domain exceptions."""
    app = FastAPI()
    app.add_middleware(ErrorHandlingMiddleware)

    @app.get("/trigger/circuit-breaker")
    async def trigger_cb() -> None:
        raise CircuitBreakerOpenError(
            provider_name="openai",
            retry_after_seconds=30.0,
            consecutive_failures=5,
        )

    @app.get("/trigger/no-provider")
    async def trigger_no_provider() -> None:
        raise NoAvailableProviderError(
            attempted_providers=["openai", "anthropic"],
            reasons={"openai": "503", "anthropic": "429"},
            model="gpt-4o",
        )

    @app.get("/trigger/rate-limit")
    async def trigger_rate_limit() -> None:
        raise ProviderRateLimitError(
            message="Rate limit exceeded on upstream",
            provider_name="anthropic",
            retry_after=15.0,
        )

    @app.get("/trigger/auth-error")
    async def trigger_auth() -> None:
        raise ProviderAuthenticationError(
            message="Invalid Bearer Token",
            provider_name="openai",
            status_code=401,
        )

    @app.get("/trigger/timeout")
    async def trigger_timeout() -> None:
        raise ProviderTimeoutError(
            message="Gateway timeout to upstream",
            provider_name="openai",
        )

    @app.get("/trigger/unhandled")
    async def trigger_unhandled() -> None:
        raise RuntimeError("database disk image is malformed at /var/data/secrets.db")

    return app


@pytest.fixture
async def error_client() -> AsyncIterator[httpx.AsyncClient]:
    test_app = create_error_triggering_app()
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_circuit_breaker_open_mapped_to_503(error_client: httpx.AsyncClient) -> None:
    """CircuitBreakerOpenError maps to HTTP 503 with Retry-After header and OpenAI schema."""
    resp = await error_client.get("/trigger/circuit-breaker")
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "30"

    data = resp.json()
    assert "error" in data
    assert data["error"]["type"] == "service_unavailable"
    assert data["error"]["code"] == "circuit_breaker_open"
    assert "openai" in data["error"]["message"]


@pytest.mark.asyncio
async def test_no_available_provider_mapped_to_503(error_client: httpx.AsyncClient) -> None:
    """NoAvailableProviderError maps to HTTP 503."""
    resp = await error_client.get("/trigger/no-provider")
    assert resp.status_code == 503

    data = resp.json()
    assert data["error"]["code"] == "all_providers_unavailable"


@pytest.mark.asyncio
async def test_rate_limit_mapped_to_429(error_client: httpx.AsyncClient) -> None:
    """ProviderRateLimitError maps to HTTP 429 with Retry-After header."""
    resp = await error_client.get("/trigger/rate-limit")
    assert resp.status_code == 429
    assert resp.headers.get("Retry-After") == "15"

    data = resp.json()
    assert data["error"]["type"] == "rate_limit_error"


@pytest.mark.asyncio
async def test_auth_error_mapped_to_401(error_client: httpx.AsyncClient) -> None:
    """ProviderAuthenticationError maps to HTTP 401."""
    resp = await error_client.get("/trigger/auth-error")
    assert resp.status_code == 401

    data = resp.json()
    assert data["error"]["type"] == "authentication_error"


@pytest.mark.asyncio
async def test_timeout_mapped_to_504(error_client: httpx.AsyncClient) -> None:
    """ProviderTimeoutError maps to HTTP 504."""
    resp = await error_client.get("/trigger/timeout")
    assert resp.status_code == 504

    data = resp.json()
    assert data["error"]["type"] == "timeout_error"


@pytest.mark.asyncio
async def test_unhandled_exception_sanitized_500(error_client: httpx.AsyncClient) -> None:
    """Unhandled internal exception must return sanitized 500 without disclosing paths or stack."""
    resp = await error_client.get("/trigger/unhandled")
    assert resp.status_code == 500

    data = resp.json()
    assert data["error"]["message"] == "Internal server error"
    assert data["error"]["code"] == "internal_error"
    assert data["error"]["type"] == "api_error"

    raw_text = resp.text
    assert "secrets.db" not in raw_text
    assert "traceback" not in raw_text.lower()
    assert "database disk image" not in raw_text
