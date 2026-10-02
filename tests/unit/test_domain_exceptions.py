"""Unit tests for domain exceptions."""

from __future__ import annotations

from domain.exceptions import (
    AegisLLMError,
    CircuitBreakerOpenError,
    DomainError,
    NoAvailableProviderError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


def test_exception_hierarchy() -> None:
    """Ensure all custom domain exceptions inherit from DomainError."""
    err = ProviderUnavailableError("Provider offline", provider_name="openai", status_code=503)
    assert isinstance(err, DomainError)
    assert isinstance(err, AegisLLMError)
    assert isinstance(err, ProviderError)
    assert err.provider_name == "openai"
    assert err.status_code == 503


def test_provider_rate_limit_error() -> None:
    """Rate limit error must preserve retry_after parameter."""
    err = ProviderRateLimitError(
        message="Too many requests",
        provider_name="anthropic",
        retry_after=5.5,
    )
    assert err.provider_name == "anthropic"
    assert err.retry_after == 5.5
    assert err.status_code == 429
    assert "retry_after" in err.details


def test_provider_timeout_error() -> None:
    """Timeout error must capture provider identity."""
    err = ProviderTimeoutError(
        message="Upstream request timed out after 30s",
        provider_name="mock",
    )
    assert err.provider_name == "mock"
    assert "mock" in str(err)


def test_circuit_breaker_open_error() -> None:
    """Circuit breaker open error should format diagnostic messages."""
    err = CircuitBreakerOpenError(
        provider_name="openai",
        retry_after_seconds=24.5,
        consecutive_failures=5,
    )
    assert err.provider_name == "openai"
    assert err.retry_after_seconds == 24.5
    assert err.consecutive_failures == 5
    assert "openai" in str(err)
    assert "24.50s" in str(err)


def test_no_available_provider_error() -> None:
    """NoAvailableProviderError aggregates failure reasons across all attempts."""
    err = NoAvailableProviderError(
        attempted_providers=["openai", "anthropic"],
        reasons={"openai": "CircuitBreakerOpen", "anthropic": "Timeout"},
        model="gpt-4o",
    )
    assert err.model == "gpt-4o"
    assert err.attempted_providers == ["openai", "anthropic"]
    assert "openai" in err.reasons
    assert "CircuitBreakerOpen" in err.reasons["openai"]
