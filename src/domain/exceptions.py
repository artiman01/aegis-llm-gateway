"""Domain-level exceptions for AegisLLM Gateway.

All exceptions in this module are independent of external frameworks (e.g. FastAPI, HTTPX),
strictly representing domain errors, operational faults, and resilience triggers.
"""

from __future__ import annotations

from typing import Any


class AegisLLMError(Exception):
    """Base exception for all AegisLLM Gateway errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class DomainError(AegisLLMError):
    """Base class for business logic and domain rule violations."""


# Provider & Upstream Exceptions


class ProviderError(DomainError):
    """Base exception for errors communicating with upstream LLM providers."""

    def __init__(
        self,
        message: str,
        provider_name: str,
        status_code: int | None = None,
        raw_response: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = details or {}
        merged_details.update(
            {
                "provider_name": provider_name,
                "status_code": status_code,
                "raw_response": raw_response,
            }
        )
        super().__init__(message, merged_details)
        self.provider_name = provider_name
        self.status_code = status_code
        self.raw_response = raw_response


class ProviderUnavailableError(ProviderError):
    """Raised when an upstream provider is offline, unreachable, or returns 5xx."""


class ProviderTimeoutError(ProviderError):
    """Raised when an upstream provider exceeds connection or read timeout."""


class ProviderRateLimitError(ProviderError):
    """Raised when an upstream provider throttles requests (HTTP 429)."""

    def __init__(
        self,
        message: str,
        provider_name: str,
        *,
        retry_after: float | None = None,
        status_code: int | None = 429,
        raw_response: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        retry_details = details or {}
        retry_details["retry_after"] = retry_after
        super().__init__(
            message=message,
            provider_name=provider_name,
            status_code=status_code,
            raw_response=raw_response,
            details=retry_details,
        )
        self.retry_after = retry_after


class ProviderAuthenticationError(ProviderError):
    """Raised when API credentials are rejected (HTTP 401 / 403)."""


class ProviderBadRequestError(ProviderError):
    """Raised when upstream provider rejects request payload (HTTP 400)."""


class ModelNotFoundError(ProviderError):
    """Raised when requested model is not supported or not recognized."""


# Resilience & Circuit Breaker Exceptions


class CircuitBreakerError(DomainError):
    """Base exception for circuit breaker faults."""


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when calls are immediately blocked because the Circuit Breaker is OPEN."""

    def __init__(
        self,
        provider_name: str,
        retry_after_seconds: float,
        consecutive_failures: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = details or {}
        merged_details.update(
            {
                "provider_name": provider_name,
                "retry_after_seconds": retry_after_seconds,
                "consecutive_failures": consecutive_failures,
            }
        )
        msg = (
            f"Circuit breaker for provider '{provider_name}' is OPEN. "
            f"Fast-failing. Cooldown remaining: {retry_after_seconds:.2f}s."
        )
        super().__init__(msg, merged_details)
        self.provider_name = provider_name
        self.retry_after_seconds = retry_after_seconds
        self.consecutive_failures = consecutive_failures


class CircuitBreakerHalfOpenCapacityExceededError(CircuitBreakerError):
    """Raised when HALF_OPEN state trial capacity is full."""


# Routing Exceptions


class RoutingError(DomainError):
    """Base exception for dynamic routing and fallback dispatching."""


class NoAvailableProviderError(RoutingError):
    """Raised when primary and all fallback providers failed or are unavailable."""

    def __init__(
        self,
        attempted_providers: list[str],
        reasons: dict[str, str],
        model: str,
    ) -> None:
        msg = (
            f"All providers exhausted for model '{model}'. "
            f"Attempted: {attempted_providers}. Errors: {reasons}"
        )
        super().__init__(
            msg,
            {
                "attempted_providers": attempted_providers,
                "reasons": reasons,
                "model": model,
            },
        )
        self.attempted_providers = attempted_providers
        self.reasons = reasons
        self.model = model


class InvalidRoutingConfigError(RoutingError):
    """Raised when routing matrix or fallback chain is misconfigured."""


# Cache Exceptions


class CacheError(DomainError):
    """Base exception for caching operations."""


class CacheUnavailableError(CacheError):
    """Raised when cache tier is unreachable (Redis or memory fault)."""


# Validation Exceptions


class InvalidRequestError(DomainError):
    """Raised when incoming client request violates domain rules."""
