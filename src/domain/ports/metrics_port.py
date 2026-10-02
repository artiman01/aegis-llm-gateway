"""Port definition for telemetry and operational metrics."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MetricsPort(Protocol):
    """Port for emitting Prometheus metrics and observability events."""

    def record_request_latency(
        self,
        provider: str,
        model: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        """Record upstream and end-to-end request duration histogram."""
        ...

    def record_token_usage(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Increment token consumption counters."""
        ...

    def record_cache_access(
        self,
        tier: str,
        hit: bool,
    ) -> None:
        """Record cache lookup (tier: 'l1_exact' or 'l2_semantic', hit: True/False)."""
        ...

    def record_circuit_breaker_state(
        self,
        provider: str,
        state: str,
    ) -> None:
        """Export current circuit breaker gauge (closed=0, half_open=1, open=2)."""
        ...

    def record_circuit_breaker_transition(
        self,
        provider: str,
        from_state: str,
        to_state: str,
    ) -> None:
        """Increment state transition counter for circuit breakers."""
        ...

    def record_fallback(
        self,
        from_provider: str,
        to_provider: str,
        reason: str,
    ) -> None:
        """Record fallback routing activation."""
        ...

    def record_stream_chunk(
        self,
        provider: str,
        model: str,
    ) -> None:
        """Record emitted SSE streaming chunk."""
        ...

    def set_active_requests(
        self,
        count: int,
    ) -> None:
        """Update gauge of currently active concurrent requests."""
        ...
