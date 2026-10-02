"""Prometheus metrics adapter implementing MetricsPort."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from domain.ports.metrics_port import MetricsPort

if TYPE_CHECKING:
    pass


class PrometheusMetricsAdapter(MetricsPort):
    """Adapter collecting Prometheus metrics for latency, tokens, cache, and resilience."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or REGISTRY

        # Request Latency Histogram
        self.request_duration = Histogram(
            name="aegis_request_duration_seconds",
            documentation="Latency of upstream and gateway chat completions in seconds",
            labelnames=["provider", "model", "status"],
            buckets=(0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
            registry=self.registry,
        )

        # TTFT (Time To First Token) for Streaming
        self.stream_ttft = Histogram(
            name="aegis_stream_ttft_seconds",
            documentation="Time to first token (TTFT) for streaming responses in seconds",
            labelnames=["provider", "model"],
            buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
            registry=self.registry,
        )

        # Token Usage Counter
        self.token_counter = Counter(
            name="aegis_tokens_total",
            documentation="Total tokens consumed across models and providers",
            labelnames=["provider", "model", "token_type"],
            registry=self.registry,
        )

        # Cache Hit/Miss Counter
        self.cache_access_counter = Counter(
            name="aegis_cache_requests_total",
            documentation="Cache lookup requests across L1 and L2 tiers",
            labelnames=["tier", "result"],
            registry=self.registry,
        )

        # Circuit Breaker Current State Gauge (0=CLOSED, 1=HALF_OPEN, 2=OPEN)
        self.circuit_breaker_state = Gauge(
            name="aegis_circuit_breaker_state",
            documentation="Current Circuit Breaker state (0=closed, 1=half_open, 2=open)",
            labelnames=["provider"],
            registry=self.registry,
        )

        # Circuit Breaker Transitions Counter
        self.circuit_breaker_transitions = Counter(
            name="aegis_circuit_breaker_transitions_total",
            documentation="Total state transitions in Circuit Breaker state machines",
            labelnames=["provider", "from_state", "to_state"],
            registry=self.registry,
        )

        # Fallback Events Counter
        self.fallback_counter = Counter(
            name="aegis_fallback_events_total",
            documentation="Total dynamic fallback events triggered between providers",
            labelnames=["from_provider", "to_provider", "reason"],
            registry=self.registry,
        )

        # Stream Chunks Emitted
        self.stream_chunks = Counter(
            name="aegis_stream_chunks_total",
            documentation="Total SSE streaming chunks emitted to clients",
            labelnames=["provider", "model"],
            registry=self.registry,
        )

        # Active Concurrent Requests Gauge
        self.active_requests = Gauge(
            name="aegis_active_requests",
            documentation="Number of currently in-flight requests processed by the gateway",
            registry=self.registry,
        )

    def record_request_latency(
        self,
        provider: str,
        model: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        self.request_duration.labels(provider=provider, model=model, status=status).observe(
            duration_seconds
        )

    def record_ttft(self, provider: str, model: str, duration_seconds: float) -> None:
        """Record Time-To-First-Token latency."""
        self.stream_ttft.labels(provider=provider, model=model).observe(duration_seconds)

    def record_token_usage(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        if prompt_tokens > 0:
            self.token_counter.labels(provider, model, "prompt").inc(prompt_tokens)
        if completion_tokens > 0:
            self.token_counter.labels(provider, model, "completion").inc(completion_tokens)

    def record_cache_access(self, tier: str, hit: bool) -> None:
        result_label = "hit" if hit else "miss"
        self.cache_access_counter.labels(tier=tier, result=result_label).inc()

    def record_circuit_breaker_state(self, provider: str, state: str) -> None:
        mapping = {"closed": 0, "half_open": 1, "open": 2}
        numeric_val = mapping.get(state.lower(), 0)
        self.circuit_breaker_state.labels(provider=provider).set(numeric_val)

    def record_circuit_breaker_transition(
        self,
        provider: str,
        from_state: str,
        to_state: str,
    ) -> None:
        self.circuit_breaker_transitions.labels(
            provider=provider,
            from_state=from_state,
            to_state=to_state,
        ).inc()

    def record_fallback(
        self,
        from_provider: str,
        to_provider: str,
        reason: str,
    ) -> None:
        # Truncate reason to avoid high cardinality explosion
        clean_reason = reason.split("\n", maxsplit=1)[0][:60]
        self.fallback_counter.labels(
            from_provider=from_provider,
            to_provider=to_provider,
            reason=clean_reason,
        ).inc()

    def record_stream_chunk(self, provider: str, model: str) -> None:
        self.stream_chunks.labels(provider=provider, model=model).inc()

    def set_active_requests(self, count: int) -> None:
        self.active_requests.set(count)

    def export_metrics(self) -> tuple[bytes, str]:
        """Export serialized Prometheus metrics and format content type."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
