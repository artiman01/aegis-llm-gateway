"""Unit tests for PrometheusMetricsAdapter."""

from __future__ import annotations

from prometheus_client import CollectorRegistry

from infrastructure.observability.prometheus_metrics_adapter import (
    PrometheusMetricsAdapter,
)


def test_prometheus_metrics_collection_and_export() -> None:
    """Validate all MetricsPort methods emit prometheus format text."""
    registry = CollectorRegistry()
    metrics = PrometheusMetricsAdapter(registry=registry)

    # 1. Latency & TTFT
    metrics.record_request_latency(
        provider="openai", model="gpt-4o", status="success", duration_seconds=0.25
    )
    metrics.record_ttft(provider="openai", model="gpt-4o", duration_seconds=0.08)

    # 2. Token usage
    metrics.record_token_usage(
        provider="openai", model="gpt-4o", prompt_tokens=100, completion_tokens=50
    )

    # 3. Cache accesses
    metrics.record_cache_access(tier="l1_exact", hit=True)
    metrics.record_cache_access(tier="l2_semantic", hit=False)

    # 4. Circuit Breaker states and transitions
    metrics.record_circuit_breaker_state(provider="anthropic", state="open")
    metrics.record_circuit_breaker_transition(
        provider="anthropic", from_state="closed", to_state="open"
    )

    # 5. Fallback and chunks
    metrics.record_fallback(from_provider="openai", to_provider="anthropic", reason="HTTP 503")
    metrics.record_stream_chunk(provider="anthropic", model="claude-3-5-sonnet")
    metrics.set_active_requests(5)

    # Export
    raw_data, content_type = metrics.export_metrics()
    text = raw_data.decode("utf-8")

    assert "text/plain" in content_type
    assert "aegis_request_duration_seconds" in text
    prompt_metric = 'aegis_tokens_total{model="gpt-4o",provider="openai",token_type="prompt"} 100.0'
    comp_metric = (
        'aegis_tokens_total{model="gpt-4o",provider="openai",token_type="completion"} 50.0'
    )
    assert prompt_metric in text
    assert comp_metric in text
    assert 'aegis_cache_requests_total{result="hit",tier="l1_exact"} 1.0' in text
    assert 'aegis_cache_requests_total{result="miss",tier="l2_semantic"} 1.0' in text
    assert 'aegis_circuit_breaker_state{provider="anthropic"} 2.0' in text
    assert "aegis_fallback_events_total" in text
    assert "aegis_active_requests 5.0" in text
