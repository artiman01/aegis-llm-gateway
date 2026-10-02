"""Unit tests for CircuitBreakerService Finite State Machine (FSM)."""

from __future__ import annotations

import asyncio

import pytest

from application.services.circuit_breaker_service import CircuitBreakerService
from domain.exceptions import CircuitBreakerOpenError
from domain.models.circuit_breaker import CircuitBreakerConfig, CircuitBreakerState


class DummyMetricsPort:
    """Mock metrics port tracking state transitions."""

    def __init__(self) -> None:
        self.transitions: list[tuple[str, str, str]] = []
        self.states: list[tuple[str, str]] = []

    def record_circuit_breaker_transition(
        self,
        provider: str,
        from_state: str,
        to_state: str,
    ) -> None:
        self.transitions.append((provider, from_state, to_state))

    def record_circuit_breaker_state(self, provider: str, state: str) -> None:
        self.states.append((provider, state))

    def record_request_latency(
        self, provider: str, model: str, status: str, duration_seconds: float
    ) -> None:
        pass

    def record_token_usage(
        self, provider: str, model: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        pass

    def record_cache_access(self, tier: str, hit: bool) -> None:
        pass

    def record_fallback(self, from_provider: str, to_provider: str, reason: str) -> None:
        pass

    def record_stream_chunk(self, provider: str, model: str) -> None:
        pass

    def set_active_requests(self, count: int) -> None:
        pass


@pytest.fixture
def metrics_mock() -> DummyMetricsPort:
    return DummyMetricsPort()


@pytest.fixture
def test_config() -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=0.1,  # Fast 100ms timeout for deterministic test
        half_open_success_threshold=2,
        half_open_max_trials=1,
    )


@pytest.mark.asyncio
async def test_initial_state_closed(
    metrics_mock: DummyMetricsPort, test_config: CircuitBreakerConfig
) -> None:
    """Circuit breaker begins in CLOSED state allowing traffic."""
    cb = CircuitBreakerService(default_config=test_config, metrics=metrics_mock)
    assert await cb.get_state("openai") == CircuitBreakerState.CLOSED

    # Should not raise
    await cb.acquire_execution_permission("openai")


@pytest.mark.asyncio
async def test_closed_to_open_transition(
    metrics_mock: DummyMetricsPort, test_config: CircuitBreakerConfig
) -> None:
    """Failures exceeding threshold trip state from CLOSED to OPEN."""
    cb = CircuitBreakerService(default_config=test_config, metrics=metrics_mock)

    # 2 failures (threshold is 3) -> still CLOSED
    await cb.record_failure("openai", "timeout 1")
    await cb.record_failure("openai", "timeout 2")
    assert await cb.get_state("openai") == CircuitBreakerState.CLOSED

    # 3rd failure -> trips to OPEN
    await cb.record_failure("openai", "timeout 3")
    assert await cb.get_state("openai") == CircuitBreakerState.OPEN

    # Subsequent call fails-fast with CircuitBreakerOpenError
    with pytest.raises(CircuitBreakerOpenError) as exc_info:
        await cb.acquire_execution_permission("openai")
    assert exc_info.value.provider_name == "openai"
    assert exc_info.value.consecutive_failures == 3

    # Check telemetry emission
    assert ("openai", "closed", "open") in metrics_mock.transitions


@pytest.mark.asyncio
async def test_open_to_half_open_and_recovery_to_closed(
    metrics_mock: DummyMetricsPort, test_config: CircuitBreakerConfig
) -> None:
    """After recovery timeout, canary probes test provider and restore to CLOSED."""
    cb = CircuitBreakerService(default_config=test_config, metrics=metrics_mock)

    # Trip to OPEN
    for _ in range(3):
        await cb.record_failure("openai", "500 Server Error")
    assert await cb.get_state("openai") == CircuitBreakerState.OPEN

    # Wait for recovery timeout (100ms)
    await asyncio.sleep(0.12)

    # First request after cooldown enters HALF_OPEN
    await cb.acquire_execution_permission("openai")
    assert await cb.get_state("openai") == CircuitBreakerState.HALF_OPEN
    assert ("openai", "open", "half_open") in metrics_mock.transitions

    # Probe 1 succeeds
    await cb.record_success("openai")
    # Still HALF_OPEN because success_threshold is 2
    assert await cb.get_state("openai") == CircuitBreakerState.HALF_OPEN

    # Probe 2 acquires permission and succeeds
    await cb.acquire_execution_permission("openai")
    await cb.record_success("openai")

    # Reached success_threshold -> restored to CLOSED
    assert await cb.get_state("openai") == CircuitBreakerState.CLOSED
    assert ("openai", "half_open", "closed") in metrics_mock.transitions


@pytest.mark.asyncio
async def test_half_open_failure_immediately_reopens(
    metrics_mock: DummyMetricsPort, test_config: CircuitBreakerConfig
) -> None:
    """Any failure during HALF_OPEN immediately trips back to OPEN."""
    cb = CircuitBreakerService(default_config=test_config, metrics=metrics_mock)

    for _ in range(3):
        await cb.record_failure("openai", "503 Unavailable")

    await asyncio.sleep(0.12)

    # Enter HALF_OPEN
    await cb.acquire_execution_permission("openai")
    assert await cb.get_state("openai") == CircuitBreakerState.HALF_OPEN

    # Canary probe fails
    await cb.record_failure("openai", "Still failing")
    assert await cb.get_state("openai") == CircuitBreakerState.OPEN
    assert ("openai", "half_open", "open") in metrics_mock.transitions


@pytest.mark.asyncio
async def test_half_open_limits_canary_probes(test_config: CircuitBreakerConfig) -> None:
    """When in HALF_OPEN, concurrent requests beyond half_open_max_trials fail-fast."""
    cb = CircuitBreakerService(default_config=test_config)

    for _ in range(3):
        await cb.record_failure("openai", "500 Internal")

    await asyncio.sleep(0.12)

    # Probe 1 permitted (half_open_max_trials = 1)
    await cb.acquire_execution_permission("openai")

    # Second concurrent probe attempt must be rejected to prevent overwhelming recovering server
    with pytest.raises(CircuitBreakerOpenError):
        await cb.acquire_execution_permission("openai")


@pytest.mark.asyncio
async def test_reset_circuit_breaker() -> None:
    """Manual reset returns breaker to CLOSED and resets counters."""
    cb = CircuitBreakerService()
    for _ in range(5):
        await cb.record_failure("openai", "error")

    snapshot_open = await cb.get_snapshot("openai")
    assert snapshot_open.state == CircuitBreakerState.OPEN
    assert snapshot_open.consecutive_failures == 5

    await cb.reset("openai")

    snapshot_reset = await cb.get_snapshot("openai")
    assert snapshot_reset.state == CircuitBreakerState.CLOSED
    assert snapshot_reset.consecutive_failures == 0


@pytest.mark.asyncio
async def test_circuit_breaker_lease_lifecycle(test_config: CircuitBreakerConfig) -> None:
    """CircuitBreakerLease records success, failure, and safe probe cancellation."""
    cb = CircuitBreakerService(default_config=test_config)

    # 1. Acquire lease and record success
    lease1 = await cb.acquire_lease("openai")
    assert lease1.provider_name == "openai"
    await lease1.record_success()
    # Idempotent: repeated calls do nothing
    await lease1.record_success()

    # 2. Trip to OPEN using failures via lease
    for _ in range(3):
        lease = await cb.acquire_lease("openai")
        await lease.record_failure(RuntimeError("error"))
    assert await cb.get_state("openai") == CircuitBreakerState.OPEN

    # 3. Wait for recovery timeout to enter HALF_OPEN
    await asyncio.sleep(0.12)
    lease_half = await cb.acquire_lease("openai")
    assert await cb.get_state("openai") == CircuitBreakerState.HALF_OPEN

    # 4. Cancel lease: should decrement in_flight_probes without tripping to OPEN
    await lease_half.record_cancelled()
    snapshot = await cb.get_snapshot("openai")
    assert snapshot.state == CircuitBreakerState.HALF_OPEN

    # Since in_flight_probes was decremented, another probe can be acquired
    lease_probe = await cb.acquire_lease("openai")
    await lease_probe.record_success()
