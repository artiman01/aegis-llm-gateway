"""Application service implementing Circuit Breaker Finite State Machine (FSM).

Guarantees thread-safe and asyncio-task-safe state transitions, monotonic clock
precision, canary probe throttling in HALF_OPEN, and Prometheus telemetry integration.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from domain.exceptions import CircuitBreakerOpenError
from domain.models.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerSnapshot,
    CircuitBreakerState,
)

if TYPE_CHECKING:
    from domain.ports.metrics_port import MetricsPort


@dataclass
class _ProviderCircuitState:
    """Internal mutable state tracker for a single provider."""

    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_failures: int = 0
    total_successes: int = 0
    in_flight_probes: int = 0
    last_state_change: float = field(default_factory=time.monotonic)
    last_failure_time: float | None = None
    last_error: str | None = None


class CircuitBreakerLease:
    """Execution token representing granted permission to call an upstream provider.

    Ensures safe, asynchronous reporting of call outcomes (success, failure, or cancellation).
    """

    def __init__(
        self,
        service: CircuitBreakerService,
        provider_name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        self._service = service
        self._provider_name = provider_name
        self._config = config
        self._completed = False

    @property
    def provider_name(self) -> str:
        """Name of provider associated with this lease."""
        return self._provider_name

    async def record_success(self) -> None:
        """Record success, transitioning HALF_OPEN -> CLOSED if threshold reached."""
        if self._completed:
            return
        self._completed = True
        await self._service.record_success(self._provider_name, self._config)

    async def record_failure(self, exc: Exception | str) -> None:
        """Record failure, transitioning CLOSED -> OPEN if threshold reached."""
        if self._completed:
            return
        self._completed = True
        await self._service.record_failure(self._provider_name, exc, self._config)

    async def record_cancelled(self) -> None:
        """Safely decrement in_flight_probes in HALF_OPEN without incrementing failure counter."""
        if self._completed:
            return
        self._completed = True
        await self._service.record_cancelled(self._provider_name, self._config)


class CircuitBreakerService:
    """Orchestrates Circuit Breaker state machines across all LLM providers."""

    def __init__(
        self,
        default_config: CircuitBreakerConfig | None = None,
        metrics: MetricsPort | None = None,
    ) -> None:
        self._default_config = default_config or CircuitBreakerConfig()
        self._metrics = metrics
        self._states: dict[str, _ProviderCircuitState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def _get_lock(self, provider_name: str) -> asyncio.Lock:
        """Retrieve or create an asyncio.Lock for a specific provider."""
        if provider_name not in self._locks:
            async with self._global_lock:
                if provider_name not in self._locks:
                    self._locks[provider_name] = asyncio.Lock()
        return self._locks[provider_name]

    def _get_or_create_state(self, provider_name: str) -> _ProviderCircuitState:
        """Internal helper to get or initialize state."""
        if provider_name not in self._states:
            self._states[provider_name] = _ProviderCircuitState()
        return self._states[provider_name]

    async def acquire_execution_permission(
        self,
        provider_name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Verify whether an upstream call can proceed or should fail-fast.

        Raises:
            CircuitBreakerOpenError: If the circuit is OPEN or probe capacity in HALF_OPEN is full.
        """
        cfg = config or self._default_config
        lock = await self._get_lock(provider_name)

        async with lock:
            circuit = self._get_or_create_state(provider_name)
            now = time.monotonic()

            if circuit.state == CircuitBreakerState.OPEN:
                elapsed = now - circuit.last_state_change
                if elapsed >= cfg.recovery_timeout_seconds:
                    # Transition OPEN -> HALF_OPEN (cooldown expired, begin probing)
                    self._transition_state(
                        provider_name=provider_name,
                        circuit=circuit,
                        new_state=CircuitBreakerState.HALF_OPEN,
                        now=now,
                    )
                    circuit.consecutive_successes = 0
                    circuit.in_flight_probes = 1
                    return
                # Still within OPEN cooldown period: fail-fast immediately
                remaining = cfg.recovery_timeout_seconds - elapsed
                raise CircuitBreakerOpenError(
                    provider_name=provider_name,
                    retry_after_seconds=max(0.0, remaining),
                    consecutive_failures=circuit.consecutive_failures,
                )

            if circuit.state == CircuitBreakerState.HALF_OPEN:
                if circuit.in_flight_probes >= cfg.half_open_max_trials:
                    # Canary trial capacity saturated; fast-fail extra tasks to fallback
                    raise CircuitBreakerOpenError(
                        provider_name=provider_name,
                        retry_after_seconds=cfg.recovery_timeout_seconds,
                        consecutive_failures=circuit.consecutive_failures,
                    )
                circuit.in_flight_probes += 1
                return

            # CLOSED state: traffic allowed unconditionally

    async def acquire_lease(
        self,
        provider_name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreakerLease:
        """Acquire an execution lease for a provider, validating circuit availability.

        Raises:
            CircuitBreakerOpenError: If the circuit is OPEN or probe capacity in HALF_OPEN is full.
        """
        await self.acquire_execution_permission(provider_name, config)
        return CircuitBreakerLease(self, provider_name, config)

    async def record_cancelled(
        self,
        provider_name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Safely release an in-flight probe if a task was cancelled without failing."""
        lock = await self._get_lock(provider_name)
        async with lock:
            circuit = self._get_or_create_state(provider_name)
            if circuit.state == CircuitBreakerState.HALF_OPEN:
                circuit.in_flight_probes = max(0, circuit.in_flight_probes - 1)

    async def record_success(
        self,
        provider_name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Record successful execution of an upstream request."""
        cfg = config or self._default_config
        lock = await self._get_lock(provider_name)

        async with lock:
            circuit = self._get_or_create_state(provider_name)
            now = time.monotonic()
            circuit.total_successes += 1

            if circuit.state == CircuitBreakerState.HALF_OPEN:
                circuit.in_flight_probes = max(0, circuit.in_flight_probes - 1)
                circuit.consecutive_successes += 1
                if circuit.consecutive_successes >= cfg.half_open_success_threshold:
                    # Probe passed threshold: transition HALF_OPEN -> CLOSED
                    self._transition_state(
                        provider_name=provider_name,
                        circuit=circuit,
                        new_state=CircuitBreakerState.CLOSED,
                        now=now,
                    )
                    circuit.consecutive_failures = 0
                    circuit.consecutive_successes = 0

            elif circuit.state == CircuitBreakerState.CLOSED:
                circuit.consecutive_failures = 0

    async def record_failure(
        self,
        provider_name: str,
        error: Exception | str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Record failed execution of an upstream request."""
        cfg = config or self._default_config
        lock = await self._get_lock(provider_name)

        async with lock:
            circuit = self._get_or_create_state(provider_name)
            now = time.monotonic()
            circuit.total_failures += 1
            circuit.consecutive_failures += 1
            circuit.last_failure_time = now

            # Sanitize last_error: remove tracebacks and raw payload leaks
            err_str = str(error)
            if "\n" in err_str:
                err_str = err_str.split("\n", maxsplit=1)[0]
            if len(err_str) > 200:
                err_str = err_str[:200] + "..."
            circuit.last_error = err_str

            if circuit.state == CircuitBreakerState.HALF_OPEN:
                # Any failure during HALF_OPEN immediately trips back to OPEN
                circuit.in_flight_probes = max(0, circuit.in_flight_probes - 1)
                self._transition_state(
                    provider_name=provider_name,
                    circuit=circuit,
                    new_state=CircuitBreakerState.OPEN,
                    now=now,
                )

            elif circuit.state == CircuitBreakerState.CLOSED:
                if circuit.consecutive_failures >= cfg.failure_threshold:
                    # Trip CLOSED -> OPEN
                    self._transition_state(
                        provider_name=provider_name,
                        circuit=circuit,
                        new_state=CircuitBreakerState.OPEN,
                        now=now,
                    )

    def _transition_state(
        self,
        provider_name: str,
        circuit: _ProviderCircuitState,
        new_state: CircuitBreakerState,
        now: float,
    ) -> None:
        """Perform state transition and emit telemetry metrics."""
        old_state = circuit.state
        if old_state != new_state:
            circuit.state = new_state
            circuit.last_state_change = now
            if self._metrics is not None:
                self._metrics.record_circuit_breaker_transition(
                    provider=provider_name,
                    from_state=old_state.value,
                    to_state=new_state.value,
                )
                self._metrics.record_circuit_breaker_state(
                    provider=provider_name,
                    state=new_state.value,
                )

    async def get_state(self, provider_name: str) -> CircuitBreakerState:
        """Get the current state for a provider, evaluating lazy OPEN -> HALF_OPEN."""
        lock = await self._get_lock(provider_name)
        async with lock:
            circuit = self._get_or_create_state(provider_name)
            if circuit.state == CircuitBreakerState.OPEN:
                elapsed = time.monotonic() - circuit.last_state_change
                if elapsed >= self._default_config.recovery_timeout_seconds:
                    self._transition_state(
                        provider_name=provider_name,
                        circuit=circuit,
                        new_state=CircuitBreakerState.HALF_OPEN,
                        now=time.monotonic(),
                    )
            return circuit.state

    async def get_snapshot(self, provider_name: str) -> CircuitBreakerSnapshot:
        """Capture an immutable diagnostic snapshot of the provider's circuit breaker."""
        lock = await self._get_lock(provider_name)
        async with lock:
            circuit = self._get_or_create_state(provider_name)
            return CircuitBreakerSnapshot(
                provider_name=provider_name,
                state=circuit.state,
                failure_count=circuit.total_failures,
                success_count=circuit.total_successes,
                consecutive_failures=circuit.consecutive_failures,
                consecutive_successes=circuit.consecutive_successes,
                last_state_change=circuit.last_state_change,
                last_failure_time=circuit.last_failure_time,
                last_error=circuit.last_error,
            )

    async def reset(self, provider_name: str) -> None:
        """Explicitly reset a provider's circuit breaker to clean CLOSED state."""
        lock = await self._get_lock(provider_name)
        async with lock:
            circuit = self._get_or_create_state(provider_name)
            self._transition_state(
                provider_name=provider_name,
                circuit=circuit,
                new_state=CircuitBreakerState.CLOSED,
                now=time.monotonic(),
            )
            circuit.consecutive_failures = 0
            circuit.consecutive_successes = 0
            circuit.in_flight_probes = 0
