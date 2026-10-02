"""Domain models and state machine specifications for Circuit Breaker."""

from __future__ import annotations

import time
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CircuitBreakerState(StrEnum):
    """Finite state machine states for Circuit Breaker."""

    CLOSED = "closed"
    """Normal operations: requests flow through directly."""

    OPEN = "open"
    """Tripped: requests fail-fast immediately without touching upstream."""

    HALF_OPEN = "half_open"
    """Recovery trial: limited canary probe traffic allowed to verify health."""


class CircuitBreakerConfig(BaseModel):
    """Configuration parameters governing state transitions."""

    failure_threshold: int = Field(
        default=5,
        ge=1,
        description="Number of consecutive failures required to trip CLOSED -> OPEN",
    )
    recovery_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Duration in seconds the breaker remains OPEN before probing in HALF_OPEN",
    )
    half_open_success_threshold: int = Field(
        default=2,
        ge=1,
        description="Consecutive successful probes in HALF_OPEN required to reset to CLOSED",
    )
    half_open_max_trials: int = Field(
        default=3,
        ge=1,
        description="Maximum concurrent trial requests allowed during HALF_OPEN state",
    )

    model_config = ConfigDict(extra="ignore")


class CircuitBreakerSnapshot(BaseModel):
    """Point-in-time diagnostic snapshot of a provider's circuit breaker."""

    provider_name: str
    state: CircuitBreakerState
    failure_count: int = 0
    success_count: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_state_change: float = Field(default_factory=time.time)
    last_failure_time: float | None = None
    last_error: str | None = None

    model_config = ConfigDict(extra="ignore")
