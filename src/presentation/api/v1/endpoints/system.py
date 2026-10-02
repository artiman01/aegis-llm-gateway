"""System health, readiness, liveness probes, and Prometheus metrics exposition."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import JSONResponse

from application.services.circuit_breaker_service import CircuitBreakerService
from domain.models.circuit_breaker import CircuitBreakerState
from infrastructure.observability.prometheus_metrics_adapter import (
    PrometheusMetricsAdapter,
)

router = APIRouter(tags=["System"])


def get_circuit_breaker_service(request: Request) -> CircuitBreakerService:
    """Dependency retrieving CircuitBreakerService from application state."""
    return request.app.state.circuit_breaker  # type: ignore[no-any-return]


def get_metrics_adapter(request: Request) -> PrometheusMetricsAdapter:
    """Dependency retrieving PrometheusMetricsAdapter from application state."""
    return request.app.state.metrics_adapter  # type: ignore[no-any-return]


def _sanitize_error_message(err: str | None) -> str | None:
    """Sanitize error messages to eliminate stack traces and raw payloads."""
    if not err:
        return None
    cleaned = err.split("\n")[0].strip()
    if "Traceback" in cleaned:
        cleaned = "Internal upstream provider error"
    if len(cleaned) > 150:
        cleaned = cleaned[:150] + "..."
    return cleaned


@router.get(
    "/health/live",
    status_code=status.HTTP_200_OK,
    summary="Kubernetes liveness probe",
    description="Always returns 200 alive when the gateway process is running.",
)
async def liveness_probe() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@router.get(
    "/health/ready",
    summary="Kubernetes readiness probe",
    description=(
        "Returns 200 OK if at least one provider is healthy or probing. "
        "Returns 503 Service Unavailable if all upstream providers are in OPEN state."
    ),
)
async def readiness_probe(
    request: Request,
    circuit_breaker: CircuitBreakerService = Depends(get_circuit_breaker_service),
) -> Response:
    """Kubernetes readiness probe evaluating circuit breaker health."""
    providers: list[str] = list(getattr(request.app.state, "registered_providers", []))

    if not providers:
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ready"})

    available: list[str] = []
    tripped: list[str] = []

    for p in providers:
        state = await circuit_breaker.get_state(p)
        if state == CircuitBreakerState.OPEN:
            tripped.append(p)
        else:
            available.append(p)

    if not available:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unavailable",
                "message": "All upstream LLM providers are currently unavailable (circuits OPEN).",
                "tripped_providers": tripped,
            },
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ready",
            "available_providers": available,
            "degraded_providers": tripped,
        },
    )


@router.get(
    "/health",
    summary="Diagnostic health and readiness overview",
    description=(
        "Sanitized diagnostic status report of gateway components and provider circuit states."
    ),
)
async def health_check(
    request: Request,
    circuit_breaker: CircuitBreakerService = Depends(get_circuit_breaker_service),
) -> Response:
    """Comprehensive diagnostic health check with sanitized provider error states."""
    providers: list[str] = list(getattr(request.app.state, "registered_providers", []))
    cb_snapshots: dict[str, dict[str, Any]] = {}
    any_available = False

    for p in providers:
        snapshot = await circuit_breaker.get_snapshot(p)
        snap_dict = snapshot.model_dump()
        raw_err = snap_dict.get("last_error")
        snap_dict["last_error"] = _sanitize_error_message(raw_err)
        snap_dict.pop("raw_response", None)

        cb_snapshots[p] = snap_dict
        if snapshot.state != CircuitBreakerState.OPEN:
            any_available = True

    is_healthy = any_available or not providers
    status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    body = {
        "status": "healthy" if is_healthy else "unhealthy",
        "timestamp": time.time(),
        "gateway": "AegisLLM",
        "demo_mode": getattr(request.app.state, "demo_mode", False),
        "providers": cb_snapshots,
    }
    return JSONResponse(status_code=status_code, content=body)


@router.get(
    "/metrics",
    summary="Prometheus metrics scrape target",
    description="Exposes Prometheus metrics for scraping.",
)
async def metrics_endpoint(
    metrics: PrometheusMetricsAdapter = Depends(get_metrics_adapter),
) -> Response:
    """Prometheus metrics exposition endpoint."""
    raw_data, content_type = metrics.export_metrics()
    return Response(content=raw_data, media_type=content_type)
