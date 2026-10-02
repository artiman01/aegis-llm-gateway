"""OpenAI-compatible /v1/chat/completions endpoint, /health, and /metrics."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Header, Request, status
from fastapi.responses import Response, StreamingResponse

from application.services.circuit_breaker_service import CircuitBreakerService
from application.use_cases.route_chat_completion_use_case import (
    RouteChatCompletionUseCase,
)
from application.use_cases.stream_chat_completion_use_case import (
    StreamChatCompletionUseCase,
)
from domain.models.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from infrastructure.observability.prometheus_metrics_adapter import (
    PrometheusMetricsAdapter,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat Completions"])


def get_route_use_case(request: Request) -> RouteChatCompletionUseCase:
    """Dependency retrieving RouteChatCompletionUseCase from application state."""
    return request.app.state.route_use_case  # type: ignore[no-any-return]


def get_stream_use_case(request: Request) -> StreamChatCompletionUseCase:
    """Dependency retrieving StreamChatCompletionUseCase from application state."""
    return request.app.state.stream_use_case  # type: ignore[no-any-return]


def get_metrics_adapter(request: Request) -> PrometheusMetricsAdapter:
    """Dependency retrieving PrometheusMetricsAdapter from application state."""
    return request.app.state.metrics_adapter  # type: ignore[no-any-return]


def get_circuit_breaker_service(request: Request) -> CircuitBreakerService:
    """Dependency retrieving CircuitBreakerService from application state."""
    return request.app.state.circuit_breaker  # type: ignore[no-any-return]


@router.post(
    "/v1/chat/completions",
    response_model=None,
    summary="Create chat completion",
    description=(
        "OpenAI-compatible chat completion endpoint supporting standard JSON and SSE streaming."
    ),
)
async def create_chat_completion(
    chat_request: ChatCompletionRequest,
    cache_control: str | None = Header(default=None),
    x_aegis_no_cache: str | None = Header(default=None),
    route_use_case: RouteChatCompletionUseCase = Depends(get_route_use_case),
    stream_use_case: StreamChatCompletionUseCase = Depends(get_stream_use_case),
) -> ChatCompletionResponse | StreamingResponse:
    """Dispatches chat completions to primary or fallback providers with optional SSE streaming."""
    bypass_cache = False
    if cache_control and "no-cache" in cache_control.lower():
        bypass_cache = True
    if x_aegis_no_cache and x_aegis_no_cache.lower() in ("true", "1"):
        bypass_cache = True

    if not chat_request.stream:
        return await route_use_case.execute(chat_request, bypass_cache=bypass_cache)

    async def sse_stream_generator() -> AsyncIterator[str]:
        async for chunk in stream_use_case.execute(chat_request):
            yield chunk.to_sse_event()
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/v1/models",
    summary="List available models",
    description="OpenAI-compatible models catalog listing virtual and upstream models.",
)
async def list_models(request: Request) -> dict[str, Any]:
    """Return available models list."""
    registered_models: list[str] = list(getattr(request.app.state, "available_models", ["gpt-4o"]))
    data = [
        {
            "id": m,
            "object": "model",
            "created": 1700000000,
            "owned_by": "aegisllm",
        }
        for m in registered_models
    ]
    return {"object": "list", "data": data}


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health and readiness probe",
    description="Returns gateway status, provider circuit breaker states, and uptime.",
)
async def health_check(
    circuit_breaker: CircuitBreakerService = Depends(get_circuit_breaker_service),
    request: Request = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """Liveness and readiness health check."""
    providers: list[str] = list(getattr(request.app.state, "registered_providers", []))
    cb_snapshots = {}
    for p in providers:
        snapshot = await circuit_breaker.get_snapshot(p)
        cb_snapshots[p] = snapshot.model_dump()

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "gateway": "AegisLLM",
        "demo_mode": getattr(request.app.state, "demo_mode", False),
        "providers": cb_snapshots,
    }


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
