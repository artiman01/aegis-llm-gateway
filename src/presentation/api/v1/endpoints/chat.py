"""OpenAI-compatible /v1/chat/completions and /v1/models endpoints."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import Response, StreamingResponse

from application.use_cases.route_chat_completion_use_case import (
    RouteChatCompletionUseCase,
)
from application.use_cases.stream_chat_completion_use_case import (
    StreamChatCompletionUseCase,
)
from domain.models.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat Completions"])


def get_route_use_case(request: Request) -> RouteChatCompletionUseCase:
    """Dependency retrieving RouteChatCompletionUseCase from application state."""
    return request.app.state.route_use_case  # type: ignore[no-any-return]


def get_stream_use_case(request: Request) -> StreamChatCompletionUseCase:
    """Dependency retrieving StreamChatCompletionUseCase from application state."""
    return request.app.state.stream_use_case  # type: ignore[no-any-return]


@router.post(
    "/v1/chat/completions",
    response_model=None,
    summary="Create chat completion",
    description=(
        "OpenAI-compatible chat completion endpoint supporting standard JSON and SSE streaming."
    ),
)
async def create_chat_completion(
    request: Request,
    chat_request: ChatCompletionRequest,
    *,
    cache_control: str | None = Header(default=None),
    x_aegis_no_cache: str | None = Header(default=None),
    route_use_case: RouteChatCompletionUseCase = Depends(get_route_use_case),
    stream_use_case: StreamChatCompletionUseCase = Depends(get_stream_use_case),
) -> Response | StreamingResponse:
    """Dispatches chat completions to primary or fallback providers with optional SSE streaming."""
    bypass_cache = False
    if cache_control and "no-cache" in cache_control.lower():
        bypass_cache = True
    if x_aegis_no_cache and x_aegis_no_cache.lower() in ("true", "1"):
        bypass_cache = True

    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id and not chat_request.tenant_id:
        chat_request = chat_request.model_copy(update={"tenant_id": tenant_id})

    if not chat_request.stream:
        comp_response = await route_use_case.execute(chat_request, bypass_cache=bypass_cache)
        cache_status = "MISS"
        if bypass_cache:
            cache_status = "BYPASS"
        elif comp_response.system_fingerprint and "cache:" in comp_response.system_fingerprint:
            cache_type = comp_response.system_fingerprint.replace("cache:", "").upper()
            cache_status = f"HIT ({cache_type})"

        return Response(
            content=comp_response.model_dump_json(),
            media_type="application/json",
            headers={"X-Cache-Status": cache_status},
        )

    async def sse_stream_generator() -> AsyncIterator[str]:
        queue: asyncio.Queue[ChatCompletionChunk | None] = asyncio.Queue(maxsize=32)
        exception_holder: list[BaseException] = []

        async def _upstream_worker() -> None:
            try:
                async for chunk in stream_use_case.execute(chat_request):
                    await queue.put(chunk)
            except BaseException as exc:
                exception_holder.append(exc)
            finally:
                await queue.put(None)

        upstream_task = asyncio.create_task(_upstream_worker())

        try:
            while True:
                if await request.is_disconnected():
                    logger.info("Client disconnected during SSE stream; cancelling upstream task.")
                    upstream_task.cancel()
                    break

                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.1)
                except TimeoutError:
                    continue

                if item is None:
                    if exception_holder:
                        exc = exception_holder[0]
                        if not isinstance(exc, asyncio.CancelledError):
                            raise exc
                    break

                yield item.to_sse_event()

            if not await request.is_disconnected() and not exception_holder:
                yield "data: [DONE]\n\n"

        finally:
            if not upstream_task.done():
                upstream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await upstream_task

    return StreamingResponse(
        sse_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Cache-Status": "BYPASS (STREAM)",
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
