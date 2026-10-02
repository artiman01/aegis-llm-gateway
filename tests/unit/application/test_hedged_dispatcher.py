"""Unit tests for HedgedDispatcherService adaptive speculative hedging."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from application.services.hedged_dispatcher_service import HedgedDispatcherService
from domain.models.chat import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    Role,
    StreamChoice,
)


@pytest.fixture
def hedged_dispatcher() -> HedgedDispatcherService:
    """Create HedgedDispatcherService with fast test thresholds."""
    return HedgedDispatcherService(
        min_delay_seconds=0.03,
        safety_margin_seconds=0.01,
        default_p90_seconds=0.04,
        window_size=10,
    )


@pytest.mark.asyncio
async def test_hedged_dispatcher_primary_wins_fast(
    hedged_dispatcher: HedgedDispatcherService,
) -> None:
    """Primary completes within hedging delay; fallback is never executed."""
    fallback_called = False

    async def primary_call() -> ChatCompletionResponse:
        await asyncio.sleep(0.01)
        return ChatCompletionResponse(
            id="chatcmpl-primary",
            model="gpt-4o",
            choices=[
                Choice(index=0, message=ChoiceMessage(role=Role.ASSISTANT, content="Fast primary"))
            ],
        )

    async def fallback_call() -> ChatCompletionResponse:
        nonlocal fallback_called
        fallback_called = True
        return ChatCompletionResponse(
            id="chatcmpl-fallback",
            model="gpt-4o",
            choices=[
                Choice(index=0, message=ChoiceMessage(role=Role.ASSISTANT, content="Fallback"))
            ],
        )

    response, winner = await hedged_dispatcher.execute_hedged_completion(
        primary_call=primary_call,
        fallback_call=fallback_call,
        primary_name="openai",
        fallback_name="anthropic",
    )

    assert winner == "openai"
    assert response.id == "chatcmpl-primary"
    assert fallback_called is False


@pytest.mark.asyncio
async def test_hedged_dispatcher_fallback_wins_when_primary_stalls(
    hedged_dispatcher: HedgedDispatcherService,
) -> None:
    """Primary stalls (> P90); fallback launches speculatively and wins; primary is cancelled."""
    primary_cancelled = False

    async def stalling_primary() -> ChatCompletionResponse:
        nonlocal primary_cancelled
        try:
            # Simulate upstream GPU stall
            await asyncio.sleep(1.0)
            return ChatCompletionResponse(
                id="chatcmpl-slow",
                model="gpt-4o",
                choices=[
                    Choice(index=0, message=ChoiceMessage(role=Role.ASSISTANT, content="Slow"))
                ],
            )
        except asyncio.CancelledError:
            primary_cancelled = True
            raise

    async def responsive_fallback() -> ChatCompletionResponse:
        # Fallback responds quickly
        await asyncio.sleep(0.02)
        return ChatCompletionResponse(
            id="chatcmpl-hedged-win",
            model="gpt-4o",
            choices=[
                Choice(index=0, message=ChoiceMessage(role=Role.ASSISTANT, content="Hedged win"))
            ],
        )

    response, winner = await hedged_dispatcher.execute_hedged_completion(
        primary_call=stalling_primary,
        fallback_call=responsive_fallback,
        primary_name="openai",
        fallback_name="anthropic",
    )

    assert winner == "anthropic"
    assert response.id == "chatcmpl-hedged-win"
    # Verify primary task cancellation (HTTP/2 RST_STREAM trigger)
    assert primary_cancelled is True


@pytest.mark.asyncio
async def test_hedged_streaming_race_to_first_chunk(
    hedged_dispatcher: HedgedDispatcherService,
) -> None:
    """In streaming mode, fallback wins first token race when primary stalls.

    Primary stream task must be cancelled immediately.
    """
    primary_stream_cancelled = False

    def stalling_primary_stream() -> AsyncIterator[ChatCompletionChunk]:
        async def _gen() -> AsyncIterator[ChatCompletionChunk]:
            nonlocal primary_stream_cancelled
            try:
                await asyncio.sleep(1.0)
                yield ChatCompletionChunk(
                    id="chunk-slow",
                    model="gpt-4o",
                    choices=[StreamChoice(index=0, delta=DeltaMessage(content="Slow token"))],
                )
            except asyncio.CancelledError:
                primary_stream_cancelled = True
                raise

        return _gen()

    def responsive_fallback_stream() -> AsyncIterator[ChatCompletionChunk]:
        async def _gen() -> AsyncIterator[ChatCompletionChunk]:
            await asyncio.sleep(0.01)
            yield ChatCompletionChunk(
                id="chunk-fb-1",
                model="gpt-4o",
                choices=[StreamChoice(index=0, delta=DeltaMessage(content="Fast token 1"))],
            )
            await asyncio.sleep(0.01)
            yield ChatCompletionChunk(
                id="chunk-fb-2",
                model="gpt-4o",
                choices=[StreamChoice(index=0, delta=DeltaMessage(content="Fast token 2"))],
            )

        return _gen()

    chunks: list[ChatCompletionChunk] = []
    winners: list[str] = []

    async for chunk, winner in hedged_dispatcher.execute_hedged_stream(
        primary_stream_factory=stalling_primary_stream,
        fallback_stream_factory=responsive_fallback_stream,
        primary_name="openai",
        fallback_name="anthropic",
    ):
        chunks.append(chunk)
        winners.append(winner)

    assert len(chunks) == 2
    assert winners[0] == "anthropic"
    assert chunks[0].choices[0].delta.content == "Fast token 1"
    assert chunks[1].choices[0].delta.content == "Fast token 2"
    assert primary_stream_cancelled is True
