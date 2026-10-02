"""Adaptive Speculative Hedged Dispatcher Service implementing 'The Tail at Scale'."""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from domain.models.chat import ChatCompletionChunk, ChatCompletionResponse
    from domain.ports.metrics_port import MetricsPort

logger = logging.getLogger(__name__)


class RollingQuantileTracker:
    """Sliding-window latency quantile tracker for dynamic hedging deadline calculation."""

    def __init__(self, capacity: int = 100, default_p90: float = 0.35) -> None:
        self._capacity = capacity
        self._default_p90 = default_p90
        self._history: collections.deque[float] = collections.deque(maxlen=capacity)

    def record(self, latency_seconds: float) -> None:
        """Record latency observation."""
        self._history.append(latency_seconds)

    def get_p90(self) -> float:
        """Calculate current 90th percentile latency."""
        if len(self._history) < 3:
            return self._default_p90
        sorted_samples = sorted(self._history)
        idx = int(0.90 * len(sorted_samples))
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]

    @property
    def sample_count(self) -> int:
        """Number of recorded latency samples in the window."""
        return len(self._history)


class HedgedDispatcherService:
    """Dispatches speculative hedged requests to combat long-tail latency spikes."""

    def __init__(
        self,
        metrics: MetricsPort | None = None,
        *,
        min_delay_seconds: float = 0.05,
        safety_margin_seconds: float = 0.05,
        default_p90_seconds: float = 0.35,
        window_size: int = 100,
        max_hedging_prompt_tokens: int = 2000,
    ) -> None:
        """Initialize HedgedDispatcherService.

        Args:
            metrics: Optional metrics port for telemetry.
            min_delay_seconds: Floor threshold for speculative hedging delay.
            safety_margin_seconds: Safety delta added to P90 latency before triggering fallback.
            default_p90_seconds: Cold-start baseline P90 prior to collecting sufficient samples.
            window_size: Sliding window capacity of latency observations per provider.
            max_hedging_prompt_tokens: Upper token limit beyond which hedging is disabled (FinOps).
        """
        self._metrics = metrics
        self._min_delay = min_delay_seconds
        self._safety_margin = safety_margin_seconds
        self._default_p90 = default_p90_seconds
        self._window_size = window_size
        self._max_hedging_prompt_tokens = max_hedging_prompt_tokens
        self._trackers: dict[str, RollingQuantileTracker] = {}
        self._lock = asyncio.Lock()

    @property
    def max_hedging_prompt_tokens(self) -> int:
        """Configured FinOps limit: prompt tokens above this threshold bypass hedging."""
        return self._max_hedging_prompt_tokens

    async def _get_tracker(self, provider_name: str) -> RollingQuantileTracker:
        """Retrieve or initialize the sliding-window tracker for a provider."""
        if provider_name not in self._trackers:
            async with self._lock:
                if provider_name not in self._trackers:
                    self._trackers[provider_name] = RollingQuantileTracker(
                        capacity=self._window_size,
                        default_p90=self._default_p90,
                    )
        return self._trackers[provider_name]

    async def record_latency(self, provider_name: str, latency_seconds: float) -> None:
        """Record an observed completion or TTFT latency for a provider."""
        tracker = await self._get_tracker(provider_name)
        tracker.record(latency_seconds)

    async def get_hedging_delay(self, provider_name: str) -> float:
        """Compute the dynamic hedging delay: max(min_delay, P90 + safety_margin)."""
        tracker = await self._get_tracker(provider_name)
        p90 = tracker.get_p90()
        return max(self._min_delay, p90 + self._safety_margin)

    async def execute_hedged_completion(
        self,
        primary_call: Callable[[], Awaitable[ChatCompletionResponse]],
        fallback_call: Callable[[], Awaitable[ChatCompletionResponse]] | None,
        primary_name: str,
        fallback_name: str | None = None,
        prompt_tokens: int = 0,
    ) -> tuple[ChatCompletionResponse, str]:
        """Execute non-streaming completion with speculative hedging against P90 tail latency.

        Args:
            primary_call: Async factory creating the primary provider request coroutine.
            fallback_call: Optional async factory for the speculative fallback request.
            primary_name: Name identifier of the primary provider.
            fallback_name: Name identifier of the fallback provider.
            prompt_tokens: Estimated or actual prompt tokens used for FinOps budget gating.

        Returns:
            Tuple of (winning_response, winning_provider_name).
        """
        start_time = time.monotonic()

        # FinOps budget guard: heavy context prompts bypass hedging to save costs
        if prompt_tokens > self._max_hedging_prompt_tokens:
            logger.info(
                "FinOps guard: prompt (%d tokens) > max_hedging_prompt_tokens (%d); "
                "bypassing speculative hedging for '%s'",
                prompt_tokens,
                self._max_hedging_prompt_tokens,
                primary_name,
            )
            res = await primary_call()
            elapsed = time.monotonic() - start_time
            await self.record_latency(primary_name, elapsed)
            return res, primary_name

        async def _invoke(
            call: Callable[[], Awaitable[ChatCompletionResponse]],
        ) -> ChatCompletionResponse:
            return await call()

        task_primary: asyncio.Task[ChatCompletionResponse] = asyncio.create_task(
            _invoke(primary_call)
        )

        if fallback_call is None or fallback_name is None:
            # No fallback available for hedging
            try:
                response = await task_primary
                duration = time.monotonic() - start_time
                await self.record_latency(primary_name, duration)
                return response, primary_name
            except Exception:
                raise

        hedging_delay = await self.get_hedging_delay(primary_name)

        # Wait for primary to complete within the dynamic P90 hedging deadline
        done, _ = await asyncio.wait({task_primary}, timeout=hedging_delay)

        if task_primary in done:
            try:
                response = task_primary.result()
                duration = time.monotonic() - start_time
                await self.record_latency(primary_name, duration)
                return response, primary_name
            except Exception as exc:
                logger.warning(
                    "Primary provider '%s' failed before hedging delay (%s); falling back",
                    primary_name,
                    exc,
                )
                # Primary failed fast: proceed immediately to fallback
                fallback_start = time.monotonic()
                fb_resp = await fallback_call()
                await self.record_latency(fallback_name, time.monotonic() - fallback_start)
                return fb_resp, fallback_name

        # Hedging delay expired: primary is in the slow tail! Launch speculative fallback.
        logger.info(
            "Primary '%s' exceeded P90 deadline (%.2fms); launching speculative fallback '%s'",
            primary_name,
            hedging_delay * 1000.0,
            fallback_name,
        )
        task_fallback: asyncio.Task[ChatCompletionResponse] = asyncio.create_task(
            _invoke(fallback_call)
        )

        # Race the two tasks
        pending = {task_primary, task_fallback}
        while pending:
            done_set, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for completed_task in done_set:
                try:
                    res = completed_task.result()
                    winner_name = primary_name if completed_task is task_primary else fallback_name

                    # Cancel losing task immediately to free connections and stop compute
                    for loser in pending:
                        loser.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)

                    elapsed = time.monotonic() - start_time
                    await self.record_latency(winner_name, elapsed)
                    logger.info(
                        "Hedged execution won by '%s' in %.2fms", winner_name, elapsed * 1000.0
                    )
                    return res, winner_name
                except Exception as exc:
                    failed_name = primary_name if completed_task is task_primary else fallback_name
                    logger.warning("Hedged task for '%s' failed during race: %s", failed_name, exc)
                    # If other task is still running, loop awaits it; else exits.

        raise RuntimeError("Both primary and hedged fallback calls failed.")

    async def execute_hedged_stream(
        self,
        primary_stream_factory: Callable[[], AsyncIterator[ChatCompletionChunk]],
        fallback_stream_factory: Callable[[], AsyncIterator[ChatCompletionChunk]] | None,
        primary_name: str,
        fallback_name: str | None = None,
        prompt_tokens: int = 0,
    ) -> AsyncIterator[tuple[ChatCompletionChunk, str]]:
        """Stream chunks from winner of a speculative first-chunk race.

        Dispatches primary stream; if first chunk is not received within P90 hedging deadline,
        speculatively launches fallback stream. Whichever yields the first token wins; the
        loser is immediately cancelled (issuing HTTP/2 RST_STREAM).

        Args:
            primary_stream_factory: Factory creating the primary provider chunk stream.
            fallback_stream_factory: Optional factory creating the speculative fallback stream.
            primary_name: Identifier for primary provider.
            fallback_name: Identifier for fallback provider.
            prompt_tokens: Estimated or actual prompt tokens used for FinOps budget gating.

        Yields:
            Tuples of (chunk, provider_name).
        """
        start_time = time.monotonic()
        primary_stream = primary_stream_factory()

        # FinOps budget guard: heavy context prompts bypass hedging
        if (
            fallback_stream_factory is None
            or fallback_name is None
            or prompt_tokens > self._max_hedging_prompt_tokens
        ):
            if prompt_tokens > self._max_hedging_prompt_tokens:
                logger.info(
                    "FinOps guard: stream prompt (%d tokens) > limit (%d); hedging off for '%s'",
                    prompt_tokens,
                    self._max_hedging_prompt_tokens,
                    primary_name,
                )
            first_chunk_emitted = False
            async for chunk in primary_stream:
                if not first_chunk_emitted:
                    await self.record_latency(primary_name, time.monotonic() - start_time)
                    first_chunk_emitted = True
                yield chunk, primary_name
            return

        hedging_delay = await self.get_hedging_delay(primary_name)

        async def fetch_first_chunk(
            name: str,
            stream: AsyncIterator[ChatCompletionChunk],
        ) -> tuple[str, ChatCompletionChunk, AsyncIterator[ChatCompletionChunk]]:
            chunk = await anext(stream)
            return name, chunk, stream

        task_primary = asyncio.create_task(fetch_first_chunk(primary_name, primary_stream))

        # Await first chunk from primary until dynamic hedging deadline
        done, _ = await asyncio.wait({task_primary}, timeout=hedging_delay)

        winner_name: str
        first_chunk: ChatCompletionChunk
        active_stream: AsyncIterator[ChatCompletionChunk]

        if task_primary in done and not task_primary.cancelled():
            try:
                winner_name, first_chunk, active_stream = task_primary.result()
                await self.record_latency(winner_name, time.monotonic() - start_time)
            except Exception as exc:
                logger.warning(
                    "Primary stream '%s' failed before hedging deadline: %s", primary_name, exc
                )
                # Primary stream failed early: transition straight to fallback
                fallback_stream = fallback_stream_factory()
                winner_name = fallback_name
                active_stream = fallback_stream
                first_chunk = await anext(active_stream)
                await self.record_latency(winner_name, time.monotonic() - start_time)
        else:
            # Hedging deadline reached without first token: spawn speculative fallback
            logger.info(
                "Primary stream '%s' TTFT exceeded P90 deadline (%.2fms); hedging with '%s'",
                primary_name,
                hedging_delay * 1000.0,
                fallback_name,
            )
            fallback_stream = fallback_stream_factory()
            task_fallback = asyncio.create_task(fetch_first_chunk(fallback_name, fallback_stream))

            pending: set[asyncio.Task[Any]] = {task_primary, task_fallback}
            race_won = False

            while pending and not race_won:
                done_set, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done_set:
                    try:
                        winner_name, first_chunk, active_stream = task.result()
                        # Cancel loser task immediately (triggers HTTP/2 RST_STREAM)
                        for loser in pending:
                            loser.cancel()
                        if pending:
                            await asyncio.gather(*pending, return_exceptions=True)
                        race_won = True
                        ttft = time.monotonic() - start_time
                        await self.record_latency(winner_name, ttft)
                        logger.info(
                            "Streaming TTFT race won by '%s' in %.2fms", winner_name, ttft * 1000.0
                        )
                        break
                    except Exception as exc:
                        logger.warning("Stream task failed during hedging race: %s", exc)

            if not race_won:
                raise RuntimeError("All hedged streaming providers failed before first token.")

        # Yield winning first chunk
        yield first_chunk, winner_name

        # Stream all remaining chunks from the winning provider
        async for chunk in active_stream:
            yield chunk, winner_name
