"""Use case orchestrating streaming chat completions with Point-of-No-Return fallback."""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from domain.exceptions import CircuitBreakerOpenError, NoAvailableProviderError
from domain.models.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    DeltaMessage,
    StreamChoice,
)

if TYPE_CHECKING:
    from application.services.circuit_breaker_service import CircuitBreakerService
    from application.services.hedged_dispatcher_service import HedgedDispatcherService
    from domain.models.provider import ProviderConfig, RoutingRule
    from domain.ports.metrics_port import MetricsPort
    from domain.ports.provider_port import LLMProviderPort
    from infrastructure.security.streaming_dfa_automaton import StreamingDFAAutomaton

logger = logging.getLogger(__name__)


class StreamChatCompletionUseCase:
    """Orchestrates resilient SSE streaming with dynamic failover before first chunk.

    Enforces the 'Point of No Return' pattern:
    - Pre-stream failure: transparent fallback to secondary providers.
    - Mid-stream failure: graceful degradation emitting an error chunk without crashing.
    """

    def __init__(
        self,
        providers: dict[str, LLMProviderPort],
        provider_configs: dict[str, ProviderConfig],
        routing_rules: dict[str, RoutingRule],
        circuit_breaker: CircuitBreakerService,
        *,
        hedged_dispatcher: HedgedDispatcherService | None = None,
        streaming_dfa: StreamingDFAAutomaton | None = None,
        metrics: MetricsPort | None = None,
        default_provider: str = "openai",
    ) -> None:
        self._providers = providers
        self._configs = provider_configs
        self._rules = routing_rules
        self._circuit_breaker = circuit_breaker
        self._hedged_dispatcher = hedged_dispatcher
        self._streaming_dfa = streaming_dfa
        self._metrics = metrics
        self._default_provider = default_provider

    def _sanitize_chunk(self, chunk: ChatCompletionChunk) -> ChatCompletionChunk:
        """Apply DFA token boundary masking to chunk delta content."""
        if self._streaming_dfa is None or not chunk.choices:
            return chunk
        choice = chunk.choices[0]
        if choice.delta.content:
            sanitized = self._streaming_dfa.process_chunk(choice.delta.content)
            new_delta = choice.delta.model_copy(update={"content": sanitized})
            new_choice = choice.model_copy(update={"delta": new_delta})
            return chunk.model_copy(update={"choices": [new_choice]})
        return chunk

    def _flush_dfa_chunk(self, chunk_id: str, model: str) -> ChatCompletionChunk | None:
        """Flush any remaining carry-over text from the DFA automaton."""
        if self._streaming_dfa is None:
            return None
        remaining = self._streaming_dfa.flush()
        if not remaining:
            return None
        return ChatCompletionChunk(
            id=chunk_id,
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content=remaining),
                )
            ],
        )

    async def execute(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream chunks from active provider with fallback and graceful degradation.

        Raises:
            NoAvailableProviderError: If all candidate providers fail prior to emitting
                the first chunk.
        """
        rule = self._rules.get(request.model)
        if rule is not None:
            candidates = [rule.primary_provider, *rule.fallback_providers]
        else:
            candidates = [self._default_provider]

        # Speculative hedged streaming when multiple healthy candidates are configured
        if self._hedged_dispatcher is not None and len(candidates) >= 2:
            p1_name = candidates[0]
            p2_name = candidates[1]
            p1 = self._providers.get(p1_name)
            p2 = self._providers.get(p2_name)
            cfg1 = self._configs.get(p1_name)
            cfg2 = self._configs.get(p2_name)

            if p1 is not None and p2 is not None and cfg1 is not None and cfg2 is not None:
                try:
                    await self._circuit_breaker.acquire_execution_permission(p1_name)
                    await self._circuit_breaker.acquire_execution_permission(p2_name)

                    async def _stream_p1() -> AsyncIterator[ChatCompletionChunk]:
                        assert cfg1 is not None
                        assert p1 is not None
                        up_m1 = cfg1.get_upstream_model(request.model)
                        raw1 = p1.stream(request.model_copy(update={"model": up_m1}), cfg1)
                        res1 = await raw1 if inspect.isawaitable(raw1) else raw1
                        async for c in res1:
                            yield c

                    async def _stream_p2() -> AsyncIterator[ChatCompletionChunk]:
                        assert cfg2 is not None
                        assert p2 is not None
                        up_m2 = cfg2.get_upstream_model(request.model)
                        raw2 = p2.stream(request.model_copy(update={"model": up_m2}), cfg2)
                        res2 = await raw2 if inspect.isawaitable(raw2) else raw2
                        async for c in res2:
                            yield c

                    prompt_text = request.extract_prompt_text()
                    prompt_tokens = max(1, len(prompt_text) // 4)
                    hedged_iter = self._hedged_dispatcher.execute_hedged_stream(
                        _stream_p1, _stream_p2, p1_name, p2_name, prompt_tokens=prompt_tokens
                    )

                    last_id = "chatcmpl-hedged"
                    winner_recorded = False
                    async for chunk, winner_name in hedged_iter:
                        last_id = chunk.id
                        if not winner_recorded:
                            await self._circuit_breaker.record_success(winner_name)
                            winner_recorded = True
                        if self._metrics is not None:
                            self._metrics.record_stream_chunk(
                                provider=winner_name, model=request.model
                            )
                        yield self._sanitize_chunk(chunk)

                    flush_chunk = self._flush_dfa_chunk(last_id, request.model)
                    if flush_chunk is not None:
                        yield flush_chunk
                    return
                except Exception as hedged_exc:
                    logger.debug(
                        "Hedged streaming bypassed (%s); proceeding to sequential", hedged_exc
                    )

        attempted: list[str] = []
        reasons: dict[str, str] = {}

        for i, provider_name in enumerate(candidates):
            provider = self._providers.get(provider_name)
            config = self._configs.get(provider_name)

            if provider is None or config is None:
                attempted.append(provider_name)
                reasons[provider_name] = f"Provider '{provider_name}' not configured"
                continue

            # Verify Circuit Breaker State (Fail-fast if OPEN)
            try:
                await self._circuit_breaker.acquire_execution_permission(provider_name)
            except CircuitBreakerOpenError as cb_err:
                attempted.append(provider_name)
                reasons[provider_name] = f"CircuitBreakerOpen: {cb_err.message}"
                logger.warning(
                    "Streaming: provider '%s' circuit breaker is OPEN, falling back.",
                    provider_name,
                )
                continue

            attempted.append(provider_name)

            # Map virtual model to upstream model
            upstream_model = config.get_upstream_model(request.model)
            upstream_request = request.model_copy(update={"model": upstream_model})

            start_time = time.monotonic()
            stream_iter: AsyncIterator[ChatCompletionChunk] | None = None
            first_chunk: ChatCompletionChunk | None = None

            # Attempt to connect and fetch the FIRST chunk (Pre-first-chunk boundary)
            try:
                raw_stream = provider.stream(upstream_request, config)
                resolved_stream = (
                    await raw_stream if inspect.isawaitable(raw_stream) else raw_stream
                )
                stream_iter = aiter(resolved_stream)
                first_chunk = await anext(stream_iter)
            except Exception as exc:
                duration = time.monotonic() - start_time
                error_msg = str(exc)
                reasons[provider_name] = error_msg
                logger.error(
                    "Provider '%s' failed before emitting first chunk: %s. Initiating fallback.",
                    provider_name,
                    error_msg,
                )

                # Record failure against Circuit Breaker
                await self._circuit_breaker.record_failure(provider_name, error=exc)

                if self._metrics is not None:
                    self._metrics.record_request_latency(
                        provider=provider_name,
                        model=request.model,
                        status="error",
                        duration_seconds=duration,
                    )
                    if i + 1 < len(candidates):
                        next_provider = candidates[i + 1]
                        self._metrics.record_fallback(
                            from_provider=provider_name,
                            to_provider=next_provider,
                            reason=error_msg,
                        )
                # Continue loop to next fallback provider
                continue

            # Point of no return crossed: first chunk emitted, failover no longer viable
            await self._circuit_breaker.record_success(provider_name)
            if self._metrics is not None:
                self._metrics.record_stream_chunk(provider=provider_name, model=request.model)

            # Emit the first chunk (sanitized)
            yield self._sanitize_chunk(first_chunk)

            # Stream remaining chunks with graceful degradation
            try:
                if stream_iter is not None:
                    async for chunk in stream_iter:
                        if self._metrics is not None:
                            self._metrics.record_stream_chunk(
                                provider=provider_name, model=request.model
                            )
                        yield self._sanitize_chunk(chunk)

                flush_chunk = self._flush_dfa_chunk(first_chunk.id, request.model)
                if flush_chunk is not None:
                    yield flush_chunk

                duration = time.monotonic() - start_time
                if self._metrics is not None:
                    self._metrics.record_request_latency(
                        provider=provider_name,
                        model=request.model,
                        status="success",
                        duration_seconds=duration,
                    )
                return

            except Exception as mid_stream_exc:
                duration = time.monotonic() - start_time
                logger.error(
                    "Stream interrupted mid-transmission from '%s': %s",
                    provider_name,
                    mid_stream_exc,
                )
                await self._circuit_breaker.record_failure(provider_name, error=mid_stream_exc)
                if self._metrics is not None:
                    self._metrics.record_request_latency(
                        provider=provider_name,
                        model=request.model,
                        status="interrupted",
                        duration_seconds=duration,
                    )

                # Graceful degradation: emit an error event chunk so client terminates cleanly
                error_chunk = ChatCompletionChunk(
                    id=f"err-{int(time.time())}",
                    model=request.model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(
                                content=(
                                    "\n\n[Gateway Warning: Upstream provider stream interrupted]"
                                )
                            ),
                            finish_reason="error",
                        )
                    ],
                )
                yield error_chunk
                return

        # If all candidates failed before the first chunk could be emitted
        raise NoAvailableProviderError(
            attempted_providers=attempted,
            reasons=reasons,
            model=request.model,
        )
