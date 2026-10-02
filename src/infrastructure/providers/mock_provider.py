"""Configurable Mock Provider for benchmarking, offline testing, and resilience simulation."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import AsyncIterator

from domain.exceptions import ProviderUnavailableError
from domain.models.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    Role,
    StreamChoice,
    UsageInfo,
)
from domain.models.provider import ProviderConfig, ProviderType
from domain.ports.provider_port import LLMProviderPort


class MockProvider(LLMProviderPort):
    """High-performance mock provider simulating real LLM network and latency dynamics."""

    def __init__(
        self,
        name: str = "mock",
        ttft_delay_seconds: float = 0.0,
        chunk_delay_seconds: float = 0.0,
        failure_rate: float = 0.0,
        canned_response: str = "This is a deterministic mock completion from AegisLLM.",
    ) -> None:
        self._name = name
        self.ttft_delay_seconds = ttft_delay_seconds
        self.chunk_delay_seconds = chunk_delay_seconds
        self.failure_rate = failure_rate
        self.canned_response = canned_response
        self.call_count: int = 0

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.MOCK

    @property
    def provider_name(self) -> str:
        return self._name

    def supports_model(self, model: str) -> bool:
        return True

    def _maybe_inject_failure(self) -> None:
        """Inject synthetic outage based on failure_rate probability."""
        if self.failure_rate > 0.0:
            sampled_ratio = secrets.randbelow(1_000_000) / 1_000_000.0
            if sampled_ratio < self.failure_rate:
                raise ProviderUnavailableError(
                    message=f"Synthetic mock failure triggered (rate={self.failure_rate:.2f})",
                    provider_name=self._name,
                    status_code=503,
                )

    async def complete(
        self,
        request: ChatCompletionRequest,
        config: ProviderConfig,
    ) -> ChatCompletionResponse:
        """Simulate non-streaming completion with latency and error injection."""
        self.call_count += 1
        self._maybe_inject_failure()

        if self.ttft_delay_seconds > 0.0:
            await asyncio.sleep(self.ttft_delay_seconds)

        words = self.canned_response.split()
        prompt_tokens = sum(len(m.get_text_content().split()) for m in request.messages)
        completion_tokens = len(words)

        return ChatCompletionResponse(
            id=f"mock-{int(time.time() * 1000)}-{self.call_count}",
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role=Role.ASSISTANT,
                        content=self.canned_response,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def stream(
        self,
        request: ChatCompletionRequest,
        config: ProviderConfig,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Simulate chunk-by-chunk streaming with Time-To-First-Token (TTFT) and pacing."""
        self.call_count += 1
        self._maybe_inject_failure()

        # Simulate initial TTFT latency
        if self.ttft_delay_seconds > 0.0:
            await asyncio.sleep(self.ttft_delay_seconds)

        words = self.canned_response.split()
        req_id = f"mock-chunk-{int(time.time() * 1000)}-{self.call_count}"

        for i, word in enumerate(words):
            if i > 0 and self.chunk_delay_seconds > 0.0:
                await asyncio.sleep(self.chunk_delay_seconds)

            is_last = i == len(words) - 1
            chunk_text = word if i == 0 else f" {word}"

            yield ChatCompletionChunk(
                id=req_id,
                model=request.model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(
                            role=Role.ASSISTANT if i == 0 else None,
                            content=chunk_text,
                        ),
                        finish_reason="stop" if is_last else None,
                    )
                ],
            )

    async def health_check(self, config: ProviderConfig) -> bool:
        """Health check returns True unless 100% failure rate configured."""
        return self.failure_rate < 1.0
