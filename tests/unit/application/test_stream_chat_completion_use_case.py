"""Unit tests for StreamChatCompletionUseCase and Point of No Return pattern."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from application.services.circuit_breaker_service import CircuitBreakerService
from application.use_cases.stream_chat_completion_use_case import (
    StreamChatCompletionUseCase,
)
from domain.exceptions import NoAvailableProviderError, ProviderUnavailableError
from domain.models.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DeltaMessage,
    Role,
    StreamChoice,
)
from domain.models.provider import ProviderConfig, ProviderType, RoutingRule


class MockStreamingProvider:
    """Mock provider with controllable stream failure modes."""

    def __init__(
        self,
        name: str,
        fail_before_first_chunk: bool = False,
        fail_mid_stream: bool = False,
        chunks_to_emit: int = 3,
    ) -> None:
        self._name = name
        self.fail_before_first_chunk = fail_before_first_chunk
        self.fail_mid_stream = fail_mid_stream
        self.chunks_to_emit = chunks_to_emit
        self.stream_calls: int = 0

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.MOCK

    @property
    def provider_name(self) -> str:
        return self._name

    def supports_model(self, model: str) -> bool:
        return True

    async def complete(
        self, request: ChatCompletionRequest, config: ProviderConfig
    ) -> ChatCompletionResponse:
        raise NotImplementedError

    async def stream(
        self, request: ChatCompletionRequest, config: ProviderConfig
    ) -> AsyncIterator[ChatCompletionChunk]:
        self.stream_calls += 1
        if self.fail_before_first_chunk:
            raise ProviderUnavailableError(
                "Stream connection failed immediately",
                provider_name=self._name,
            )

        for i in range(self.chunks_to_emit):
            if self.fail_mid_stream and i == 1:
                # Simulate mid-stream network interruption
                raise ConnectionResetError("Connection reset by peer mid-stream")
            yield ChatCompletionChunk(
                id=f"chunk-{self._name}-{i}",
                model=request.model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(content=f"token_{i}"),
                        finish_reason="stop" if i == self.chunks_to_emit - 1 else None,
                    )
                ],
            )

    async def health_check(self, config: ProviderConfig) -> bool:
        return True


@pytest.fixture
def provider_configs() -> dict[str, ProviderConfig]:
    return {
        "openai": ProviderConfig(
            name="openai",
            provider_type=ProviderType.OPENAI,
            base_url="https://api.openai.com/v1",
        ),
        "anthropic": ProviderConfig(
            name="anthropic",
            provider_type=ProviderType.ANTHROPIC,
            base_url="https://api.anthropic.com",
            model_mapping={"gpt-4o": "claude-3-5-sonnet-20241022"},
        ),
    }


@pytest.fixture
def routing_rules() -> dict[str, RoutingRule]:
    return {
        "gpt-4o": RoutingRule(
            virtual_model="gpt-4o",
            primary_provider="openai",
            fallback_providers=["anthropic"],
        )
    }


@pytest.mark.asyncio
async def test_stream_primary_success(
    provider_configs: dict[str, ProviderConfig],
    routing_rules: dict[str, RoutingRule],
) -> None:
    """Normal streaming emits all chunks from primary provider."""
    openai_mock = MockStreamingProvider(name="openai", chunks_to_emit=3)
    anthropic_mock = MockStreamingProvider(name="anthropic")
    cb_service = CircuitBreakerService()

    use_case = StreamChatCompletionUseCase(
        providers={"openai": openai_mock, "anthropic": anthropic_mock},
        provider_configs=provider_configs,
        routing_rules=routing_rules,
        circuit_breaker=cb_service,
    )

    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="Count to 3")],
        stream=True,
    )

    emitted_chunks: list[ChatCompletionChunk] = []
    async for chunk in use_case.execute(req):
        emitted_chunks.append(chunk)

    assert len(emitted_chunks) == 3
    assert emitted_chunks[0].id == "chunk-openai-0"
    assert emitted_chunks[2].choices[0].finish_reason == "stop"
    assert openai_mock.stream_calls == 1
    assert anthropic_mock.stream_calls == 0


@pytest.mark.asyncio
async def test_stream_fallback_before_first_chunk(
    provider_configs: dict[str, ProviderConfig],
    routing_rules: dict[str, RoutingRule],
) -> None:
    """When primary provider fails before first chunk, fallback seamlessly kicks in."""
    openai_mock = MockStreamingProvider(name="openai", fail_before_first_chunk=True)
    anthropic_mock = MockStreamingProvider(name="anthropic", chunks_to_emit=2)
    cb_service = CircuitBreakerService()

    use_case = StreamChatCompletionUseCase(
        providers={"openai": openai_mock, "anthropic": anthropic_mock},
        provider_configs=provider_configs,
        routing_rules=routing_rules,
        circuit_breaker=cb_service,
    )

    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="Hello")],
        stream=True,
    )

    emitted_chunks: list[ChatCompletionChunk] = []
    async for chunk in use_case.execute(req):
        emitted_chunks.append(chunk)

    # Chunks should be yielded by fallback provider
    assert len(emitted_chunks) == 2
    assert emitted_chunks[0].id == "chunk-anthropic-0"
    assert openai_mock.stream_calls == 1
    assert anthropic_mock.stream_calls == 1


@pytest.mark.asyncio
async def test_stream_point_of_no_return_mid_stream_failure(
    provider_configs: dict[str, ProviderConfig],
    routing_rules: dict[str, RoutingRule],
) -> None:
    """Mid-stream failure cannot switch providers; emits error chunk gracefully."""
    openai_mock = MockStreamingProvider(name="openai", fail_mid_stream=True, chunks_to_emit=3)
    anthropic_mock = MockStreamingProvider(name="anthropic")
    cb_service = CircuitBreakerService()

    use_case = StreamChatCompletionUseCase(
        providers={"openai": openai_mock, "anthropic": anthropic_mock},
        provider_configs=provider_configs,
        routing_rules=routing_rules,
        circuit_breaker=cb_service,
    )

    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="Hello")],
        stream=True,
    )

    emitted_chunks: list[ChatCompletionChunk] = []
    async for chunk in use_case.execute(req):
        emitted_chunks.append(chunk)

    # Chunk 0 was emitted, chunk 1 failed mid-stream, followed by graceful error chunk
    assert len(emitted_chunks) == 2
    assert emitted_chunks[0].id == "chunk-openai-0"

    # Last chunk is the graceful degradation error chunk
    error_chunk = emitted_chunks[1]
    assert error_chunk.choices[0].finish_reason == "error"
    assert "interrupted" in (error_chunk.choices[0].delta.content or "")

    # Secondary provider must NOT have been called mid-stream
    assert anthropic_mock.stream_calls == 0


@pytest.mark.asyncio
async def test_stream_all_providers_fail_before_first_chunk(
    provider_configs: dict[str, ProviderConfig],
    routing_rules: dict[str, RoutingRule],
) -> None:
    """If all providers fail before emitting first chunk, raises NoAvailableProviderError."""
    openai_mock = MockStreamingProvider(name="openai", fail_before_first_chunk=True)
    anthropic_mock = MockStreamingProvider(name="anthropic", fail_before_first_chunk=True)
    cb_service = CircuitBreakerService()

    use_case = StreamChatCompletionUseCase(
        providers={"openai": openai_mock, "anthropic": anthropic_mock},
        provider_configs=provider_configs,
        routing_rules=routing_rules,
        circuit_breaker=cb_service,
    )

    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="Hello")],
        stream=True,
    )

    with pytest.raises(NoAvailableProviderError) as exc_info:
        async for _ in use_case.execute(req):
            pass

    assert exc_info.value.model == "gpt-4o"
    assert "openai" in exc_info.value.attempted_providers
    assert "anthropic" in exc_info.value.attempted_providers
