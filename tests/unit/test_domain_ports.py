"""Unit tests for domain ports (typing.Protocol interfaces)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from domain.models.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from domain.models.provider import ProviderConfig, ProviderType
from domain.ports.cache_port import CachePort, L1CachePort, L2SemanticCachePort
from domain.ports.embedding_port import EmbeddingPort
from domain.ports.metrics_port import MetricsPort
from domain.ports.provider_port import LLMProviderPort


class DummyProvider:
    """Mock implementation conforming to LLMProviderPort."""

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.MOCK

    @property
    def provider_name(self) -> str:
        return "mock"

    def supports_model(self, model: str) -> bool:
        return True

    async def complete(
        self,
        request: ChatCompletionRequest,
        config: ProviderConfig,
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="test-id",
            model="mock",
            choices=[],
        )

    async def stream(
        self,
        request: ChatCompletionRequest,
        config: ProviderConfig,
    ) -> AsyncIterator[ChatCompletionChunk]:
        yield ChatCompletionChunk(id="test-chunk", model="mock", choices=[])

    async def health_check(self, config: ProviderConfig) -> bool:
        return True


class DummyL1Cache:
    """Mock implementation conforming to L1CachePort."""

    async def get(self, key: str) -> ChatCompletionResponse | None:
        return None

    async def set(
        self,
        key: str,
        response: ChatCompletionResponse,
        ttl_seconds: int | None = None,
    ) -> None:
        pass

    async def delete(self, key: str) -> bool:
        return True

    async def exists(self, key: str) -> bool:
        return False

    async def clear(self) -> None:
        pass


class DummyL2Cache:
    """Mock implementation conforming to L2SemanticCachePort."""

    async def search(
        self,
        embedding: list[float],
        model: str,
        similarity_threshold: float,
    ) -> tuple[ChatCompletionResponse, float] | None:
        return None

    async def store(
        self,
        prompt: str,
        embedding: list[float],
        model: str,
        response: ChatCompletionResponse,
        ttl_seconds: int | None = None,
    ) -> None:
        pass

    async def clear(self) -> None:
        pass


class DummyMetrics:
    """Mock implementation conforming to MetricsPort."""

    def record_request_latency(
        self,
        provider: str,
        model: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        pass

    def record_token_usage(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        pass

    def record_cache_access(self, tier: str, hit: bool) -> None:
        pass

    def record_circuit_breaker_state(self, provider: str, state: str) -> None:
        pass

    def record_circuit_breaker_transition(
        self,
        provider: str,
        from_state: str,
        to_state: str,
    ) -> None:
        pass

    def record_fallback(self, from_provider: str, to_provider: str, reason: str) -> None:
        pass

    def record_stream_chunk(self, provider: str, model: str) -> None:
        pass

    def set_active_requests(self, count: int) -> None:
        pass


class DummyEmbedding:
    """Mock implementation conforming to EmbeddingPort."""

    async def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3]]

    @property
    def dimension(self) -> int:
        return 3


def test_provider_port_runtime_check() -> None:
    """Ensure dummy provider satisfies LLMProviderPort."""
    provider = DummyProvider()
    assert isinstance(provider, LLMProviderPort)


def test_cache_ports_runtime_check() -> None:
    """Ensure cache implementations satisfy respective ports."""
    l1 = DummyL1Cache()
    assert isinstance(l1, L1CachePort)
    assert isinstance(l1, CachePort)

    l2 = DummyL2Cache()
    assert isinstance(l2, L2SemanticCachePort)


def test_metrics_port_runtime_check() -> None:
    """Ensure metrics adapter satisfies MetricsPort."""
    metrics = DummyMetrics()
    assert isinstance(metrics, MetricsPort)


def test_embedding_port_runtime_check() -> None:
    """Ensure embedding adapter satisfies EmbeddingPort."""
    embedding = DummyEmbedding()
    assert isinstance(embedding, EmbeddingPort)
