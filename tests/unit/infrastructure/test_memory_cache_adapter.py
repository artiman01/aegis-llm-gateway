"""Unit tests for MemoryCacheAdapter and MemorySemanticCacheAdapter."""

from __future__ import annotations

import asyncio

import pytest

from domain.models.chat import (
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Role,
    UsageInfo,
)
from infrastructure.cache.memory_cache_adapter import (
    MemoryCacheAdapter,
    MemorySemanticCacheAdapter,
)


@pytest.fixture
def sample_response() -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="resp-test-01",
        model="gpt-4o",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role=Role.ASSISTANT, content="Test content"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@pytest.mark.asyncio
async def test_memory_cache_crud_and_serialization(sample_response: ChatCompletionResponse) -> None:
    """Validate L1 cache store, retrieve, exists, and delete."""
    cache = MemoryCacheAdapter(max_entries=10)
    key = "test:exact:key1"

    assert await cache.get(key) is None
    assert await cache.exists(key) is False

    await cache.set(key, sample_response)
    assert await cache.exists(key) is True

    retrieved = await cache.get(key)
    assert retrieved is not None
    assert retrieved.id == sample_response.id
    assert retrieved.choices[0].message.content == "Test content"

    assert await cache.delete(key) is True
    assert await cache.exists(key) is False


@pytest.mark.asyncio
async def test_memory_cache_ttl_expiration(sample_response: ChatCompletionResponse) -> None:
    """Entries past TTL must expire and return None."""
    cache = MemoryCacheAdapter()
    key = "test:ttl:key"

    # Set with tiny 50ms TTL
    await cache.set(key, sample_response, ttl_seconds=0.05)  # type: ignore[arg-type]
    assert await cache.get(key) is not None

    await asyncio.sleep(0.06)
    assert await cache.get(key) is None
    assert await cache.exists(key) is False


@pytest.mark.asyncio
async def test_memory_cache_capacity_eviction(sample_response: ChatCompletionResponse) -> None:
    """Exceeding max capacity triggers oldest entry eviction."""
    cache = MemoryCacheAdapter(max_entries=2)

    await cache.set("k1", sample_response)
    await cache.set("k2", sample_response)
    # Inserting 3rd entry evicts "k1"
    await cache.set("k3", sample_response)

    assert await cache.get("k1") is None
    assert await cache.get("k2") is not None
    assert await cache.get("k3") is not None


@pytest.mark.asyncio
async def test_memory_semantic_cache_search(sample_response: ChatCompletionResponse) -> None:
    """Validate vector cosine similarity matching in MemorySemanticCacheAdapter."""
    semantic_cache = MemorySemanticCacheAdapter(max_entries=10)

    embedding_a = [1.0, 0.0, 0.0]
    embedding_similar = [0.99, 0.05, 0.0]
    embedding_orthogonal = [0.0, 1.0, 0.0]

    await semantic_cache.store(
        prompt="Tell me a joke",
        embedding=embedding_a,
        model="gpt-4o",
        response=sample_response,
    )

    # 1. Near identical query matches above 0.90 threshold
    hit = await semantic_cache.search(embedding_similar, model="gpt-4o", similarity_threshold=0.90)
    assert hit is not None
    resp, score = hit
    assert resp.id == sample_response.id
    assert score > 0.98

    # 2. Orthogonal query fails to match
    miss = await semantic_cache.search(
        embedding_orthogonal, model="gpt-4o", similarity_threshold=0.90
    )
    assert miss is None

    # 3. Model mismatch fails to match even with identical vector
    model_miss = await semantic_cache.search(
        embedding_a, model="claude-3-5", similarity_threshold=0.90
    )
    assert model_miss is None
