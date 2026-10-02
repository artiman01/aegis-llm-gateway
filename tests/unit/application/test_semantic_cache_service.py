"""Unit tests for SemanticCacheService (L1 Exact + L2 Semantic)."""

from __future__ import annotations

import pytest

from application.services.semantic_cache_service import SemanticCacheService
from domain.models.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    Role,
)


class InMemoryL1CacheMock:
    """In-memory exact cache mock."""

    def __init__(self) -> None:
        self.store: dict[str, ChatCompletionResponse] = {}

    async def get(self, key: str) -> ChatCompletionResponse | None:
        resp = self.store.get(key)
        return resp.model_copy(deep=True) if resp else None

    async def set(
        self,
        key: str,
        response: ChatCompletionResponse,
        ttl_seconds: int | None = None,
    ) -> None:
        self.store[key] = response.model_copy(deep=True)

    async def delete(self, key: str) -> bool:
        return self.store.pop(key, None) is not None

    async def exists(self, key: str) -> bool:
        return key in self.store

    async def clear(self) -> None:
        self.store.clear()


class InMemoryL2CacheMock:
    """In-memory semantic cache mock."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, list[float], str, ChatCompletionResponse, str | None]] = []

    async def search(
        self,
        embedding: list[float],
        model: str,
        similarity_threshold: float,
        user: str | None = None,
    ) -> tuple[ChatCompletionResponse, float] | None:
        for _prompt, emb, m, resp, u in self.entries:
            if m != model:
                continue
            if u != user:
                continue
            # Simple dot product simulation
            sim = sum(a * b for a, b in zip(embedding, emb, strict=False))
            if sim >= similarity_threshold:
                return resp.model_copy(deep=True), sim
        return None

    async def store(
        self,
        prompt: str,
        embedding: list[float],
        model: str,
        response: ChatCompletionResponse,
        *,
        ttl_seconds: int | None = None,
        user: str | None = None,
    ) -> None:
        self.entries.append((prompt, embedding, model, response.model_copy(deep=True), user))

    async def clear(self) -> None:
        self.entries.clear()


class DummyEmbeddingPort:
    """Dummy embedding port generating deterministic vectors."""

    async def embed_text(self, text: str) -> list[float]:
        # Return simple normalized unit vector representation
        if "weather" in text.lower():
            return [0.0, 1.0, 0.0]
        return [1.0, 0.0, 0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_text(t) for t in texts]

    @property
    def dimension(self) -> int:
        return 3


@pytest.fixture
def sample_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="What is the weather in Paris?")],
        temperature=0.0,
    )


@pytest.fixture
def sample_response() -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="resp-123",
        model="gpt-4o",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role=Role.ASSISTANT,
                    content="The weather is sunny in Paris.",
                ),
                finish_reason="stop",
            )
        ],
    )


@pytest.mark.asyncio
async def test_l1_exact_cache_hit(
    sample_request: ChatCompletionRequest,
    sample_response: ChatCompletionResponse,
) -> None:
    """Exact request produces immediate L1 hit with provenance marker."""
    l1 = InMemoryL1CacheMock()
    cache_svc = SemanticCacheService(l1_cache=l1)

    # Initially empty
    assert await cache_svc.get(sample_request) is None

    # Store response
    await cache_svc.set(sample_request, sample_response)

    # Retrieval should hit L1
    cached = await cache_svc.get(sample_request)
    assert cached is not None
    assert cached.choices[0].message.content == sample_response.choices[0].message.content
    assert cached.system_fingerprint == "cache:l1_exact"


@pytest.mark.asyncio
async def test_l2_semantic_cache_hit_on_l1_miss(
    sample_response: ChatCompletionResponse,
) -> None:
    """Slightly different phrasing misses L1 but hits L2 via embedding similarity."""
    l1 = InMemoryL1CacheMock()
    l2 = InMemoryL2CacheMock()
    emb = DummyEmbeddingPort()
    cache_svc = SemanticCacheService(
        l1_cache=l1,
        l2_cache=l2,
        embedding=emb,
        similarity_threshold=0.90,
    )

    req1 = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="What is the weather in Paris?")],
    )
    await cache_svc.set(req1, sample_response)

    # req2 has different wording but produces same embedding vector in DummyEmbeddingPort
    req2 = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="Tell me Paris weather today")],
    )

    # L1 key for req2 will be different
    assert req1.compute_cache_key() != req2.compute_cache_key()

    # L2 semantic match should trigger
    cached = await cache_svc.get(req2)
    assert cached is not None
    assert cached.system_fingerprint is not None
    assert "cache:l2_semantic" in cached.system_fingerprint


@pytest.mark.asyncio
async def test_bypass_cache_flag(
    sample_request: ChatCompletionRequest,
    sample_response: ChatCompletionResponse,
) -> None:
    """bypass_cache=True skips both lookup and store."""
    l1 = InMemoryL1CacheMock()
    cache_svc = SemanticCacheService(l1_cache=l1)
    await cache_svc.set(sample_request, sample_response)

    # Normal lookup hits
    assert await cache_svc.get(sample_request) is not None

    # Bypassed lookup returns None
    assert await cache_svc.get(sample_request, bypass_cache=True) is None


@pytest.mark.asyncio
async def test_fail_open_on_cache_error(
    sample_request: ChatCompletionRequest,
) -> None:
    """If cache raises unexpected error, service fails open returning None."""

    class BrokenCache:
        async def get(self, key: str) -> None:
            raise ConnectionError("Redis cluster unreachable")

    broken_svc = SemanticCacheService(l1_cache=BrokenCache())  # type: ignore[arg-type]
    result = await broken_svc.get(sample_request)
    assert result is None  # Does not raise, safely fails open


@pytest.mark.asyncio
async def test_tenant_user_isolation_l1_and_l2(
    sample_response: ChatCompletionResponse,
) -> None:
    """Verify strict tenant/user isolation in both L1 and L2 caches."""
    l1 = InMemoryL1CacheMock()
    l2 = InMemoryL2CacheMock()
    cache_svc = SemanticCacheService(
        l1_cache=l1,
        l2_cache=l2,
        embedding=DummyEmbeddingPort(),
    )

    req_alice = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="Private data query")],
        user="user_alice",
    )
    req_bob = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="Private data query")],
        user="user_bob",
    )
    req_public = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="Private data query")],
        user=None,
    )

    # 1. L1 exact cache keys must be isolated
    assert req_alice.compute_cache_key() != req_bob.compute_cache_key()
    assert req_alice.compute_cache_key() != req_public.compute_cache_key()
    assert "user_alice" in req_alice.compute_cache_key()
    assert "user_bob" in req_bob.compute_cache_key()

    # 2. Store response under Alice's user context
    await cache_svc.set(req_alice, sample_response)

    # 3. Alice hits cache
    hit_alice = await cache_svc.get(req_alice)
    assert hit_alice is not None

    # 4. Bob misses cache (no cross-user leakage)
    miss_bob = await cache_svc.get(req_bob)
    assert miss_bob is None

    # 5. Public query misses cache (no leakage to unauthenticated queries)
    miss_public = await cache_svc.get(req_public)
    assert miss_public is None
