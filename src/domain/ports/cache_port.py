"""Port definitions for multi-tier caching (L1 Exact Hash and L2 Semantic Similarity)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from domain.models.chat import ChatCompletionResponse


@runtime_checkable
class L1CachePort(Protocol):
    """Port for Level 1 exact hash match cache (In-memory or Redis)."""

    async def get(self, key: str) -> ChatCompletionResponse | None:
        """Retrieve cached completion response by deterministic hash key."""
        ...

    async def set(
        self,
        key: str,
        response: ChatCompletionResponse,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store completion response with optional time-to-live in seconds."""
        ...

    async def delete(self, key: str) -> bool:
        """Evict entry by key."""
        ...

    async def exists(self, key: str) -> bool:
        """Check key presence."""
        ...

    async def clear(self) -> None:
        """Flush the cache."""
        ...


@runtime_checkable
class L2SemanticCachePort(Protocol):
    """Port for Level 2 semantic vector similarity cache."""

    async def search(
        self,
        embedding: list[float],
        model: str,
        similarity_threshold: float,
        user: str | None = None,
    ) -> tuple[ChatCompletionResponse, float] | None:
        """Find most similar cached completion matching embedding above threshold.

        Returns tuple of (CachedResponse, CosineSimilarityScore) or None if no match.
        """
        ...

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
        """Index a prompt, embedding vector, and response into the semantic cache."""
        ...

    async def clear(self) -> None:
        """Flush all semantic cache entries."""
        ...


@runtime_checkable
class CachePort(L1CachePort, Protocol):
    """General cache port alias combining primary L1 cache operations."""

    ...
