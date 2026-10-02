"""High-performance in-memory cache adapters with TTL and orjson serialization."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass

import orjson

from domain.models.chat import ChatCompletionResponse
from domain.ports.cache_port import L1CachePort, L2SemanticCachePort


@dataclass
class _MemoryCacheEntry:
    payload: bytes
    expires_at: float | None = None

    def is_expired(self, now: float) -> bool:
        if self.expires_at is None:
            return False
        return now >= self.expires_at


class MemoryCacheAdapter(L1CachePort):
    """In-memory L1 exact cache using orjson serialization and TTL management."""

    def __init__(self, max_entries: int = 50000) -> None:
        self._max_entries = max_entries
        self._store: dict[str, _MemoryCacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> ChatCompletionResponse | None:
        """Retrieve and deserialize response if present and not expired."""
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None

            now = time.monotonic()
            if entry.is_expired(now):
                del self._store[key]
                return None

            data = orjson.loads(entry.payload)
            return ChatCompletionResponse.model_validate(data)

    async def set(
        self,
        key: str,
        response: ChatCompletionResponse,
        ttl_seconds: int | None = None,
    ) -> None:
        """Serialize via orjson and store with expiration."""
        payload = orjson.dumps(response.model_dump())
        now = time.monotonic()
        expires_at = now + ttl_seconds if ttl_seconds is not None else None

        async with self._lock:
            # Enforce max capacity
            if len(self._store) >= self._max_entries and key not in self._store:
                self._prune_expired(now)
                if len(self._store) >= self._max_entries:
                    # Pop oldest entry
                    first_key = next(iter(self._store))
                    del self._store[first_key]

            self._store[key] = _MemoryCacheEntry(payload=payload, expires_at=expires_at)

    def _prune_expired(self, now: float) -> None:
        """Evict all expired entries."""
        expired = [k for k, v in self._store.items() if v.is_expired(now)]
        for k in expired:
            del self._store[k]

    async def delete(self, key: str) -> bool:
        """Delete specific entry."""
        async with self._lock:
            return self._store.pop(key, None) is not None

    async def exists(self, key: str) -> bool:
        """Verify presence and validity."""
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            if entry.is_expired(time.monotonic()):
                del self._store[key]
                return False
            return True

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._store.clear()


@dataclass
class _SemanticEntry:
    prompt: str
    embedding: list[float]
    model: str
    payload: bytes
    expires_at: float | None = None

    def is_expired(self, now: float) -> bool:
        if self.expires_at is None:
            return False
        return now >= self.expires_at


class MemorySemanticCacheAdapter(L2SemanticCachePort):
    """In-memory L2 semantic vector similarity cache with cosine similarity search."""

    def __init__(self, max_entries: int = 10000) -> None:
        self._max_entries = max_entries
        self._entries: list[_SemanticEntry] = []
        self._lock = asyncio.Lock()

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Calculate cosine similarity between two float vectors."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def search(
        self,
        embedding: list[float],
        model: str,
        similarity_threshold: float,
    ) -> tuple[ChatCompletionResponse, float] | None:
        """Search candidate entries for highest cosine similarity above threshold."""
        now = time.monotonic()
        async with self._lock:
            best_match: _SemanticEntry | None = None
            best_score: float = -1.0

            active_entries: list[_SemanticEntry] = []
            for entry in self._entries:
                if entry.is_expired(now):
                    continue
                active_entries.append(entry)

                if entry.model != model:
                    continue

                sim = self._cosine_similarity(embedding, entry.embedding)
                if sim >= similarity_threshold and sim > best_score:
                    best_score = sim
                    best_match = entry

            self._entries = active_entries

            if best_match is not None:
                data = orjson.loads(best_match.payload)
                response = ChatCompletionResponse.model_validate(data)
                response._cached_prompt = best_match.prompt
                response._cached_embedding = best_match.embedding
                return response, best_score

            return None

    async def store(
        self,
        prompt: str,
        embedding: list[float],
        model: str,
        response: ChatCompletionResponse,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store prompt, embedding, and response in semantic index."""
        now = time.monotonic()
        expires_at = now + ttl_seconds if ttl_seconds is not None else None
        payload = orjson.dumps(response.model_dump())

        async with self._lock:
            if len(self._entries) >= self._max_entries:
                # Remove expired or oldest
                self._entries = [e for e in self._entries if not e.is_expired(now)]
                if len(self._entries) >= self._max_entries:
                    self._entries.pop(0)

            self._entries.append(
                _SemanticEntry(
                    prompt=prompt,
                    embedding=embedding,
                    model=model,
                    payload=payload,
                    expires_at=expires_at,
                )
            )

    async def clear(self) -> None:
        """Flush semantic cache."""
        async with self._lock:
            self._entries.clear()
