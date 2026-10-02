"""Redis infrastructure adapter implementing L1CachePort with orjson serialization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import orjson

from domain.exceptions import CacheUnavailableError
from domain.models.chat import ChatCompletionResponse
from domain.ports.cache_port import L1CachePort

if TYPE_CHECKING:
    import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RedisCacheAdapter(L1CachePort):
    """Distributed L1 exact cache powered by Redis with orjson serialization."""

    def __init__(
        self,
        redis_client: aioredis.Redis[bytes],
        key_prefix: str = "aegis:l1:",
        default_ttl_seconds: int = 3600,
    ) -> None:
        self._redis = redis_client
        self._prefix = key_prefix
        self._default_ttl = default_ttl_seconds

    def _format_key(self, key: str) -> str:
        if key.startswith(self._prefix):
            return key
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> ChatCompletionResponse | None:
        """Retrieve and deserialize response from Redis."""
        redis_key = self._format_key(key)
        try:
            raw_bytes: bytes | None = await self._redis.get(redis_key)
            if raw_bytes is None:
                return None
            data = orjson.loads(raw_bytes)
            return ChatCompletionResponse.model_validate(data)
        except Exception as exc:
            logger.error("Redis get error for key '%s': %s", redis_key, exc)
            raise CacheUnavailableError(f"Redis get failed: {exc}") from exc

    async def set(
        self,
        key: str,
        response: ChatCompletionResponse,
        ttl_seconds: int | None = None,
    ) -> None:
        """Serialize and store response into Redis with expiration."""
        redis_key = self._format_key(key)
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        try:
            payload = orjson.dumps(response.model_dump())
            await self._redis.set(redis_key, payload, ex=ttl)
        except Exception as exc:
            logger.error("Redis set error for key '%s': %s", redis_key, exc)
            raise CacheUnavailableError(f"Redis set failed: {exc}") from exc

    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        redis_key = self._format_key(key)
        try:
            deleted_count: int = await self._redis.delete(redis_key)
            return deleted_count > 0
        except Exception as exc:
            logger.error("Redis delete error: %s", exc)
            raise CacheUnavailableError(f"Redis delete failed: {exc}") from exc

    async def exists(self, key: str) -> bool:
        """Check key presence in Redis."""
        redis_key = self._format_key(key)
        try:
            count: int = await self._redis.exists(redis_key)
            return count > 0
        except Exception as exc:
            logger.error("Redis exists error: %s", exc)
            raise CacheUnavailableError(f"Redis exists failed: {exc}") from exc

    async def clear(self) -> None:
        """Flush keys with configured prefix from Redis."""
        try:
            cursor = 0
            pattern = f"{self._prefix}*"
            while True:
                cursor, keys = await self._redis.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    await self._redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as exc:
            logger.error("Redis clear error: %s", exc)
            raise CacheUnavailableError(f"Redis clear failed: {exc}") from exc

    async def aclose(self) -> None:
        """Close Redis connection."""
        await self._redis.close()
