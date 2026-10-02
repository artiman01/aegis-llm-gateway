"""Unit tests for RedisCacheAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock

import orjson
import pytest

from domain.exceptions import CacheUnavailableError
from domain.models.chat import (
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Role,
    UsageInfo,
)
from infrastructure.cache.redis_cache_adapter import RedisCacheAdapter


@pytest.fixture
def sample_response() -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="resp-redis-01",
        model="gpt-4o",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role=Role.ASSISTANT, content="Redis response"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=12, completion_tokens=4, total_tokens=16),
    )


@pytest.mark.asyncio
async def test_redis_cache_get_and_set(sample_response: ChatCompletionResponse) -> None:
    """Validate Redis get and set with JSON deserialization."""
    mock_redis = AsyncMock()
    adapter = RedisCacheAdapter(redis_client=mock_redis)

    # 1. Miss
    mock_redis.get.return_value = None
    assert await adapter.get("mykey") is None
    mock_redis.get.assert_awaited_with("aegis:l1:mykey")

    # 2. Set
    await adapter.set("mykey", sample_response, ttl_seconds=600)
    mock_redis.set.assert_awaited_once()
    args, kwargs = mock_redis.set.await_args
    assert args[0] == "aegis:l1:mykey"
    assert kwargs["ex"] == 600

    # 3. Hit
    mock_redis.get.return_value = orjson.dumps(sample_response.model_dump())
    retrieved = await adapter.get("mykey")
    assert retrieved is not None
    assert retrieved.id == sample_response.id
    assert retrieved.choices[0].message.content == "Redis response"


@pytest.mark.asyncio
async def test_redis_cache_delete_and_exists() -> None:
    """Validate Redis key existence check and deletion."""
    mock_redis = AsyncMock()
    adapter = RedisCacheAdapter(redis_client=mock_redis)

    mock_redis.exists.return_value = 1
    assert await adapter.exists("key1") is True

    mock_redis.delete.return_value = 1
    assert await adapter.delete("key1") is True


@pytest.mark.asyncio
async def test_redis_cache_error_handling() -> None:
    """Redis connection faults must raise CacheUnavailableError."""
    mock_redis = AsyncMock()
    mock_redis.get.side_effect = ConnectionError("Connection refused")
    mock_redis.set.side_effect = ConnectionError("Connection refused")

    adapter = RedisCacheAdapter(redis_client=mock_redis)

    with pytest.raises(CacheUnavailableError):
        await adapter.get("err_key")

    dummy_resp = ChatCompletionResponse(id="x", model="m", choices=[])
    with pytest.raises(CacheUnavailableError):
        await adapter.set("err_key", dummy_resp)
