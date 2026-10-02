"""Unit tests for OpenAIAdapter using httpx MockTransport."""

from __future__ import annotations

import httpx
import orjson
import pytest

from domain.exceptions import (
    ProviderAuthenticationError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from domain.models.chat import ChatCompletionRequest, ChatMessage, Role
from domain.models.provider import ProviderConfig, ProviderType
from infrastructure.providers.openai_adapter import OpenAIAdapter


@pytest.fixture
def provider_config() -> ProviderConfig:
    return ProviderConfig(
        name="openai",
        provider_type=ProviderType.OPENAI,
        api_key="test-key-mock-openai-9876",
        base_url="https://api.openai.com/v1",
        timeout_seconds=5.0,
    )


@pytest.fixture
def sample_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="Hello OpenAI")],
    )


@pytest.mark.asyncio
async def test_openai_complete_success(
    provider_config: ProviderConfig, sample_request: ChatCompletionRequest
) -> None:
    """Validate successful JSON completion parsing."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key-mock-openai-9876"
        assert str(request.url) == "https://api.openai.com/v1/chat/completions"

        data = {
            "id": "chatcmpl-mock-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from mock OpenAI!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        }
        return httpx.Response(200, json=data)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAIAdapter(client=client)

    response = await adapter.complete(sample_request, provider_config)
    assert response.id == "chatcmpl-mock-123"
    assert response.choices[0].message.content == "Hello from mock OpenAI!"
    assert response.usage is not None
    assert response.usage.total_tokens == 11


@pytest.mark.asyncio
async def test_openai_stream_success(
    provider_config: ProviderConfig, sample_request: ChatCompletionRequest
) -> None:
    """Validate SSE streaming chunk parsing."""
    chunk1 = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Hello"},
                "finish_reason": None,
            }
        ],
    }
    chunk2 = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"content": " world!"}, "finish_reason": "stop"}],
    }

    sse_body = (
        f"data: {orjson.dumps(chunk1).decode()}\n\n"
        f"data: {orjson.dumps(chunk2).decode()}\n\n"
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse_body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAIAdapter(client=client)

    chunks = []
    async for chunk in adapter.stream(sample_request, provider_config):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "Hello"
    assert chunks[1].choices[0].delta.content == " world!"
    assert chunks[1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_authentication_error(
    provider_config: ProviderConfig, sample_request: ChatCompletionRequest
) -> None:
    """HTTP 401 must raise ProviderAuthenticationError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAIAdapter(client=client)

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        await adapter.complete(sample_request, provider_config)
    assert exc_info.value.status_code == 401
    assert exc_info.value.provider_name == "openai"


@pytest.mark.asyncio
async def test_openai_rate_limit_error_with_retry_after(
    provider_config: ProviderConfig, sample_request: ChatCompletionRequest
) -> None:
    """HTTP 429 must parse Retry-After header and raise ProviderRateLimitError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "15.5"}, json={"error": "rate limit"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAIAdapter(client=client)

    with pytest.raises(ProviderRateLimitError) as exc_info:
        await adapter.complete(sample_request, provider_config)
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 15.5


@pytest.mark.asyncio
async def test_openai_bad_request_error(
    provider_config: ProviderConfig, sample_request: ChatCompletionRequest
) -> None:
    """HTTP 400 must raise ProviderBadRequestError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Bad Request Payload")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAIAdapter(client=client)

    with pytest.raises(ProviderBadRequestError):
        await adapter.complete(sample_request, provider_config)


@pytest.mark.asyncio
async def test_openai_server_unavailable_error(
    provider_config: ProviderConfig, sample_request: ChatCompletionRequest
) -> None:
    """HTTP 503 must raise ProviderUnavailableError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAIAdapter(client=client)

    with pytest.raises(ProviderUnavailableError):
        await adapter.complete(sample_request, provider_config)


@pytest.mark.asyncio
async def test_openai_timeout_error(
    provider_config: ProviderConfig, sample_request: ChatCompletionRequest
) -> None:
    """Connection timeout must raise ProviderTimeoutError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Timed out reading from socket")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAIAdapter(client=client)

    with pytest.raises(ProviderTimeoutError):
        await adapter.complete(sample_request, provider_config)
