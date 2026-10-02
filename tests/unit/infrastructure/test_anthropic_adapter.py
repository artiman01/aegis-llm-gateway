"""Unit tests for AnthropicAdapter message normalization and SSE streaming translation."""

from __future__ import annotations

import httpx
import pytest

from domain.exceptions import ProviderRateLimitError
from domain.models.chat import ChatCompletionRequest, ChatMessage, Role
from domain.models.provider import ProviderConfig, ProviderType
from infrastructure.providers.anthropic_adapter import AnthropicAdapter


@pytest.fixture
def provider_config() -> ProviderConfig:
    return ProviderConfig(
        name="anthropic",
        provider_type=ProviderType.ANTHROPIC,
        api_key="test-key-mock-anthropic-5432",
        base_url="https://api.anthropic.com/v1",
        timeout_seconds=10.0,
    )


def test_anthropic_message_normalization() -> None:
    """Validate system extraction, user alternation, and leading role enforcement."""
    adapter = AnthropicAdapter()

    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are Claude."),
        ChatMessage(role=Role.SYSTEM, content="Be concise."),
        ChatMessage(role=Role.USER, content="Hello 1"),
        ChatMessage(role=Role.USER, content="Hello 2"),
        ChatMessage(role=Role.ASSISTANT, content="Hi 1"),
        ChatMessage(role=Role.ASSISTANT, content="Hi 2"),
        ChatMessage(role=Role.USER, content="Final query"),
    ]

    system_prompt, normalized = adapter._normalize_messages(messages)

    # Verify system prompts are concatenated
    assert system_prompt == "You are Claude.\n\nBe concise."

    # Verify strict message alternation
    assert len(normalized) == 3
    assert normalized[0]["role"] == "user"
    assert normalized[0]["content"] == "Hello 1\n\nHello 2"
    assert normalized[1]["role"] == "assistant"
    assert normalized[1]["content"] == "Hi 1\n\nHi 2"
    assert normalized[2]["role"] == "user"
    assert normalized[2]["content"] == "Final query"


def test_anthropic_leading_non_user_message_enforcement() -> None:
    """If conversation starts with assistant, a leading user turn must be prepended."""
    adapter = AnthropicAdapter()
    messages = [
        ChatMessage(role=Role.ASSISTANT, content="I spoke first."),
        ChatMessage(role=Role.USER, content="Okay."),
    ]
    _, normalized = adapter._normalize_messages(messages)
    assert normalized[0]["role"] == "user"
    assert normalized[1]["role"] == "assistant"
    assert normalized[2]["role"] == "user"


@pytest.mark.asyncio
async def test_anthropic_complete_translation(provider_config: ProviderConfig) -> None:
    """Validate Anthropic JSON response mapping to ChatCompletionResponse."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "test-key-mock-anthropic-5432"
        assert request.headers["anthropic-version"] == "2023-06-01"

        data = {
            "id": "msg_anthropic_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello from Claude 3.5 Sonnet!"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 15, "output_tokens": 8},
        }
        return httpx.Response(200, json=data)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AnthropicAdapter(client=client)

    req = ChatCompletionRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[ChatMessage(role=Role.USER, content="Hello Claude")],
    )

    response = await adapter.complete(req, provider_config)
    assert response.id == "msg_anthropic_123"
    assert response.choices[0].message.content == "Hello from Claude 3.5 Sonnet!"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 15
    assert response.usage.completion_tokens == 8
    assert response.usage.total_tokens == 23


@pytest.mark.asyncio
async def test_anthropic_stream_translation(provider_config: ProviderConfig) -> None:
    """Validate Anthropic SSE event translation to OpenAI ChatCompletionChunk."""
    m_start = 'data: {"type":"message_start","message":{"id":"msg_stream_01","role":"assistant"}}'
    d_delta1 = 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}'
    d_delta2 = 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" world!"}}'
    m_delta = (
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        '"usage":{"output_tokens":4}}'
    )

    event_lines = [
        "event: message_start",
        m_start,
        "",
        "event: content_block_delta",
        d_delta1,
        "",
        "event: content_block_delta",
        d_delta2,
        "",
        "event: message_delta",
        m_delta,
        "",
        "event: message_stop",
        'data: {"type":"message_stop"}',
        "",
    ]
    sse_text = "\n".join(event_lines)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse_text)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AnthropicAdapter(client=client)

    req = ChatCompletionRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[ChatMessage(role=Role.USER, content="Hello")],
        stream=True,
    )

    chunks = []
    async for chunk in adapter.stream(req, provider_config):
        chunks.append(chunk)

    # Chunks: message_start -> content_delta 1 -> content_delta 2 -> message_delta
    assert len(chunks) == 4
    assert chunks[0].choices[0].delta.role == Role.ASSISTANT
    assert chunks[1].choices[0].delta.content == "Hello"
    assert chunks[2].choices[0].delta.content == " world!"
    assert chunks[3].choices[0].finish_reason == "stop"
    assert chunks[3].usage is not None
    assert chunks[3].usage.completion_tokens == 4


@pytest.mark.asyncio
async def test_anthropic_rate_limit_mapping(provider_config: ProviderConfig) -> None:
    """HTTP 429 must trigger ProviderRateLimitError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "20.0"}, text="Rate limit reached")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AnthropicAdapter(client=client)

    req = ChatCompletionRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[ChatMessage(role=Role.USER, content="Hello")],
    )

    with pytest.raises(ProviderRateLimitError) as exc_info:
        await adapter.complete(req, provider_config)
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 20.0
