"""Unit tests for MockProvider."""

from __future__ import annotations

import pytest

from domain.exceptions import ProviderUnavailableError
from domain.models.chat import ChatCompletionRequest, ChatMessage, Role
from domain.models.provider import ProviderConfig, ProviderType
from infrastructure.providers.mock_provider import MockProvider


@pytest.fixture
def mock_config() -> ProviderConfig:
    return ProviderConfig(
        name="mock",
        provider_type=ProviderType.MOCK,
        base_url="http://mock.internal",
    )


@pytest.fixture
def sample_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="Hello test")],
    )


@pytest.mark.asyncio
async def test_mock_provider_complete(
    mock_config: ProviderConfig, sample_request: ChatCompletionRequest
) -> None:
    """MockProvider returns structured response with valid token usage."""
    provider = MockProvider(canned_response="Hello world response")
    response = await provider.complete(sample_request, mock_config)

    assert response.model == "gpt-4o"
    assert response.choices[0].message.content == "Hello world response"
    assert response.usage is not None
    assert response.usage.completion_tokens == 3
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_mock_provider_stream(
    mock_config: ProviderConfig, sample_request: ChatCompletionRequest
) -> None:
    """MockProvider streaming emits individual word chunks and finishes with stop."""
    provider = MockProvider(
        canned_response="Alpha Beta Gamma",
        ttft_delay_seconds=0.01,
        chunk_delay_seconds=0.005,
    )

    chunks = []
    async for chunk in provider.stream(sample_request, mock_config):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0].choices[0].delta.content == "Alpha"
    assert chunks[1].choices[0].delta.content == " Beta"
    assert chunks[2].choices[0].delta.content == " Gamma"
    assert chunks[2].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_mock_provider_failure_injection(
    mock_config: ProviderConfig, sample_request: ChatCompletionRequest
) -> None:
    """Configuring failure_rate=1.0 reliably triggers ProviderUnavailableError."""
    provider = MockProvider(failure_rate=1.0)

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.complete(sample_request, mock_config)
    assert exc_info.value.status_code == 503

    with pytest.raises(ProviderUnavailableError):
        async for _ in provider.stream(sample_request, mock_config):
            pass

    assert await provider.health_check(mock_config) is False
