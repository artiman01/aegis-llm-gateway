"""Unit tests for domain models (Chat, Provider, Circuit Breaker)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from domain.models.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    ContentPartText,
    DeltaMessage,
    FunctionCall,
    Role,
    StreamChoice,
    ToolCall,
    UsageInfo,
)
from domain.models.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerSnapshot,
    CircuitBreakerState,
)
from domain.models.provider import (
    GatewayConfig,
    ProviderConfig,
    ProviderType,
    RoutingRule,
)


class TestChatModels:
    """Tests for chat completion request and response domain models."""

    def test_create_valid_request(self) -> None:
        """Test minimal valid ChatCompletionRequest instantiation."""
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                ChatMessage(role=Role.SYSTEM, content="You are an expert AI architect."),
                ChatMessage(role=Role.USER, content="Explain Hexagonal Architecture."),
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        assert req.model == "gpt-4o"
        assert len(req.messages) == 2
        assert req.stream is False
        assert req.temperature == 0.7

    def test_request_temperature_validation(self) -> None:
        """Temperature must be between 0.0 and 2.0."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-4o",
                messages=[ChatMessage(role=Role.USER, content="Hello")],
                temperature=2.5,
            )

    def test_compute_cache_key_deterministic(self) -> None:
        """Cache keys for identical semantic content must match deterministically."""
        req1 = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=Role.USER, content="Hello World")],
            temperature=0.5,
        )
        req2 = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=Role.USER, content="Hello World")],
            temperature=0.5,
        )
        assert req1.compute_cache_key() == req2.compute_cache_key()
        assert req1.compute_cache_key().startswith("llm:exact:")

    def test_compute_cache_key_differs_on_params(self) -> None:
        """Cache keys must differ if temperature or model differs."""
        req1 = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
            temperature=0.5,
        )
        req2 = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
            temperature=0.8,
        )
        assert req1.compute_cache_key() != req2.compute_cache_key()

    def test_compute_cache_key_full_completeness_and_composite_namespace(self) -> None:
        """Cache keys must incorporate seed, penalties, tool_calls, and composite namespace."""
        req_base = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
            tenant_id="tenant-alpha",
            user="user-123",
        )
        key_base = req_base.compute_cache_key()
        assert "tenant:tenant-alpha:user:user-123" in key_base

        # Default namespace fallback
        req_default = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
        )
        assert "tenant:default:user:anonymous" in req_default.compute_cache_key()

        # Seed differs
        req_seed = req_base.model_copy(update={"seed": 42})
        assert req_seed.compute_cache_key() != key_base

        # Presence penalty differs
        req_pp = req_base.model_copy(update={"presence_penalty": 0.5})
        assert req_pp.compute_cache_key() != key_base

        # Frequency penalty differs
        req_fp = req_base.model_copy(update={"frequency_penalty": 0.5})
        assert req_fp.compute_cache_key() != key_base

        # Logit bias differs
        req_lb = req_base.model_copy(update={"logit_bias": {"123": -1.0}})
        assert req_lb.compute_cache_key() != key_base

        # Tool choice differs
        req_tc = req_base.model_copy(update={"tool_choice": "auto"})
        assert req_tc.compute_cache_key() != key_base

        # Tool calls in message history differ
        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="get_weather", arguments='{"loc": "Paris"}'),
        )
        req_with_tool_call = ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                ChatMessage(role=Role.USER, content="Hello"),
                ChatMessage(
                    role=Role.ASSISTANT,
                    content=None,
                    tool_calls=[tool_call],
                ),
            ],
            tenant_id="tenant-alpha",
            user="user-123",
        )
        assert req_with_tool_call.compute_cache_key() != key_base

    def test_extract_prompt_text_multimodal(self) -> None:
        """Extract prompt text converts multipart messages into concatenated text."""
        msg = ChatMessage(
            role=Role.USER,
            content=[
                ContentPartText(text="Look at this code"),
                ContentPartText(text="and review it"),
            ],
        )
        assert msg.get_text_content() == "Look at this code and review it"

        req = ChatCompletionRequest(model="gpt-4o", messages=[msg])
        assert req.extract_prompt_text() == "user: Look at this code and review it"

    def test_chat_completion_response_serialization(self) -> None:
        """Validate response serialization and field integrity."""
        resp = ChatCompletionResponse(
            id="chatcmpl-test12345",
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role=Role.ASSISTANT, content="Hello! How can I help?"),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        )
        assert resp.id == "chatcmpl-test12345"
        assert resp.choices[0].message.content == "Hello! How can I help?"
        assert resp.usage is not None
        assert resp.usage.total_tokens == 18

    def test_streaming_chunk_sse_event_format(self) -> None:
        """Ensure SSE serialization produces compliant 'data: {json}\n\n' format."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-chunk-01",
            model="gpt-4o",
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(role=Role.ASSISTANT, content="Hello"),
                    finish_reason=None,
                )
            ],
        )
        sse_line = chunk.to_sse_event()
        assert sse_line.startswith("data: ")
        assert sse_line.endswith("\n\n")

        json_str = sse_line.removeprefix("data: ").strip()
        parsed = json.loads(json_str)
        assert parsed["id"] == "chatcmpl-chunk-01"
        assert parsed["choices"][0]["delta"]["content"] == "Hello"


class TestProviderModels:
    """Tests for Provider and Routing configuration models."""

    def test_provider_config_model_mapping(self) -> None:
        """Provider mapping must translate virtual models to upstream models."""
        config = ProviderConfig(
            name="anthropic-main",
            provider_type=ProviderType.ANTHROPIC,
            base_url="https://api.anthropic.com",
            model_mapping={"gpt-4o": "claude-3-5-sonnet-20241022"},
        )
        assert config.get_upstream_model("gpt-4o") == "claude-3-5-sonnet-20241022"
        assert config.get_upstream_model("unknown-model") == "unknown-model"

    def test_gateway_config_composition(self) -> None:
        """Test full gateway composition model."""
        gateway = GatewayConfig(
            providers={
                "openai": ProviderConfig(
                    name="openai",
                    provider_type=ProviderType.OPENAI,
                    base_url="https://api.openai.com/v1",
                ),
                "anthropic": ProviderConfig(
                    name="anthropic",
                    provider_type=ProviderType.ANTHROPIC,
                    base_url="https://api.anthropic.com",
                ),
            },
            routes={
                "gpt-4o": RoutingRule(
                    virtual_model="gpt-4o",
                    primary_provider="openai",
                    fallback_providers=["anthropic"],
                )
            },
        )
        assert "openai" in gateway.providers
        assert gateway.routes["gpt-4o"].primary_provider == "openai"
        assert gateway.routes["gpt-4o"].fallback_providers == ["anthropic"]


class TestCircuitBreakerModels:
    """Tests for Circuit Breaker domain state models."""

    def test_circuit_breaker_snapshot(self) -> None:
        """Snapshot must record state and failure counters."""
        snapshot = CircuitBreakerSnapshot(
            provider_name="openai",
            state=CircuitBreakerState.CLOSED,
            failure_count=0,
            consecutive_failures=0,
        )
        assert snapshot.provider_name == "openai"
        assert snapshot.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_config_bounds(self) -> None:
        """Config must validate bounds on thresholds."""
        with pytest.raises(ValidationError):
            CircuitBreakerConfig(failure_threshold=0)
