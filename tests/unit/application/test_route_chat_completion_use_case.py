"""Unit tests for RouteChatCompletionUseCase."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from application.services.circuit_breaker_service import CircuitBreakerService
from application.use_cases.route_chat_completion_use_case import (
    RouteChatCompletionUseCase,
)
from domain.exceptions import NoAvailableProviderError, ProviderUnavailableError
from domain.models.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    Role,
)
from domain.models.provider import ProviderConfig, ProviderType, RoutingRule


class MockProviderAdapter:
    """Configurable mock provider adapter."""

    def __init__(
        self,
        name: str,
        should_fail: bool = False,
        fail_exc: Exception | None = None,
    ) -> None:
        self._name = name
        self.should_fail = should_fail
        self.fail_exc = fail_exc or ProviderUnavailableError("Provider offline", provider_name=name)
        self.calls: list[ChatCompletionRequest] = []

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.MOCK

    @property
    def provider_name(self) -> str:
        return self._name

    def supports_model(self, model: str) -> bool:
        return True

    async def complete(
        self, request: ChatCompletionRequest, config: ProviderConfig
    ) -> ChatCompletionResponse:
        self.calls.append(request)
        if self.should_fail:
            raise self.fail_exc
        return ChatCompletionResponse(
            id=f"resp-{self._name}",
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role=Role.ASSISTANT,
                        content=f"Response from {self._name}",
                    ),
                    finish_reason="stop",
                )
            ],
        )

    async def stream(
        self, request: ChatCompletionRequest, config: ProviderConfig
    ) -> AsyncIterator[ChatCompletionChunk]:
        yield ChatCompletionChunk(id=f"chunk-{self._name}", model=request.model, choices=[])

    async def health_check(self, config: ProviderConfig) -> bool:
        return True


@pytest.fixture
def provider_configs() -> dict[str, ProviderConfig]:
    return {
        "openai": ProviderConfig(
            name="openai",
            provider_type=ProviderType.OPENAI,
            base_url="https://api.openai.com/v1",
        ),
        "anthropic": ProviderConfig(
            name="anthropic",
            provider_type=ProviderType.ANTHROPIC,
            base_url="https://api.anthropic.com",
            model_mapping={"gpt-4o": "claude-3-5-sonnet-20241022"},
        ),
    }


@pytest.fixture
def routing_rules() -> dict[str, RoutingRule]:
    return {
        "gpt-4o": RoutingRule(
            virtual_model="gpt-4o",
            primary_provider="openai",
            fallback_providers=["anthropic"],
        )
    }


@pytest.mark.asyncio
async def test_route_completion_primary_success(
    provider_configs: dict[str, ProviderConfig],
    routing_rules: dict[str, RoutingRule],
) -> None:
    """Normal execution calls primary provider directly."""
    openai_mock = MockProviderAdapter(name="openai")
    anthropic_mock = MockProviderAdapter(name="anthropic")
    cb_service = CircuitBreakerService()

    use_case = RouteChatCompletionUseCase(
        providers={"openai": openai_mock, "anthropic": anthropic_mock},
        provider_configs=provider_configs,
        routing_rules=routing_rules,
        circuit_breaker=cb_service,
    )

    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="Hello")],
    )

    resp = await use_case.execute(req)
    assert resp.id == "resp-openai"
    assert len(openai_mock.calls) == 1
    assert len(anthropic_mock.calls) == 0


@pytest.mark.asyncio
async def test_route_completion_fallback_on_primary_failure(
    provider_configs: dict[str, ProviderConfig],
    routing_rules: dict[str, RoutingRule],
) -> None:
    """When primary provider fails, gateway transparently falls back to secondary."""
    openai_mock = MockProviderAdapter(name="openai", should_fail=True)
    anthropic_mock = MockProviderAdapter(name="anthropic", should_fail=False)
    cb_service = CircuitBreakerService()

    use_case = RouteChatCompletionUseCase(
        providers={"openai": openai_mock, "anthropic": anthropic_mock},
        provider_configs=provider_configs,
        routing_rules=routing_rules,
        circuit_breaker=cb_service,
    )

    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="Hello")],
    )

    resp = await use_case.execute(req)
    assert resp.id == "resp-anthropic"
    assert len(openai_mock.calls) == 1
    assert len(anthropic_mock.calls) == 1
    # Verify model mapping on fallback (gpt-4o -> claude-3-5-sonnet-20241022)
    assert anthropic_mock.calls[0].model == "claude-3-5-sonnet-20241022"


@pytest.mark.asyncio
async def test_route_completion_skips_tripped_circuit_breaker(
    provider_configs: dict[str, ProviderConfig],
    routing_rules: dict[str, RoutingRule],
) -> None:
    """If primary provider's breaker is OPEN, request bypasses it without making a call."""
    openai_mock = MockProviderAdapter(name="openai")
    anthropic_mock = MockProviderAdapter(name="anthropic")
    cb_service = CircuitBreakerService()

    # Pre-trip openai breaker by recording failures
    for _ in range(5):
        await cb_service.record_failure("openai", "503 Outage")

    use_case = RouteChatCompletionUseCase(
        providers={"openai": openai_mock, "anthropic": anthropic_mock},
        provider_configs=provider_configs,
        routing_rules=routing_rules,
        circuit_breaker=cb_service,
    )

    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="Hello")],
    )

    resp = await use_case.execute(req)
    assert resp.id == "resp-anthropic"
    # OpenAI was not called because its breaker was OPEN
    assert len(openai_mock.calls) == 0
    assert len(anthropic_mock.calls) == 1


@pytest.mark.asyncio
async def test_route_completion_all_providers_exhausted(
    provider_configs: dict[str, ProviderConfig],
    routing_rules: dict[str, RoutingRule],
) -> None:
    """When all providers fail, NoAvailableProviderError is raised with diagnostic context."""
    openai_mock = MockProviderAdapter(name="openai", should_fail=True)
    anthropic_mock = MockProviderAdapter(name="anthropic", should_fail=True)
    cb_service = CircuitBreakerService()

    use_case = RouteChatCompletionUseCase(
        providers={"openai": openai_mock, "anthropic": anthropic_mock},
        provider_configs=provider_configs,
        routing_rules=routing_rules,
        circuit_breaker=cb_service,
    )

    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role=Role.USER, content="Hello")],
    )

    with pytest.raises(NoAvailableProviderError) as exc_info:
        await use_case.execute(req)

    assert exc_info.value.model == "gpt-4o"
    assert "openai" in exc_info.value.attempted_providers
    assert "anthropic" in exc_info.value.attempted_providers
