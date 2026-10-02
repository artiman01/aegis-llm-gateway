"""Use case orchestrating non-streaming chat completions with fallback routing."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from domain.exceptions import CircuitBreakerOpenError, NoAvailableProviderError
from domain.models.chat import ChatCompletionRequest, ChatCompletionResponse

if TYPE_CHECKING:
    from application.services.circuit_breaker_service import CircuitBreakerService
    from application.services.hedged_dispatcher_service import HedgedDispatcherService
    from application.services.semantic_cache_service import SemanticCacheService
    from domain.models.provider import ProviderConfig, RoutingRule
    from domain.ports.metrics_port import MetricsPort
    from domain.ports.provider_port import LLMProviderPort

logger = logging.getLogger(__name__)


class RouteChatCompletionUseCase:
    """Orchestrates caching, circuit breaking, and dynamic fallback dispatching."""

    def __init__(
        self,
        providers: dict[str, LLMProviderPort],
        provider_configs: dict[str, ProviderConfig],
        routing_rules: dict[str, RoutingRule],
        circuit_breaker: CircuitBreakerService,
        *,
        cache_service: SemanticCacheService | None = None,
        hedged_dispatcher: HedgedDispatcherService | None = None,
        metrics: MetricsPort | None = None,
        default_provider: str = "openai",
    ) -> None:
        self._providers = providers
        self._configs = provider_configs
        self._rules = routing_rules
        self._circuit_breaker = circuit_breaker
        self._cache = cache_service
        self._hedged_dispatcher = hedged_dispatcher
        self._metrics = metrics
        self._default_provider = default_provider

    async def execute(
        self,
        request: ChatCompletionRequest,
        bypass_cache: bool = False,
    ) -> ChatCompletionResponse:
        """Execute chat completion with cache evaluation and resilient failover.

        Raises:
            NoAvailableProviderError: If primary and all fallback providers fail or are tripped.
        """
        # Two-tier cache lookup
        if self._cache is not None:
            cached_response = await self._cache.get(request, bypass_cache=bypass_cache)
            if cached_response is not None:
                return cached_response

        # Resolve candidate provider chain
        rule = self._rules.get(request.model)
        if rule is not None:
            candidates = [rule.primary_provider, *rule.fallback_providers]
        else:
            candidates = [self._default_provider]

        attempted: list[str] = []
        reasons: dict[str, str] = {}

        # Speculative hedged completion when multiple healthy candidates are configured
        if self._hedged_dispatcher is not None and len(candidates) >= 2:
            p1_name = candidates[0]
            p2_name = candidates[1]
            p1 = self._providers.get(p1_name)
            p2 = self._providers.get(p2_name)
            cfg1 = self._configs.get(p1_name)
            cfg2 = self._configs.get(p2_name)

            if p1 is not None and p2 is not None and cfg1 is not None and cfg2 is not None:
                try:
                    await self._circuit_breaker.acquire_execution_permission(p1_name)
                    await self._circuit_breaker.acquire_execution_permission(p2_name)

                    async def _call_p1() -> ChatCompletionResponse:
                        assert cfg1 is not None
                        assert p1 is not None
                        up_m1 = cfg1.get_upstream_model(request.model)
                        return await p1.complete(request.model_copy(update={"model": up_m1}), cfg1)

                    async def _call_p2() -> ChatCompletionResponse:
                        assert cfg2 is not None
                        assert p2 is not None
                        up_m2 = cfg2.get_upstream_model(request.model)
                        return await p2.complete(request.model_copy(update={"model": up_m2}), cfg2)

                    prompt_text = request.extract_prompt_text()
                    prompt_tokens = max(1, len(prompt_text) // 4)
                    response, winner_name = await self._hedged_dispatcher.execute_hedged_completion(
                        _call_p1, _call_p2, p1_name, p2_name, prompt_tokens=prompt_tokens
                    )
                    await self._circuit_breaker.record_success(winner_name)
                    if self._cache is not None:
                        await self._cache.set(request, response, bypass_cache=bypass_cache)
                    return response
                except Exception as hedged_exc:
                    logger.debug("Hedged dispatch bypassed: %s", hedged_exc)

        # Traverse fallback chain
        for i, provider_name in enumerate(candidates):
            provider = self._providers.get(provider_name)
            config = self._configs.get(provider_name)

            if provider is None or config is None:
                attempted.append(provider_name)
                reasons[provider_name] = f"Provider '{provider_name}' not configured"
                continue

            # Verify Circuit Breaker State (Fail-fast if OPEN)
            try:
                await self._circuit_breaker.acquire_execution_permission(provider_name)
            except CircuitBreakerOpenError as cb_err:
                attempted.append(provider_name)
                reasons[provider_name] = f"CircuitBreakerOpen: {cb_err.message}"
                logger.warning(
                    "Provider '%s' tripped circuit breaker; skipping to fallback.",
                    provider_name,
                )
                continue

            attempted.append(provider_name)

            # Map virtual model to provider upstream model
            upstream_model = config.get_upstream_model(request.model)
            upstream_request = request.model_copy(update={"model": upstream_model})

            start_time = time.monotonic()
            try:
                response = await provider.complete(upstream_request, config)
                duration = time.monotonic() - start_time

                # Success: record healthy state & telemetry
                await self._circuit_breaker.record_success(provider_name)

                if self._metrics is not None:
                    self._metrics.record_request_latency(
                        provider=provider_name,
                        model=request.model,
                        status="success",
                        duration_seconds=duration,
                    )
                    if response.usage is not None:
                        self._metrics.record_token_usage(
                            provider=provider_name,
                            model=request.model,
                            prompt_tokens=response.usage.prompt_tokens,
                            completion_tokens=response.usage.completion_tokens,
                        )

                # Store response in L1 and L2 caches
                if self._cache is not None:
                    await self._cache.set(request, response, bypass_cache=bypass_cache)

                return response

            except Exception as exc:
                duration = time.monotonic() - start_time
                error_msg = str(exc)
                reasons[provider_name] = error_msg
                logger.error("Provider '%s' failed: %s", provider_name, error_msg)

                # Record failure against Circuit Breaker
                await self._circuit_breaker.record_failure(provider_name, error=exc)

                if self._metrics is not None:
                    self._metrics.record_request_latency(
                        provider=provider_name,
                        model=request.model,
                        status="error",
                        duration_seconds=duration,
                    )
                    # Record fallback transition metric if next candidate is present
                    if i + 1 < len(candidates):
                        next_provider = candidates[i + 1]
                        self._metrics.record_fallback(
                            from_provider=provider_name,
                            to_provider=next_provider,
                            reason=error_msg,
                        )

        # All candidates exhausted
        raise NoAvailableProviderError(
            attempted_providers=attempted,
            reasons=reasons,
            model=request.model,
        )
