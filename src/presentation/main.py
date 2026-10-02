"""FastAPI Application entrypoint with Lifespan management and DEMO_MODE support."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CollectorRegistry

from application.services.circuit_breaker_service import CircuitBreakerService
from application.services.hedged_dispatcher_service import HedgedDispatcherService
from application.services.saliency_guard_service import SaliencyGuardService
from application.services.semantic_cache_service import SemanticCacheService
from application.use_cases.route_chat_completion_use_case import (
    RouteChatCompletionUseCase,
)
from application.use_cases.stream_chat_completion_use_case import (
    StreamChatCompletionUseCase,
)
from domain.models.provider import ProviderConfig, ProviderType, RoutingRule
from domain.ports.provider_port import LLMProviderPort
from infrastructure.cache.fastembed_adapter import FastEmbedAdapter
from infrastructure.cache.memory_cache_adapter import (
    MemoryCacheAdapter,
    MemorySemanticCacheAdapter,
)
from infrastructure.cache.redis_cache_adapter import RedisCacheAdapter
from infrastructure.cache.whitening_transformer import WhiteningTransformer
from infrastructure.observability.prometheus_metrics_adapter import (
    PrometheusMetricsAdapter,
)
from infrastructure.providers.anthropic_adapter import AnthropicAdapter
from infrastructure.providers.mock_provider import MockProvider
from infrastructure.providers.openai_adapter import OpenAIAdapter
from infrastructure.security.streaming_dfa_automaton import StreamingDFAAutomaton
from presentation.api.v1.endpoints.chat import router as chat_router
from presentation.middlewares.error_handling_middleware import ErrorHandlingMiddleware
from presentation.middlewares.logging_middleware import LoggingMiddleware
from presentation.middlewares.security_headers_middleware import (
    SecurityHeadersMiddleware,
)
from presentation.middlewares.timing_middleware import TimingMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aegisllm.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context managing lifecycle, pools, and DEMO_MODE."""
    logger.info("Initializing AegisLLM Gateway components...")

    # Observability and circuit breaker
    metrics_registry = CollectorRegistry(auto_describe=True)
    metrics_adapter = PrometheusMetricsAdapter(registry=metrics_registry)
    circuit_breaker = CircuitBreakerService(metrics=metrics_adapter)

    # Caching infrastructure (L1 & L2)
    redis_url = os.getenv("REDIS_URL")
    redis_adapter: RedisCacheAdapter | None = None
    l1_cache: Any
    l2_cache: Any

    if redis_url:
        try:
            redis_client = aioredis.from_url(redis_url)
            redis_adapter = RedisCacheAdapter(redis_client=redis_client)
            l1_cache = redis_adapter
            logger.info("Connected to Redis L1 cache at %s", redis_url)
        except Exception as exc:
            logger.warning("Failed to connect to Redis (%s); falling back to MemoryCache", exc)
            l1_cache = MemoryCacheAdapter()
    else:
        l1_cache = MemoryCacheAdapter()
        logger.info("Using In-Memory L1 Cache.")

    l2_cache = MemorySemanticCacheAdapter()
    embedding_adapter = FastEmbedAdapter()
    whitening_transformer = WhiteningTransformer(dimension=384, epsilon=1e-5)
    saliency_guard = SaliencyGuardService(
        whitening=whitening_transformer,
        similarity_threshold=float(os.getenv("L2_SIMILARITY_THRESHOLD", "0.90")),
    )

    cache_service = SemanticCacheService(
        l1_cache=l1_cache,
        l2_cache=l2_cache,
        embedding=embedding_adapter,
        metrics=metrics_adapter,
        saliency_guard=saliency_guard,
        similarity_threshold=float(os.getenv("L2_SIMILARITY_THRESHOLD", "0.90")),
    )

    hedged_dispatcher = HedgedDispatcherService(
        metrics=metrics_adapter,
        max_hedging_prompt_tokens=int(os.getenv("MAX_HEDGING_PROMPT_TOKENS", "2000")),
    )
    streaming_dfa = StreamingDFAAutomaton()

    # Provider configuration and DEMO_MODE evaluation
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    force_demo = os.getenv("DEMO_MODE", "false").lower() in ("true", "1")

    demo_mode = force_demo or (not openai_key and not anthropic_key)

    providers: dict[str, LLMProviderPort] = {}
    provider_configs: dict[str, ProviderConfig] = {}
    routing_rules: dict[str, RoutingRule] = {}

    if demo_mode:
        logger.warning(
            "*** DEMO_MODE ACTIVE ***: No API keys configured. Using MockProvider as primary."
        )
        mock_adapter = MockProvider(
            name="mock",
            canned_response=(
                "Hello from AegisLLM Gateway (Demo Mode)! "
                "Resilience, caching, and fallback policies are fully active."
            ),
            ttft_delay_seconds=0.03,
            chunk_delay_seconds=0.01,
        )
        providers["mock"] = mock_adapter
        providers["openai"] = mock_adapter
        providers["anthropic"] = mock_adapter

        provider_configs["mock"] = ProviderConfig(
            name="mock",
            provider_type=ProviderType.MOCK,
            base_url="http://mock.internal",
        )
        provider_configs["openai"] = provider_configs["mock"]
        provider_configs["anthropic"] = provider_configs["mock"]

        routing_rules["gpt-4o"] = RoutingRule(
            virtual_model="gpt-4o",
            primary_provider="mock",
            fallback_providers=[],
        )
    else:
        openai_adapter = OpenAIAdapter(name="openai")
        anthropic_adapter = AnthropicAdapter(name="anthropic")
        mock_fallback = MockProvider(name="mock_fallback")

        providers["openai"] = openai_adapter
        providers["anthropic"] = anthropic_adapter
        providers["mock_fallback"] = mock_fallback

        provider_configs["openai"] = ProviderConfig(
            name="openai",
            provider_type=ProviderType.OPENAI,
            api_key=openai_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout_seconds=float(os.getenv("OPENAI_TIMEOUT", "30.0")),
        )
        provider_configs["anthropic"] = ProviderConfig(
            name="anthropic",
            provider_type=ProviderType.ANTHROPIC,
            api_key=anthropic_key,
            base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
            timeout_seconds=float(os.getenv("ANTHROPIC_TIMEOUT", "30.0")),
            model_mapping={"gpt-4o": "claude-3-5-sonnet-20241022"},
        )
        provider_configs["mock_fallback"] = ProviderConfig(
            name="mock_fallback",
            provider_type=ProviderType.MOCK,
            base_url="http://mock.internal",
        )

        routing_rules["gpt-4o"] = RoutingRule(
            virtual_model="gpt-4o",
            primary_provider="openai",
            fallback_providers=["anthropic", "mock_fallback"],
        )

    # Use case orchestration
    route_use_case = RouteChatCompletionUseCase(
        providers=providers,
        provider_configs=provider_configs,
        routing_rules=routing_rules,
        circuit_breaker=circuit_breaker,
        cache_service=cache_service,
        hedged_dispatcher=hedged_dispatcher,
        metrics=metrics_adapter,
        default_provider=next(iter(providers)),
    )

    stream_use_case = StreamChatCompletionUseCase(
        providers=providers,
        provider_configs=provider_configs,
        routing_rules=routing_rules,
        circuit_breaker=circuit_breaker,
        hedged_dispatcher=hedged_dispatcher,
        streaming_dfa=streaming_dfa,
        metrics=metrics_adapter,
        default_provider=next(iter(providers)),
    )

    # Application state registration
    app.state.metrics_adapter = metrics_adapter
    app.state.circuit_breaker = circuit_breaker
    app.state.cache_service = cache_service
    app.state.whitening_transformer = whitening_transformer
    app.state.saliency_guard = saliency_guard
    app.state.hedged_dispatcher = hedged_dispatcher
    app.state.streaming_dfa = streaming_dfa
    app.state.route_use_case = route_use_case
    app.state.stream_use_case = stream_use_case
    app.state.providers = providers
    app.state.registered_providers = list(providers.keys())
    app.state.available_models = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022", "mock"]
    app.state.demo_mode = demo_mode

    logger.info("AegisLLM Gateway started successfully. Serving requests.")
    yield

    # Shutdown logic
    logger.info("Shutting down AegisLLM Gateway...")
    for p in providers.values():
        if hasattr(p, "aclose"):
            await p.aclose()
    if redis_adapter is not None:
        await redis_adapter.aclose()
    logger.info("AegisLLM Gateway shutdown complete.")


def create_app() -> FastAPI:
    """Application factory configuring routes, middlewares, and OpenAPI documentation."""
    app = FastAPI(
        title="AegisLLM Gateway",
        description="""
# AegisLLM Gateway API

Production-grade, resilient, observable LLM Gateway built with Hexagonal Architecture.

## Key Features:
- **Zero Vendor Lock-in:** Drop-in OpenAI `/v1/chat/completions` compatibility across models.
- **Circuit Breaker Resilience:** 3-state FSM (Closed, Open, Half-Open) per provider.
- **Two-Tier Cache:** Sub-millisecond L1 (Exact SHA-256) + L2 (Semantic Cosine Similarity).
- **Server-Sent Events (SSE):** Point-of-No-Return streaming failover.
- **Full Observability:** Prometheus metrics (`/metrics`) and health telemetry (`/health`).
        """,
        version="0.1.0",
        lifespan=lifespan,
    )

    cors_allowed_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "*").strip()
    cors_origins = [o.strip() for o in cors_allowed_origins_env.split(",") if o.strip()]
    allow_credentials = cors_origins != ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
        allow_headers=["*"],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(LoggingMiddleware)

    # Register Routers
    app.include_router(chat_router)

    return app


app = create_app()
