# AegisLLM Gateway

> **Production-grade, resilient, observable LLM Gateway built with Hexagonal Architecture.**

AegisLLM is an enterprise gateway designed to protect against downstream model outages, eliminate vendor lock-in, slash LLM inferencing costs via a two-tier caching strategy (exact SHA-256 + ONNX semantic embeddings), and guarantee 99.99% availability via state-machine circuit breakers.

---

## Architecture Overview (C4 Model)

### System Context Diagram (Level 1)

```mermaid
C4Context
    title System Context Diagram for AegisLLM Gateway

    Person(client, "Downstream Services", "Web apps, Agents, Microservices consuming OpenAI-compatible APIs")
    System(aegis, "AegisLLM Gateway", "Resilient Hexagonal LLM Proxy with circuit breaking, two-tier cache, and fallback routing")
    System_Ext(openai, "OpenAI API", "Upstream GPT-4o / GPT-4o-mini provider")
    System_Ext(anthropic, "Anthropic API", "Upstream Claude 3.5 Sonnet / Haiku provider")
    System_Ext(redis, "Redis Cluster", "Distributed L1 exact hash cache")
    System_Ext(prometheus, "Prometheus", "Metrics collection and alerting")

    Rel(client, aegis, "POST /v1/chat/completions", "JSON / SSE over HTTP/2")
    Rel(aegis, redis, "Get / Set Exact Hash", "RESP")
    Rel(aegis, openai, "Forward completions / streams", "HTTPS / HTTP/2")
    Rel(aegis, anthropic, "Fallback completions / streams", "HTTPS / HTTP/2")
    Rel(prometheus, aegis, "Scrape /metrics", "HTTP")
```

### Container Diagram (Level 2)

```mermaid
C4Container
    title Container Diagram for AegisLLM Gateway

    Container_Boundary(c1, "AegisLLM Core Service") {
        Component(api, "Presentation Layer", "FastAPI, SSE Handlers, Middlewares", "Exposes /v1/chat/completions")
        Component(usecase, "Application Layer", "RouteChatCompletionUseCase, StreamChatCompletionUseCase", "Coordinates routing, caching, and resiliency")
        Component(cb, "Circuit Breaker Service", "Closed, Open, Half-Open FSM", "Isolates upstream provider faults")
        Component(cache_svc, "Semantic Cache Service", "FastEmbed ONNX + Cosine Similarity", "Evaluates L2 semantic matches")
        Component(domain, "Domain Layer", "Protocols, Exceptions, Chat/Provider Models", "Pure business logic without external dependencies")
        Component(adapters, "Infrastructure Adapters", "OpenAIAdapter, AnthropicAdapter, RedisCacheAdapter, PrometheusAdapter", "Implements domain ports")
    }

    Rel(api, usecase, "Dispatches requests to")
    Rel(usecase, cb, "Consults provider availability")
    Rel(usecase, cache_svc, "Checks L1 / L2 cache")
    Rel(usecase, domain, "Uses models & ports")
    Rel(adapters, domain, "Implements ports")
```

---

## Core Capabilities

- **Hexagonal Architecture (Ports and Adapters):** Domain layer has zero external framework dependencies. Core entities and business rules are completely decoupled from FastAPI, HTTPX, and third-party SDKs.
- **Circuit Breaker Pattern:** Isolated 3-state finite state machine (`CLOSED`, `OPEN`, `HALF_OPEN`) per provider prevents cascading failures and provides instant failover.
- **Two-Tier Caching:**
  - **L1 Exact Cache:** Sub-millisecond SHA-256 hash match via In-Memory / Redis.
  - **L2 Semantic Cache:** Embedding generation via local ONNX runtime (`FastEmbed` `all-MiniLM-L6-v2`) and Cosine Similarity thresholding (> 0.90).
- **Streaming over SSE:** Native, non-blocking Server-Sent Events (`stream=True`) with automatic disconnect handling and resource cleanup.
- **Deep Observability:** Prometheus metrics (`p50`, `p95`, `p99` latency, cache hit ratios, token usage counters) and OpenTelemetry spans.

---

## Project Structure

```
├── docs/
│   └── adr/                   # Architecture Decision Records
│       └── 0001-hexagonal-architecture-and-resilience.md
├── src/
│   ├── domain/                # Pure interfaces (Protocols) & Pydantic models
│   │   ├── exceptions.py      # Domain-specific error hierarchy
│   │   ├── models/            # Chat, Provider, CircuitBreaker models
│   │   └── ports/             # LLMProviderPort, CachePort, MetricsPort
│   ├── application/           # Use cases & orchestration services
│   │   ├── use_cases/         # RouteChatCompletionUseCase, StreamChatCompletionUseCase
│   │   └── services/          # CircuitBreakerService, SemanticCacheService
│   ├── infrastructure/        # Outbound adapters (network, DB, metrics)
│   │   ├── providers/         # OpenAIAdapter, AnthropicAdapter, MockProvider
│   │   ├── cache/             # MemoryCacheAdapter, RedisCacheAdapter
│   │   └── observability/     # PrometheusMetricsAdapter
│   └── presentation/          # Inbound API (FastAPI)
│       ├── api/v1/endpoints/  # /chat/completions
│       └── middlewares/       # Logging, Timing, ErrorHandling
├── tests/
│   ├── unit/                  # Unit tests (Domain, CB, Cache)
│   ├── integration/           # Integration tests with provider mocks
│   └── load/                  # k6 load testing scripts
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml             # Ruff, Mypy (strict mode), Pytest
├── Makefile
└── README.md
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

```bash
# Clone and enter workspace
git clone <repo-url> aegisllm && cd aegisllm

# Setup virtual environment and install dev dependencies
make dev

# Run strict type checking
make typecheck

# Run unit tests
make test-unit

# Start the gateway
make run
```
