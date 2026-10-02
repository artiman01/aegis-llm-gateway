---
title: AegisLLM Gateway
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# AegisLLM Gateway

> **Production-grade, resilient, observable LLM Gateway built with Hexagonal Architecture.**

AegisLLM is an enterprise gateway designed to protect against downstream model outages, eliminate vendor lock-in, slash LLM inferencing costs via a two-tier caching strategy (exact SHA-256 + ONNX semantic embeddings), and guarantee 99.99% availability via state-machine circuit breakers.

---

## Architecture Overview (C4 Model)

### System Context Diagram (Level 1)

```mermaid
flowchart TD
    subgraph Clients ["Downstream Consumers"]
        Client["Web Apps / Autonomous Agents / Microservices<br/><i>(OpenAI-compatible clients)</i>"]
    end

    subgraph Aegis ["AegisLLM Resilience Perimeter"]
        Gateway["<b>AegisLLM Gateway</b><br/>High-Performance Proxy, Circuit Breaker & Semantic Router"]
    end

    subgraph Storage ["Storage & Telemetry"]
        Redis[("<b>Redis Cluster</b><br/>L1 Hash Cache & Sliding Limits")]
        Prometheus[("<b>Prometheus</b><br/>TTFT, P95 & FSM Metrics")]
    end

    subgraph Upstream ["Upstream Intelligence Providers"]
        OpenAI["<b>OpenAI API</b><br/>Primary Upstream (HTTP/2)"]
        Anthropic["<b>Anthropic API</b><br/>Hedged Speculative Fallback"]
    end

    Client ==>|"POST /v1/chat/completions<br/>(JSON / SSE Stream)"| Gateway
    Gateway <-->|"L1 Cache Lookup / Store"| Redis
    Gateway -.->|"/metrics scrape"| Prometheus
    Gateway -->|"Primary Speculative Stream"| OpenAI
    Gateway -.->|"Hedged Fallback (P90 Trigger)"| Anthropic

    style Gateway fill:#0284c7,stroke:#38bdf8,stroke-width:2px,color:#ffffff
    style Redis fill:#334155,stroke:#64748b,stroke-width:1px,color:#f8fafc
    style Prometheus fill:#334155,stroke:#64748b,stroke-width:1px,color:#f8fafc
    style OpenAI fill:#1e293b,stroke:#475569,stroke-width:1px,color:#f8fafc
    style Anthropic fill:#1e293b,stroke:#475569,stroke-width:1px,color:#f8fafc
    style Client fill:#0f172a,stroke:#334155,stroke-width:1px,color:#cbd5e1
```

### Container Diagram (Level 2 — Hexagonal Architecture)

```mermaid
flowchart LR
    subgraph Presentation ["Presentation Layer (FastAPI)"]
        direction TB
        API["<b>REST API Controller</b><br/>/v1/chat/completions"]
        MW["<b>Pipeline Middlewares</b><br/>Timing, Logging, Security, Error Handler"]
    end

    subgraph Core ["Application Core & Domain Layer"]
        direction TB
        UseCases["<b>Route & Stream Use Cases</b><br/>Execution & Failover Coordinator"]
        
        subgraph Services ["Domain Services"]
            Hedged["<b>Hedged Dispatcher</b><br/>P90 TTFT Speculative Race"]
            CB["<b>Circuit Breaker FSM</b><br/>Closed / Open / Half-Open"]
            Saliency["<b>Saliency Guard</b><br/>Temporal & Lexical Invariant Check"]
        end
    end

    subgraph Infrastructure ["Infrastructure Adapters"]
        direction TB
        subgraph NetAdapters ["Network Providers"]
            OpenAIAdapter["OpenAI HTTP/2 Adapter"]
            AnthropicAdapter["Anthropic Schema Adapter"]
        end
        subgraph CacheAdapters ["Cache & Math"]
            ZCA["ZCA Whitening & FastEmbed"]
            RedisAdapter["Redis L1 & L2 Store"]
            DFA["Streaming DFA Redactor"]
        end
    end

    API --> MW
    MW --> UseCases
    UseCases --> Services
    
    Hedged --> NetAdapters
    Hedged --> DFA
    Saliency --> CacheAdapters
    CB --> NetAdapters

    style Core fill:#0f172a,stroke:#0284c7,stroke-width:2px,color:#ffffff
    style Presentation fill:#1e293b,stroke:#475569,stroke-width:1px,color:#f8fafc
    style Infrastructure fill:#1e293b,stroke:#475569,stroke-width:1px,color:#f8fafc
    style Services fill:#1e293b,stroke:#334155,stroke-width:1px,color:#cbd5e1
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
