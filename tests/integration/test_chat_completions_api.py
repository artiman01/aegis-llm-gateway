"""Integration tests for AegisLLM FastAPI presentation endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import patch

import httpx
import pytest

from presentation.main import app


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """Provide initialized AsyncClient with app lifespan active and mock embedding inference."""

    async def mock_embed_batch(self, texts: list[str]) -> list[list[float]]:  # type: ignore[no-untyped-def]
        return [[0.05] * 384 for _ in texts]

    target = "infrastructure.cache.fastembed_adapter.FastEmbedAdapter.embed_batch"
    with patch(target, mock_embed_batch):
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                yield ac


@pytest.mark.asyncio
async def test_api_chat_completions_non_streaming(client: httpx.AsyncClient) -> None:
    """Test non-streaming chat completion with X-Response-Time header verification."""
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello AegisLLM"}],
        "temperature": 0.7,
        "stream": False,
    }

    response = await client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "X-Response-Time" in response.headers
    assert response.headers["X-Response-Time"].endswith("ms")
    assert response.headers.get("Server") == "AegisLLM"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"

    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "gpt-4o"
    assert len(data["choices"]) > 0
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0
    assert data["usage"]["total_tokens"] > 0


@pytest.mark.asyncio
async def test_api_chat_completions_streaming_sse(client: httpx.AsyncClient) -> None:
    """Test streaming completion delivering Server-Sent Events (SSE)."""
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Stream me a message"}],
        "stream": True,
    }

    response = await client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert response.headers.get("X-Accel-Buffering") == "no"

    lines = response.text.split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]

    # Must contain data chunks and terminate with [DONE]
    assert len(data_lines) >= 2
    assert data_lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_api_two_tier_cache_hit(client: httpx.AsyncClient) -> None:
    """Subsequent identical requests must yield an instant L1 cache hit."""
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Deterministic Cache Test Prompt"}],
        "temperature": 0.0,
        "stream": False,
    }

    # 1. First request primes cache
    res1 = await client.post("/v1/chat/completions", json=payload)
    assert res1.status_code == 200

    # 2. Second request hits L1 exact cache
    res2 = await client.post("/v1/chat/completions", json=payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2.get("system_fingerprint") == "cache:l1_exact"

    # 3. Request with Cache-Control: no-cache bypasses cache
    res3 = await client.post(
        "/v1/chat/completions", json=payload, headers={"Cache-Control": "no-cache"}
    )
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3.get("system_fingerprint") != "cache:l1_exact"


@pytest.mark.asyncio
async def test_api_health_endpoint(client: httpx.AsyncClient) -> None:
    """Health probe returns healthy status and provider snapshots."""
    response = await client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["gateway"] == "AegisLLM"
    assert "providers" in data


@pytest.mark.asyncio
async def test_api_metrics_endpoint(client: httpx.AsyncClient) -> None:
    """Metrics scrape target returns valid Prometheus format text."""
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "aegis_request_duration_seconds" in response.text


@pytest.mark.asyncio
async def test_api_models_endpoint(client: httpx.AsyncClient) -> None:
    """Models endpoint lists available virtual models."""
    response = await client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    assert "gpt-4o" in model_ids


@pytest.mark.asyncio
async def test_api_invalid_request_error_handling(client: httpx.AsyncClient) -> None:
    """Invalid requests return standard OpenAI error JSON payload."""
    payload = {
        "model": "gpt-4o",
        "messages": [],  # Valid schema requires model and messages
        "temperature": 5.0,  # Invalid: out of bounds [0.0, 2.0]
    }

    response = await client.post("/v1/chat/completions", json=payload)
    assert response.status_code in (400, 422)
    data = response.json()
    assert "error" in data or "detail" in data
