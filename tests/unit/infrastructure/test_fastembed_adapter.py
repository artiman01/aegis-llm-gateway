"""Unit tests for FastEmbedAdapter demonstrating non-blocking thread execution."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from infrastructure.cache.fastembed_adapter import FastEmbedAdapter


def test_fastembed_initial_state_is_lazy() -> None:
    """FastEmbedAdapter must not load heavy weights in __init__."""
    adapter = FastEmbedAdapter()
    assert adapter._model is None
    assert adapter.dimension == 384


@pytest.mark.asyncio
async def test_fastembed_embed_uses_asyncio_to_thread() -> None:
    """Verify inference runs off-loop via asyncio.to_thread and returns normalized vectors."""
    fake_vector = [0.05] * 384

    class FakeNumpyArray:
        def tolist(self) -> list[float]:
            return fake_vector

    fake_text_embedding_instance = MagicMock()
    fake_text_embedding_instance.embed.return_value = [FakeNumpyArray()]

    with patch(
        "infrastructure.cache.fastembed_adapter.TextEmbedding",
        return_value=fake_text_embedding_instance,
    ) as mock_cls:
        adapter = FastEmbedAdapter()

        # Thread check spy
        original_to_thread = asyncio.to_thread
        thread_calls = []

        async def spy_to_thread(func, *args, **kwargs):  # type: ignore[no-untyped-def]
            thread_calls.append(func)
            return await original_to_thread(func, *args, **kwargs)

        with patch("asyncio.to_thread", side_effect=spy_to_thread):
            vec = await adapter.embed_text("What is AegisLLM?")

        assert len(vec) == 384
        assert vec == fake_vector
        # Model class was instantiated lazily
        assert mock_cls.call_count == 1
        # Inference function was executed inside asyncio.to_thread
        assert adapter._embed_sync in thread_calls


@pytest.mark.asyncio
async def test_fastembed_embed_batch() -> None:
    """Validate batch embedding extraction."""
    fake_vectors = [[0.1] * 384, [0.2] * 384]

    class FakeNumpyArray:
        def __init__(self, vals: list[float]) -> None:
            self.vals = vals

        def tolist(self) -> list[float]:
            return self.vals

    fake_instance = MagicMock()
    fake_instance.embed.return_value = [FakeNumpyArray(v) for v in fake_vectors]

    with patch(
        "infrastructure.cache.fastembed_adapter.TextEmbedding",
        return_value=fake_instance,
    ):
        adapter = FastEmbedAdapter()
        results = await adapter.embed_batch(["Prompt 1", "Prompt 2"])

        assert len(results) == 2
        assert len(results[0]) == 384
        assert results[0][0] == 0.1
        assert results[1][0] == 0.2


@pytest.mark.asyncio
async def test_fastembed_threadpool_exhaustion_guard() -> None:
    """FastEmbedAdapter must raise RuntimeError when concurrency limit is saturated."""
    adapter = FastEmbedAdapter(max_concurrent=1)

    # Acquire the single available semaphore slot to simulate active threadpool computation
    await adapter.semaphore.acquire()

    with pytest.raises(RuntimeError, match="concurrency limit of 1 exceeded"):
        await adapter.embed_text("Overload prompt")

    adapter.semaphore.release()
