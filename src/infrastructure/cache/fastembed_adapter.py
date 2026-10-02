"""FastEmbed ONNX infrastructure adapter implementing EmbeddingPort with non-blocking execution."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastembed import TextEmbedding

from domain.ports.embedding_port import EmbeddingPort

logger = logging.getLogger(__name__)


class FastEmbedAdapter(EmbeddingPort):
    """Local ONNX-based embedding extractor wrapping FastEmbed in non-blocking threads."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        cache_dir: str | None = None,
        threads: int | None = None,
        max_concurrent: int = 16,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._cache_dir = cache_dir
        self._threads = threads
        self._max_concurrent = max_concurrent
        self._model: TextEmbedding | None = None
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Internal concurrency limit semaphore."""
        return self._semaphore

    def _get_or_load_model(self) -> TextEmbedding:
        """Synchronously load FastEmbed ONNX weights on first invocation (lazy loading)."""
        if self._model is None:
            kwargs: dict[str, Any] = {"model_name": self._model_name}
            if self._cache_dir is not None:
                kwargs["cache_dir"] = self._cache_dir
            if self._threads is not None:
                kwargs["threads"] = self._threads

            logger.info("Loading FastEmbed model '%s' via ONNX Runtime...", self._model_name)
            self._model = TextEmbedding(**kwargs)
            logger.info("FastEmbed model '%s' loaded successfully.", self._model_name)

        return self._model

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """CPU-bound synchronous embedding generation run within worker threads."""
        model = self._get_or_load_model()
        # model.embed returns a generator yielding numpy ndarrays
        embeddings_iter = model.embed(texts)
        return [vec.tolist() for vec in embeddings_iter]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings asynchronously without blocking the asyncio event loop."""
        if not texts:
            return []

        # Threadpool exhaustion guard: fail-fast if concurrency limit reached
        if self._semaphore.locked():
            logger.warning(
                "FastEmbed exhaustion guard: %d concurrent operations active; failing open",
                self._max_concurrent,
            )
            raise RuntimeError(f"FastEmbed concurrency limit of {self._max_concurrent} exceeded")

        async with self._semaphore:
            # Ensure model initialization doesn't race on first call
            if self._model is None:
                async with self._lock:
                    if self._model is None:
                        await asyncio.to_thread(self._get_or_load_model)

            return await asyncio.to_thread(self._embed_sync, texts)

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for a single prompt."""
        result = await self.embed_batch([text])
        return result[0]
