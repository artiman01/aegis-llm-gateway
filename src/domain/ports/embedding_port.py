"""Port definition for embedding generation required by semantic caching."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingPort(Protocol):
    """Port for converting text prompts into vector representations (FastEmbed ONNX)."""

    async def embed_text(self, text: str) -> list[float]:
        """Generate a normalized dense vector embedding for a single text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of text prompts."""
        ...

    @property
    def dimension(self) -> int:
        """Vector dimensionality of the embedding model."""
        ...
