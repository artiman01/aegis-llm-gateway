"""Domain port for Isotropic ZCA-Whitening vector transformation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class WhiteningPort(Protocol):
    """Port defining embedding whitening and isotropic projection operations."""

    @property
    def is_fitted(self) -> bool:
        """Indicate whether the whitening transformation matrix has been calibrated."""
        ...

    def fit(self, embeddings: list[list[float]]) -> None:
        """Calibrate mean and whitening projection matrix from sample embeddings.

        Args:
            embeddings: Collection of dense embedding vectors (dim d).
        """
        ...

    def whiten(self, vector: list[float]) -> list[float]:
        """Transform input vector into isotropic space with unit L2 norm.

        Args:
            vector: Raw embedding vector.

        Returns:
            Normalized whitened embedding vector on the unit hypersphere.
        """
        ...
