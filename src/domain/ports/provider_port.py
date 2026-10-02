"""Port definition for upstream LLM providers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Coroutine
from typing import Any, Protocol, runtime_checkable

from domain.models.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from domain.models.provider import ProviderConfig, ProviderType

StreamResponse = (
    AsyncIterator[ChatCompletionChunk] | Coroutine[Any, Any, AsyncIterator[ChatCompletionChunk]]
)


@runtime_checkable
class LLMProviderPort(Protocol):
    """Port for communicating with an external LLM provider.

    Adheres strictly to the Hexagonal Architecture pattern where the domain
    defines the contract and infrastructure adapters implement it.
    """

    @property
    def provider_type(self) -> ProviderType:
        """Provider classification."""
        ...

    @property
    def provider_name(self) -> str:
        """Unique provider identifier."""
        ...

    def supports_model(self, model: str) -> bool:
        """Determine if provider can serve the designated model."""
        ...

    async def complete(
        self,
        request: ChatCompletionRequest,
        config: ProviderConfig,
    ) -> ChatCompletionResponse:
        """Execute a non-streaming chat completion."""
        ...

    def stream(
        self,
        request: ChatCompletionRequest,
        config: ProviderConfig,
    ) -> StreamResponse:
        """Execute a streaming chat completion delivering SSE chunks."""
        ...

    async def health_check(self, config: ProviderConfig) -> bool:
        """Verify upstream connectivity and operational readiness."""
        ...
