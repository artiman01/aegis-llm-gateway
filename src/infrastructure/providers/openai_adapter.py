"""OpenAI Infrastructure Adapter implementing LLMProviderPort via HTTPX HTTP/2 client."""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import AsyncIterator
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import orjson

from domain.exceptions import (
    ProviderAuthenticationError,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from domain.models.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from domain.models.provider import ProviderConfig, ProviderType
from domain.ports.provider_port import LLMProviderPort

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMProviderPort):
    """Adapter for communicating with standard OpenAI API endpoints."""

    def __init__(
        self,
        name: str = "openai",
        client: httpx.AsyncClient | None = None,
        max_keepalive_connections: int = 50,
        max_connections: int = 200,
    ) -> None:
        self._name = name
        self._custom_client = client is not None
        self._client = client or httpx.AsyncClient(
            http2=True,
            limits=httpx.Limits(
                max_keepalive_connections=max_keepalive_connections,
                max_connections=max_connections,
            ),
        )

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI

    @property
    def provider_name(self) -> str:
        return self._name

    def supports_model(self, model: str) -> bool:
        return True

    async def aclose(self) -> None:
        """Close internal HTTP client session if owned."""
        if not self._custom_client and not self._client.is_closed:
            await self._client.aclose()

    def _parse_retry_after(self, response: httpx.Response) -> float | None:
        """Parse Retry-After header as integer/float seconds or HTTP date."""
        header_val = response.headers.get("retry-after")
        if not header_val:
            return None
        try:
            return float(header_val)
        except ValueError:
            pass

        try:
            dt = parsedate_to_datetime(header_val)
            delta = dt.timestamp() - time.time()
            return max(0.0, delta)
        except Exception:
            return None

    def _handle_http_error(self, response: httpx.Response) -> None:
        """Map HTTP error responses to specific domain provider exceptions."""
        status = response.status_code
        try:
            raw_text = response.text
        except Exception:
            raw_text = "<unreadable response body>"

        if status in (401, 403):
            raise ProviderAuthenticationError(
                message=f"OpenAI authentication failed: HTTP {status}",
                provider_name=self._name,
                status_code=status,
                raw_response=raw_text,
            )

        if status == 429:
            retry_after = self._parse_retry_after(response)
            raise ProviderRateLimitError(
                message="OpenAI rate limit exceeded",
                provider_name=self._name,
                retry_after=retry_after,
                status_code=429,
                raw_response=raw_text,
            )

        if status == 400:
            raise ProviderBadRequestError(
                message="OpenAI rejected request payload (Bad Request)",
                provider_name=self._name,
                status_code=400,
                raw_response=raw_text,
            )

        if 500 <= status <= 599:
            raise ProviderUnavailableError(
                message=f"OpenAI upstream server error: HTTP {status}",
                provider_name=self._name,
                status_code=status,
                raw_response=raw_text,
            )

        raise ProviderError(
            message=f"OpenAI request failed: HTTP {status}",
            provider_name=self._name,
            status_code=status,
            raw_response=raw_text,
        )

    def _build_url_and_headers(self, config: ProviderConfig) -> tuple[str, dict[str, str]]:
        """Construct standard target endpoint and authorization headers."""
        base = config.base_url.rstrip("/")
        url = f"{base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        }
        return url, headers

    async def complete(
        self,
        request: ChatCompletionRequest,
        config: ProviderConfig,
    ) -> ChatCompletionResponse:
        """Execute non-streaming completion."""
        url, headers = self._build_url_and_headers(config)
        payload: dict[str, Any] = request.model_dump(exclude_none=True)
        payload["stream"] = False

        timeout = httpx.Timeout(
            timeout=config.timeout_seconds,
            connect=config.connect_timeout_seconds,
        )

        try:
            response = await self._client.post(
                url=url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                message=f"OpenAI connection/read timeout after {config.timeout_seconds}s",
                provider_name=self._name,
            ) from exc
        except httpx.NetworkError as exc:
            raise ProviderUnavailableError(
                message=f"OpenAI network connectivity error: {exc}",
                provider_name=self._name,
            ) from exc

        if response.is_error:
            self._handle_http_error(response)

        data = orjson.loads(response.content)
        return ChatCompletionResponse.model_validate(data)

    async def stream(
        self,
        request: ChatCompletionRequest,
        config: ProviderConfig,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Execute SSE streaming completion."""
        url, headers = self._build_url_and_headers(config)
        payload: dict[str, Any] = request.model_dump(exclude_none=True)
        payload["stream"] = True

        timeout = httpx.Timeout(
            timeout=config.timeout_seconds,
            connect=config.connect_timeout_seconds,
        )

        req = self._client.build_request(
            method="POST",
            url=url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )

        try:
            response = await self._client.send(req, stream=True)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                message=f"OpenAI streaming timeout: {exc}",
                provider_name=self._name,
            ) from exc
        except httpx.NetworkError as exc:
            raise ProviderUnavailableError(
                message=f"OpenAI streaming network error: {exc}",
                provider_name=self._name,
            ) from exc

        if response.is_error:
            with contextlib.suppress(Exception):
                await response.aread()
            self._handle_http_error(response)

        try:
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or line.startswith(":"):
                    continue

                if line == "data: [DONE]":
                    break

                if line.startswith("data: "):
                    json_str = line[6:].strip()
                    try:
                        chunk_dict = orjson.loads(json_str)
                        yield ChatCompletionChunk.model_validate(chunk_dict)
                    except Exception as err:
                        logger.warning("Failed to parse OpenAI SSE chunk: %s", err)
        finally:
            await response.aclose()

    async def health_check(self, config: ProviderConfig) -> bool:
        """Verify provider availability."""
        try:
            base = config.base_url.rstrip("/")
            url = f"{base}/models"
            headers = {"Authorization": f"Bearer {config.api_key}"}
            res = await self._client.get(url, headers=headers, timeout=5.0)
            return res.status_code == 200
        except Exception:
            return False
