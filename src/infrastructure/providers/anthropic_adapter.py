"""Anthropic Infrastructure Adapter with bidirectional OpenAI <-> Anthropic translation."""

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
    ChatMessage,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    Role,
    StreamChoice,
    UsageInfo,
)
from domain.models.provider import ProviderConfig, ProviderType
from domain.ports.provider_port import LLMProviderPort

logger = logging.getLogger(__name__)


def map_anthropic_stop_reason(reason: str | None) -> str | None:
    """Map Anthropic stop_reason values to standard OpenAI finish_reason."""
    if not reason:
        return None
    mapping = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }
    return mapping.get(reason, reason)


class AnthropicAdapter(LLMProviderPort):
    """Adapter translating between OpenAI schema and Anthropic Messages API."""

    def __init__(
        self,
        name: str = "anthropic",
        client: httpx.AsyncClient | None = None,
        max_keepalive_connections: int = 50,
        max_connections: int = 200,
        anthropic_version: str = "2023-06-01",
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
        self._anthropic_version = anthropic_version

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC

    @property
    def provider_name(self) -> str:
        return self._name

    def supports_model(self, model: str) -> bool:
        return True

    async def aclose(self) -> None:
        """Close internal HTTP client session if owned."""
        if not self._custom_client and not self._client.is_closed:
            await self._client.aclose()

    def _normalize_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Extract system messages and enforce Anthropic's strict message alternation rules.

        Anthropic Requirements:
        1. 'system' prompt must be top-level string parameter, not inside messages.
        2. Strict user/assistant alternation: consecutive messages of same role must be merged.
        3. First conversation message must be from 'user'.
        """
        system_parts: list[str] = []
        raw_non_system: list[dict[str, str]] = []

        for msg in messages:
            text = msg.get_text_content()
            if msg.role == Role.SYSTEM:
                if text:
                    system_parts.append(text)
            else:
                role_str = "user" if msg.role == Role.USER else "assistant"
                raw_non_system.append({"role": role_str, "content": text})

        system_prompt = "\n\n".join(system_parts) if system_parts else None

        if not raw_non_system:
            # Anthropic requires at least one user message
            return system_prompt, [{"role": "user", "content": "Hello"}]

        # Ensure first message is 'user'
        normalized: list[dict[str, str]] = []
        if raw_non_system[0]["role"] != "user":
            normalized.append({"role": "user", "content": "Proceed."})

        # Merge consecutive identical roles
        for item in raw_non_system:
            if not normalized:
                normalized.append(item)
            elif normalized[-1]["role"] == item["role"]:
                normalized[-1]["content"] = f"{normalized[-1]['content']}\n\n{item['content']}"
            else:
                normalized.append(item)

        return system_prompt, normalized

    def _build_payload_and_headers(
        self, request: ChatCompletionRequest, config: ProviderConfig
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """Construct Anthropic Messages API payload and headers."""
        system_prompt, messages = self._normalize_messages(request.messages)

        # Anthropic requires max_tokens to be explicitly specified
        max_tokens = request.max_tokens or 4096

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        if system_prompt:
            payload["system"] = system_prompt
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p

        base = config.base_url.rstrip("/")
        url = f"{base}/messages" if not base.endswith("/messages") else base

        headers = {
            "Content-Type": "application/json",
            "x-api-key": config.api_key,
            "anthropic-version": self._anthropic_version,
        }

        return url, payload, headers

    def _parse_retry_after(self, response: httpx.Response) -> float | None:
        """Parse Retry-After header."""
        val = response.headers.get("retry-after")
        if not val:
            return None
        try:
            return float(val)
        except ValueError:
            pass
        try:
            dt = parsedate_to_datetime(val)
            return max(0.0, dt.timestamp() - time.time())
        except Exception:
            return None

    def _handle_http_error(self, response: httpx.Response) -> None:
        """Map Anthropic HTTP error codes to domain exceptions."""
        status = response.status_code
        try:
            raw_text = response.text
        except Exception:
            raw_text = "<unreadable response body>"

        if status in (401, 403):
            raise ProviderAuthenticationError(
                message=f"Anthropic authentication failed: HTTP {status}",
                provider_name=self._name,
                status_code=status,
                raw_response=raw_text,
            )

        if status == 429:
            retry_after = self._parse_retry_after(response)
            raise ProviderRateLimitError(
                message="Anthropic rate limit exceeded",
                provider_name=self._name,
                retry_after=retry_after,
                status_code=429,
                raw_response=raw_text,
            )

        if status == 400:
            raise ProviderBadRequestError(
                message="Anthropic rejected request payload",
                provider_name=self._name,
                status_code=400,
                raw_response=raw_text,
            )

        if 500 <= status <= 599:
            raise ProviderUnavailableError(
                message=f"Anthropic server error: HTTP {status}",
                provider_name=self._name,
                status_code=status,
                raw_response=raw_text,
            )

        raise ProviderError(
            message=f"Anthropic request failed: HTTP {status}",
            provider_name=self._name,
            status_code=status,
            raw_response=raw_text,
        )

    async def complete(
        self,
        request: ChatCompletionRequest,
        config: ProviderConfig,
    ) -> ChatCompletionResponse:
        """Execute completion and translate Anthropic JSON response to OpenAI schema."""
        url, payload, headers = self._build_payload_and_headers(request, config)
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
                message=f"Anthropic timeout after {config.timeout_seconds}s",
                provider_name=self._name,
            ) from exc
        except httpx.NetworkError as exc:
            raise ProviderUnavailableError(
                message=f"Anthropic network error: {exc}",
                provider_name=self._name,
            ) from exc

        if response.is_error:
            self._handle_http_error(response)

        data = orjson.loads(response.content)

        # Extract textual content from content blocks
        text_content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_content += block.get("text", "")

        finish_reason = map_anthropic_stop_reason(data.get("stop_reason"))
        usage_data = data.get("usage", {})
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)

        return ChatCompletionResponse(
            id=data.get("id", f"msg-{int(time.time())}"),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role=Role.ASSISTANT,
                        content=text_content,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )

    async def stream(
        self,
        request: ChatCompletionRequest,
        config: ProviderConfig,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Execute stream and translate Anthropic SSE event stream to OpenAI SSE chunks."""
        url, payload, headers = self._build_payload_and_headers(request, config)
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
                message=f"Anthropic stream timeout: {exc}",
                provider_name=self._name,
            ) from exc
        except httpx.NetworkError as exc:
            raise ProviderUnavailableError(
                message=f"Anthropic stream network error: {exc}",
                provider_name=self._name,
            ) from exc

        if response.is_error:
            with contextlib.suppress(Exception):
                await response.aread()
            self._handle_http_error(response)

        current_event = ""
        msg_id = f"msg-{int(time.time())}"

        try:
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or line.startswith(":"):
                    continue

                if line.startswith("event: "):
                    current_event = line[7:].strip()
                    continue

                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if not data_str:
                        continue
                    try:
                        data = orjson.loads(data_str)
                    except (orjson.JSONDecodeError, ValueError) as err:
                        logger.debug("Skipping unparseable SSE line: %s", err)
                        continue

                    # Handle Anthropic streaming events
                    if current_event == "message_start":
                        msg_obj = data.get("message", {})
                        msg_id = msg_obj.get("id", msg_id)
                        yield ChatCompletionChunk(
                            id=msg_id,
                            model=request.model,
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta=DeltaMessage(role=Role.ASSISTANT, content=""),
                                    finish_reason=None,
                                )
                            ],
                        )

                    elif current_event == "content_block_delta":
                        delta_obj = data.get("delta", {})
                        if delta_obj.get("type") == "text_delta":
                            text_piece = delta_obj.get("text", "")
                            if text_piece:
                                yield ChatCompletionChunk(
                                    id=msg_id,
                                    model=request.model,
                                    choices=[
                                        StreamChoice(
                                            index=0,
                                            delta=DeltaMessage(content=text_piece),
                                            finish_reason=None,
                                        )
                                    ],
                                )

                    elif current_event == "message_delta":
                        delta_obj = data.get("delta", {})
                        stop_reason = map_anthropic_stop_reason(delta_obj.get("stop_reason"))
                        usage_obj = data.get("usage", {})
                        output_tokens = usage_obj.get("output_tokens", 0)

                        yield ChatCompletionChunk(
                            id=msg_id,
                            model=request.model,
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta=DeltaMessage(content=""),
                                    finish_reason=stop_reason,
                                )
                            ],
                            usage=UsageInfo(
                                prompt_tokens=0,
                                completion_tokens=output_tokens,
                                total_tokens=output_tokens,
                            )
                            if output_tokens
                            else None,
                        )

                    elif current_event == "message_stop":
                        break
        finally:
            await response.aclose()

    async def health_check(self, config: ProviderConfig) -> bool:
        """Verify Anthropic connectivity."""
        try:
            return bool(config.api_key)
        except Exception:
            return False
