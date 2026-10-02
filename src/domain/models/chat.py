"""OpenAI-compatible Domain Models for Chat Completions.

Implements full schema validation for `/v1/chat/completions` request, response,
and Server-Sent Events (SSE) streaming chunks under Pydantic v2.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Compatibility fallback for Python < 3.11."""

        pass


from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class Role(StrEnum):
    """Message author role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class FinishReason(StrEnum):
    """Reason for completion termination."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    FUNCTION_CALL = "function_call"


# Content Parts (Multimodal Support)


class ImageURL(BaseModel):
    """Image URL payload for multimodal vision inputs."""

    url: str
    detail: Literal["auto", "low", "high"] = "auto"

    model_config = ConfigDict(extra="ignore")


class ContentPartText(BaseModel):
    """Textual component of a multipart message."""

    type: Literal["text"] = "text"
    text: str

    model_config = ConfigDict(extra="ignore")


class ContentPartImage(BaseModel):
    """Visual component of a multipart message."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageURL

    model_config = ConfigDict(extra="ignore")


ContentPart = Annotated[ContentPartText | ContentPartImage, Field(discriminator="type")]


# Tool Calling & Function Calling


class FunctionCall(BaseModel):
    """Legacy and tool-based function invocation details."""

    name: str
    arguments: str

    model_config = ConfigDict(extra="ignore")


class ToolCall(BaseModel):
    """OpenAI tool invocation structure."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall

    model_config = ConfigDict(extra="ignore")


class FunctionDefinition(BaseModel):
    """Definition of a function callable by the model."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None

    model_config = ConfigDict(extra="ignore")


class ToolDefinition(BaseModel):
    """Definition of an available tool."""

    type: Literal["function"] = "function"
    function: FunctionDefinition

    model_config = ConfigDict(extra="ignore")


# Chat Messages


class ChatMessage(BaseModel):
    """Message entity within a conversation turn."""

    role: Role
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    model_config = ConfigDict(extra="ignore")

    def get_text_content(self) -> str:
        """Extract flat text representation for caching and semantic indexing."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        texts: list[str] = []
        for part in self.content:
            if isinstance(part, ContentPartText):
                texts.append(part.text)
        return " ".join(texts)


# Request Definition


class ResponseFormat(BaseModel):
    """Format constraint for model output."""

    type: Literal["text", "json_object"] = "text"

    model_config = ConfigDict(extra="ignore")


class StreamOptions(BaseModel):
    """Additional options for streaming responses."""

    include_usage: bool = False

    model_config = ConfigDict(extra="ignore")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request schema."""

    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    n: int | None = Field(default=1, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    tenant_id: str | None = None
    response_format: ResponseFormat | None = None
    seed: int | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    stream_options: StreamOptions | None = None

    model_config = ConfigDict(extra="ignore")

    def compute_cache_key(self) -> str:
        """Compute deterministic SHA-256 hash for L1 exact match cache.

        Incorporates model, serialized messages (including tool_calls), temperature,
        top_p, max_tokens, stop, seed, presence_penalty, frequency_penalty, logit_bias,
        tool_choice, tools, response_format, and strictly isolates under a multi-tenant namespace.
        """
        namespace = f"tenant:{self.tenant_id or 'default'}:user:{self.user or 'anonymous'}"
        canonical_dict = {
            "model": self.model,
            "messages": [
                {
                    "role": m.role.value,
                    "content": m.get_text_content(),
                    "name": m.name,
                    "tool_calls": [tc.model_dump() for tc in m.tool_calls]
                    if m.tool_calls
                    else None,
                    "tool_call_id": m.tool_call_id,
                }
                for m in self.messages
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "tool_choice": self.tool_choice,
            "tools": [t.model_dump() for t in self.tools] if self.tools else None,
            "response_format": self.response_format.model_dump() if self.response_format else None,
        }
        encoded = json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        return f"llm:exact:{namespace}:{digest}"

    def extract_prompt_text(self) -> str:
        """Extract concatenated prompt text for L2 semantic embedding generation."""
        extracted: list[str] = []
        for msg in self.messages:
            text = msg.get_text_content()
            if text:
                extracted.append(f"{msg.role.value}: {text}")
        return "\n".join(extracted)


# Response Definition


class UsageInfo(BaseModel):
    """Token consumption statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    model_config = ConfigDict(extra="ignore")


class ChoiceMessage(BaseModel):
    """Model-generated message in response."""

    role: Role = Role.ASSISTANT
    content: str | None = None
    tool_calls: list[ToolCall] | None = None

    model_config = ConfigDict(extra="ignore")


class Choice(BaseModel):
    """A single completion choice option."""

    index: int = 0
    message: ChoiceMessage
    finish_reason: str | None = None
    logprobs: Any | None = None

    model_config = ConfigDict(extra="ignore")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible synchronous chat completion response schema."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: UsageInfo | None = None
    system_fingerprint: str | None = None
    _cached_prompt: str | None = PrivateAttr(default=None)
    _cached_embedding: list[float] | None = PrivateAttr(default=None)
    _cached_user: str | None = PrivateAttr(default=None)

    model_config = ConfigDict(extra="ignore")


# Streaming Chunk Definition (SSE)


class DeltaMessage(BaseModel):
    """Incremental delta delivered inside a streaming chunk."""

    role: Role | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None

    model_config = ConfigDict(extra="ignore")


class StreamChoice(BaseModel):
    """Choice representation in a stream chunk."""

    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None
    logprobs: Any | None = None

    model_config = ConfigDict(extra="ignore")


class ChatCompletionChunk(BaseModel):
    """Server-Sent Event (SSE) chunk payload for streamed completions."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]
    system_fingerprint: str | None = None
    usage: UsageInfo | None = None

    model_config = ConfigDict(extra="ignore")

    def to_sse_event(self) -> str:
        """Format chunk as standard Server-Sent Event (SSE) data line."""
        return f"data: {self.model_dump_json()}\n\n"
