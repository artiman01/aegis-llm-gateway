"""Provider configuration and dynamic routing models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ProviderType(StrEnum):
    """Supported upstream provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"
    AZURE = "azure"
    OLLAMA = "ollama"


class ProviderConfig(BaseModel):
    """Configuration contract for an upstream LLM provider adapter."""

    name: str
    provider_type: ProviderType
    api_key: str = ""
    base_url: str
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    connect_timeout_seconds: float = Field(default=5.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0)
    priority: int = Field(
        default=1,
        ge=1,
        description="Routing priority (1 is highest priority, 2+ are fallbacks)",
    )
    weight: int = Field(default=100, ge=1, description="Traffic distribution weight")
    enabled: bool = True
    model_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from incoming model name to provider-specific upstream model name",
    )

    model_config = ConfigDict(extra="ignore")

    def get_upstream_model(self, requested_model: str) -> str:
        """Resolve upstream model name from mapping or fallback to requested model."""
        return self.model_mapping.get(requested_model, requested_model)


class RoutingRule(BaseModel):
    """Dynamic routing and fallback policy for a specific model or pattern."""

    virtual_model: str
    primary_provider: str
    fallback_providers: list[str] = Field(default_factory=list)
    retry_budget: int = Field(default=2, ge=0)
    timeout_seconds: float = Field(default=25.0, gt=0.0)

    model_config = ConfigDict(extra="ignore")


class GatewayConfig(BaseModel):
    """Global gateway orchestration settings."""

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    routes: dict[str, RoutingRule] = Field(default_factory=dict)
    default_timeout: float = 30.0
    circuit_breaker_enabled: bool = True
    l1_cache_enabled: bool = True
    l2_cache_enabled: bool = True
    l2_similarity_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    l1_cache_ttl_seconds: int = Field(default=3600, ge=0)
    l2_cache_ttl_seconds: int = Field(default=86400, ge=0)

    model_config = ConfigDict(extra="ignore")
