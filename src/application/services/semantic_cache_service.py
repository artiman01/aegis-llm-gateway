"""Application service orchestrating two-tier (L1 Exact Hash + L2 Semantic) caching."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.services.saliency_guard_service import SaliencyGuardService
from domain.models.chat import ChatCompletionRequest, ChatCompletionResponse

if TYPE_CHECKING:
    from domain.ports.cache_port import L1CachePort, L2SemanticCachePort
    from domain.ports.embedding_port import EmbeddingPort
    from domain.ports.metrics_port import MetricsPort

logger = logging.getLogger(__name__)


class SemanticCacheService:
    """Manages hierarchical caching with fail-open safety, saliency gating, and telemetry."""

    def __init__(
        self,
        l1_cache: L1CachePort | None = None,
        l2_cache: L2SemanticCachePort | None = None,
        *,
        embedding: EmbeddingPort | None = None,
        metrics: MetricsPort | None = None,
        saliency_guard: SaliencyGuardService | None = None,
        similarity_threshold: float = 0.90,
        l1_ttl_seconds: int = 3600,
        l2_ttl_seconds: int = 86400,
    ) -> None:
        self._l1_cache = l1_cache
        self._l2_cache = l2_cache
        self._embedding = embedding
        self._metrics = metrics
        self._saliency_guard = saliency_guard
        self._similarity_threshold = similarity_threshold
        self._l1_ttl_seconds = l1_ttl_seconds
        self._l2_ttl_seconds = l2_ttl_seconds

    def is_cache_eligible(self, request: ChatCompletionRequest) -> bool:
        """Evaluate whether a request is eligible for semantic L2 cache lookups and storage.

        Semantic L2 caching is bypassed when:
        1. Temperature is non-zero (request.temperature > 0.0) indicating non-deterministic output.
        2. Tools or functions are configured in the request definition or conversation history.
        """
        if request.temperature is not None and request.temperature > 0.0:
            return False

        if request.tools or request.tool_choice:
            return False

        for msg in request.messages:
            if msg.tool_calls or msg.role.value in ("tool", "function"):
                return False

        return True

    async def get(
        self,
        request: ChatCompletionRequest,
        bypass_cache: bool = False,
    ) -> ChatCompletionResponse | None:
        """Lookup cached completion across L1 and L2 tiers.

        Returns None if cache is bypassed, missing, or encounters an internal fault.
        """
        if bypass_cache or request.stream:
            return None

        # Tier 1: Exact Hash Match (<1ms)
        if self._l1_cache is not None:
            try:
                exact_key = request.compute_cache_key()
                cached_l1 = await self._l1_cache.get(exact_key)
                if cached_l1 is not None:
                    if self._metrics is not None:
                        self._metrics.record_cache_access(tier="l1_exact", hit=True)
                    # Mark response provenance
                    cached_l1.system_fingerprint = "cache:l1_exact"
                    return cached_l1
                if self._metrics is not None:
                    self._metrics.record_cache_access(tier="l1_exact", hit=False)
            except Exception as e:
                logger.warning("L1 cache retrieval failed (fail-open): %s", e)

        # Tier 2: Semantic Similarity Match (~5-15ms)
        if (
            self._l2_cache is not None
            and self._embedding is not None
            and self.is_cache_eligible(request)
        ):
            try:
                prompt_text = request.extract_prompt_text()
                if prompt_text:
                    query_embedding = await self._embedding.embed_text(prompt_text)
                    user_identifier = request.user or request.tenant_id
                    match = await self._l2_cache.search(
                        embedding=query_embedding,
                        model=request.model,
                        similarity_threshold=self._similarity_threshold,
                        user=user_identifier,
                    )
                    if match is not None:
                        cached_l2, score = match

                        # Evaluate two-phase saliency invariant and whitened similarity gate
                        if self._saliency_guard is not None:
                            cached_prompt = getattr(cached_l2, "_cached_prompt", None)
                            cached_vec = getattr(cached_l2, "_cached_embedding", None)
                            if cached_prompt is not None and cached_vec is not None:
                                is_valid, whitened_score, reason = (
                                    self._saliency_guard.verify_cache_candidate(
                                        query_text=prompt_text,
                                        query_vec=query_embedding,
                                        cached_text=cached_prompt,
                                        cached_vec=cached_vec,
                                    )
                                )
                                if not is_valid:
                                    logger.info("SaliencyGuard rejected semantic hit: %s", reason)
                                    if self._metrics is not None:
                                        self._metrics.record_cache_access(
                                            tier="l2_semantic", hit=False
                                        )
                                    return None
                                score = whitened_score

                        if self._metrics is not None:
                            self._metrics.record_cache_access(tier="l2_semantic", hit=True)
                        cached_l2.system_fingerprint = f"cache:l2_semantic:{score:.3f}"
                        return cached_l2

                    if self._metrics is not None:
                        self._metrics.record_cache_access(tier="l2_semantic", hit=False)
            except Exception as e:
                logger.warning("L2 semantic cache retrieval failed (fail-open): %s", e)

        return None

    async def set(
        self,
        request: ChatCompletionRequest,
        response: ChatCompletionResponse,
        bypass_cache: bool = False,
    ) -> None:
        """Populate L1 and L2 caches asynchronously with fail-open resilience."""
        if bypass_cache or request.stream:
            return

        # Populate L1 Exact Cache
        if self._l1_cache is not None:
            try:
                exact_key = request.compute_cache_key()
                await self._l1_cache.set(
                    key=exact_key,
                    response=response,
                    ttl_seconds=self._l1_ttl_seconds,
                )
            except Exception as e:
                logger.warning("L1 cache write failed: %s", e)

        # Populate L2 Semantic Cache
        if (
            self._l2_cache is not None
            and self._embedding is not None
            and self.is_cache_eligible(request)
        ):
            try:
                prompt_text = request.extract_prompt_text()
                if prompt_text:
                    query_embedding = await self._embedding.embed_text(prompt_text)
                    user_identifier = request.user or request.tenant_id
                    await self._l2_cache.store(
                        prompt=prompt_text,
                        embedding=query_embedding,
                        model=request.model,
                        response=response,
                        ttl_seconds=self._l2_ttl_seconds,
                        user=user_identifier,
                    )
            except Exception as e:
                logger.warning("L2 semantic cache write failed: %s", e)
