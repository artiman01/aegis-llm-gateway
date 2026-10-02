"""Unit tests for SaliencyGuardService two-phase semantic cache verification."""

from __future__ import annotations

import numpy as np
import pytest

from application.services.saliency_guard_service import SaliencyGuardService
from infrastructure.cache.whitening_transformer import WhiteningTransformer


@pytest.fixture
def calibrated_guard() -> SaliencyGuardService:
    """Provide SaliencyGuardService with calibrated ZCA-Whitening transformer."""
    dim = 384
    rng = np.random.default_rng(1234)
    # Background embeddings modeling an anisotropic narrow cone distribution
    center = np.ones(dim, dtype=np.float64) * 0.1
    noise = rng.normal(0.0, 0.02, size=(60, dim))
    calibration_data = (center + noise).tolist()

    transformer = WhiteningTransformer(dimension=dim, epsilon=1e-4)
    transformer.fit(calibration_data)
    return SaliencyGuardService(whitening=transformer, similarity_threshold=0.85)


@pytest.fixture
def base_vectors() -> tuple[list[float], list[float]]:
    """Provide two high-similarity embedding vectors for semantic cache candidates."""
    dim = 384
    rng = np.random.default_rng(5678)
    base = np.ones(dim, dtype=np.float64) * 0.1 + rng.normal(0.0, 0.01, size=dim)
    perturbed = base + rng.normal(0.0, 0.001, size=dim)
    return base.tolist(), perturbed.tolist()


def test_saliency_guard_temporal_drift_rejection(
    calibrated_guard: SaliencyGuardService,
    base_vectors: tuple[list[float], list[float]],
) -> None:
    """Queries with different years must trigger CACHE MISS despite high cosine similarity."""
    query_vec, cached_vec = base_vectors
    query_text = "Who is the CEO in 2024?"
    cached_text = "Who is the CEO in 2020?"

    is_valid, score, reason = calibrated_guard.verify_cache_candidate(
        query_text=query_text,
        query_vec=query_vec,
        cached_text=cached_text,
        cached_vec=cached_vec,
    )

    # Must pass Phase 1 (score > 0.85) but fail Phase 2 ('2024' != '2020')
    assert score >= 0.85
    assert is_valid is False
    assert "saliency_invariant_mismatch" in reason
    assert "2020" in reason or "2024" in reason


def test_saliency_guard_numeric_mismatch_rejection(
    calibrated_guard: SaliencyGuardService,
    base_vectors: tuple[list[float], list[float]],
) -> None:
    """Queries with different numerical values must trigger CACHE MISS."""
    query_vec, cached_vec = base_vectors
    query_text = "Total budget is 1000 USD."
    cached_text = "Total budget is 5000 USD."

    is_valid, score, reason = calibrated_guard.verify_cache_candidate(
        query_text=query_text,
        query_vec=query_vec,
        cached_text=cached_text,
        cached_vec=cached_vec,
    )

    assert score >= 0.85
    assert is_valid is False
    assert "saliency_invariant_mismatch" in reason
    assert "1000" in reason or "5000" in reason


def test_saliency_guard_identifier_mismatch_rejection(
    calibrated_guard: SaliencyGuardService,
    base_vectors: tuple[list[float], list[float]],
) -> None:
    """Queries with distinct entity identifiers must trigger CACHE MISS."""
    query_vec, cached_vec = base_vectors
    query_text = "Show ticket details for ID-100."
    cached_text = "Show ticket details for ID-200."

    is_valid, score, reason = calibrated_guard.verify_cache_candidate(
        query_text=query_text,
        query_vec=query_vec,
        cached_text=cached_text,
        cached_vec=cached_vec,
    )

    assert score >= 0.85
    assert is_valid is False
    assert "saliency_invariant_mismatch" in reason
    assert "ID-100" in reason or "ID-200" in reason


def test_saliency_guard_valid_semantic_hit(
    calibrated_guard: SaliencyGuardService,
    base_vectors: tuple[list[float], list[float]],
) -> None:
    """Syntactic paraphrases sharing the same invariants must be accepted as CACHE HIT."""
    query_vec, cached_vec = base_vectors
    query_text = "Can you describe the system architecture in 2024?"
    cached_text = "Please describe system architecture in 2024."

    is_valid, score, reason = calibrated_guard.verify_cache_candidate(
        query_text=query_text,
        query_vec=query_vec,
        cached_text=cached_text,
        cached_vec=cached_vec,
    )

    assert is_valid is True
    assert reason == "verified_hit"
    assert score >= 0.85
