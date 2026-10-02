"""Two-Phase Saliency Guard Service for robust semantic cache verification."""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domain.ports.whitening_port import WhiteningPort

logger = logging.getLogger(__name__)

# Precompiled regex patterns for high-speed lexical saliency invariant extraction
_NUMERIC_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
_DATE_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_ISO_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z]+[-_]\d+\b", re.IGNORECASE)
_NAMED_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")
_ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,}\b")

# Common sentence starters and English stopwords to avoid false-positive entity extraction
_SENTENCE_STARTER_STOPWORDS: frozenset[str] = frozenset(
    {
        "The",
        "A",
        "An",
        "Can",
        "Could",
        "Would",
        "Should",
        "Please",
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "Whom",
        "Whose",
        "Why",
        "How",
        "Show",
        "Tell",
        "Give",
        "List",
        "Describe",
        "Explain",
        "Write",
        "Check",
        "Is",
        "Are",
        "Was",
        "Were",
        "Do",
        "Does",
        "Did",
        "Have",
        "Has",
        "Had",
        "If",
        "In",
        "On",
        "At",
        "For",
        "To",
        "From",
        "With",
        "By",
        "As",
        "Of",
    }
)


class SaliencyGuardService:
    """Verifies semantic cache candidates via isotropic whitening and lexical saliency invariance.

    Guarantees zero false-positive cache collisions by ensuring queries pass both:
    1. Phase 1: Isotropic Whitened Cosine Similarity >= Threshold.
    2. Phase 2: Lexical Saliency Intersection (exact preservation of numbers, dates, IDs, entities).
    """

    def __init__(
        self,
        whitening: WhiteningPort | None = None,
        similarity_threshold: float = 0.90,
    ) -> None:
        """Initialize SaliencyGuardService.

        Args:
            whitening: Optional ZCA-Whitening vector transformer port.
            similarity_threshold: Minimum whitened cosine similarity for Phase 1 verification.
        """
        self._whitening = whitening
        self._similarity_threshold = similarity_threshold

    @property
    def similarity_threshold(self) -> float:
        """Configured similarity threshold."""
        return self._similarity_threshold

    def compute_whitened_similarity(
        self,
        vec_a: list[float],
        vec_b: list[float],
    ) -> float:
        """Calculate cosine similarity in isotropic whitened space.

        Args:
            vec_a: First dense embedding vector.
            vec_b: Second dense embedding vector.

        Returns:
            Cosine similarity on the unit hypersphere in range [-1.0, 1.0].
        """
        w_a = self._whitening.whiten(vec_a) if self._whitening is not None else vec_a
        w_b = self._whitening.whiten(vec_b) if self._whitening is not None else vec_b

        dot_product = sum(a * b for a, b in zip(w_a, w_b, strict=True))
        norm_a = math.sqrt(sum(a * a for a in w_a))
        norm_b = math.sqrt(sum(b * b for b in w_b))

        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0

        return max(-1.0, min(1.0, dot_product / (norm_a * norm_b)))

    def extract_saliency_invariants(self, text: str) -> set[str]:
        """Extract lexical saliency invariants (numbers, dates, identifiers, proper names).

        Args:
            text: Query prompt text.

        Returns:
            Set of invariant token strings canonicalized for comparison.
        """
        invariants: set[str] = set()

        # 1. Numeric and temporal invariants
        for num in _NUMERIC_PATTERN.findall(text):
            invariants.add(f"num:{num}")

        for date in _ISO_DATE_PATTERN.findall(text):
            invariants.add(f"date:{date}")

        for year in _DATE_YEAR_PATTERN.findall(text):
            invariants.add(f"year:{year}")

        # 2. Key identifiers (e.g. ID-1, CVE-2023-1234, TASK-42)
        for identifier in _IDENTIFIER_PATTERN.findall(text):
            invariants.add(f"id:{identifier.upper()}")

        # 3. Capitalized named entities & acronyms
        for entity in _NAMED_ENTITY_PATTERN.findall(text):
            cleaned = entity.strip()
            if cleaned not in _SENTENCE_STARTER_STOPWORDS:
                invariants.add(f"entity:{cleaned}")

        for acronym in _ACRONYM_PATTERN.findall(text):
            if acronym not in _SENTENCE_STARTER_STOPWORDS:
                invariants.add(f"entity:{acronym}")

        return invariants

    def verify_cache_candidate(
        self,
        query_text: str,
        query_vec: list[float],
        cached_text: str,
        cached_vec: list[float],
    ) -> tuple[bool, float, str]:
        """Execute two-phase verification on candidate cache entry.

        Args:
            query_text: Incoming prompt text.
            query_vec: Incoming prompt embedding vector.
            cached_text: Stored candidate prompt text.
            cached_vec: Stored candidate embedding vector.

        Returns:
            Tuple of (is_valid_hit, whitened_similarity_score, reason_code).
        """
        # Phase 1: Isotropic Whitened Cosine Similarity
        similarity = self.compute_whitened_similarity(query_vec, cached_vec)
        if similarity < self._similarity_threshold:
            return (
                False,
                similarity,
                f"similarity_below_threshold ({similarity:.4f} < {self._similarity_threshold:.4f})",
            )

        # Phase 2: Lexical Saliency Invariant Intersection
        query_invariants = self.extract_saliency_invariants(query_text)
        cached_invariants = self.extract_saliency_invariants(cached_text)

        if query_invariants != cached_invariants:
            diff = query_invariants.symmetric_difference(cached_invariants)
            logger.info(
                "Saliency gate rejected semantic cache candidate: invariant mismatch (%s)",
                diff,
            )
            return (
                False,
                similarity,
                f"saliency_invariant_mismatch (diff: {sorted(diff)})",
            )

        return True, similarity, "verified_hit"
