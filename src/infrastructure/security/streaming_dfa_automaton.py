"""Stateful Deterministic Finite Automaton (DFA) for streaming token-boundary redaction."""

from __future__ import annotations

import codecs
import logging
import re
from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Canonical sensitive token patterns for LLM gateway data loss prevention
DEFAULT_SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # API Keys: sk-..., ant-..., ghp_...
    re.compile(r"\b(?:sk|ant|ghp|gho)_[a-zA-Z0-9_\-]{10,}\b"),
    re.compile(r"\bsk-[a-zA-Z0-9_\-]{10,}\b"),
    # AWS Access Key ID
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # Credit Card Numbers (13-19 digits with optional spaces/hyphens)
    re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),
    # Bearer Authorization Tokens
    re.compile(r"\bBearer\s+[a-zA-Z0-9_\-\.]{20,}\b", re.IGNORECASE),
    # Private Key Headers
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
)

# Prefix patterns to identify incomplete potential tokens split across streaming boundaries
DEFAULT_PREFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    # API Keys: sk-..., ant-..., ghp_...
    re.compile(r"(?:^|(?<=[\s\"':=]))(?:sk|ant|ghp|gho)_[a-zA-Z0-9_\-]{0,64}$"),
    re.compile(r"(?:^|(?<=[\s\"':=]))sk-[a-zA-Z0-9_\-]{0,64}$"),
    # AWS Access Key ID
    re.compile(r"(?:^|(?<=[\s\"':=]))AKIA[0-9A-Z]{0,16}$"),
    # Credit Card Numbers
    re.compile(r"(?:^|(?<=[\s\"':=]))(?:\d{1,4}[ -]?){1,4}$"),
    # Bearer Authorization Tokens
    re.compile(r"(?:^|(?<=[\s\"':=]))Bearer\s+[a-zA-Z0-9_\-\.]{0,64}$", re.IGNORECASE),
    # Private Key Headers
    re.compile(r"-----BEGIN[A-Z ]*$"),
)


class StreamingDFAAutomaton:
    """Stateful DFA Token-Boundary Matcher for streaming chunk redaction.

    Maintains carry-over state across SSE chunks to detect and redact sensitive
    tokens split across arbitrary tokenizer (BPE) boundaries without full stream buffering.
    """

    def __init__(
        self,
        patterns: Sequence[re.Pattern[str]] | None = None,
        prefix_patterns: Sequence[re.Pattern[str]] | None = None,
        replacement: str = "[REDACTED]",
        max_carry_over: int = 128,
    ) -> None:
        """Initialize StreamingDFAAutomaton.

        Args:
            patterns: Full regex patterns for sensitive token matching.
            prefix_patterns: Patterns to detect partial tokens at chunk boundaries.
            replacement: Redaction replacement text.
            max_carry_over: Maximum length of carry-over buffer.
        """
        self._patterns = (
            list(patterns) if patterns is not None else list(DEFAULT_SENSITIVE_PATTERNS)
        )
        self._prefix_patterns = (
            list(prefix_patterns) if prefix_patterns is not None else list(DEFAULT_PREFIX_PATTERNS)
        )
        self._replacement = replacement
        self._max_carry_over = max_carry_over
        self._carry_over = ""
        self._utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="surrogatepass")

    @property
    def carry_over(self) -> str:
        """Current internal carry-over buffer."""
        return self._carry_over

    def reset(self) -> None:
        """Reset internal DFA and UTF-8 incremental decoder state."""
        self._carry_over = ""
        self._utf8_decoder.reset()

    def process_bytes(self, chunk_bytes: bytes) -> str:
        """Process incremental raw bytes, resolving split multibyte UTF-8 sequences.

        Args:
            chunk_bytes: Incremental raw bytes from network stream.

        Returns:
            Sanitized and redacted text.
        """
        if not chunk_bytes:
            return ""
        decoded = self._utf8_decoder.decode(chunk_bytes, final=False)
        return self.process_chunk(decoded)

    def _redact_full_matches(self, text: str) -> str:
        """Apply all full sensitive regex patterns and replace occurrences."""
        redacted = text
        for pattern in self._patterns:
            redacted = pattern.sub(self._replacement, redacted)
        return redacted

    def _find_candidate_boundary_split(self, text: str) -> int:
        """Find the index where a potential cross-boundary token prefix begins.

        Returns:
            The start index in `text` of the earliest candidate prefix matching any
            prefix pattern at the tail of the string, or len(text) if no match is detected.
        """
        if not text:
            return 0

        # Scan tail up to max_carry_over
        search_window = min(len(text), self._max_carry_over)
        tail_start = len(text) - search_window
        tail_text = text[tail_start:]

        earliest_split_idx = len(text)

        for prefix_pattern in self._prefix_patterns:
            match = prefix_pattern.search(tail_text)
            if match is not None:
                split_idx = tail_start + match.start()
                earliest_split_idx = min(earliest_split_idx, split_idx)

        return earliest_split_idx

    def process_chunk(self, chunk_text: str) -> str:
        """Process an incremental chunk of streamed text and emit sanitized output.

        Appends chunk to carry-over buffer, redacts complete matches, detects
        boundary-crossing prefixes, and emits only the finalized non-ambiguous text.

        Args:
            chunk_text: Next incremental text delta from SSE stream.

        Returns:
            Sanitized text ready for downstream delivery.
        """
        if not chunk_text and not self._carry_over:
            return ""

        combined = self._carry_over + chunk_text
        self._carry_over = ""

        # Step 1: Redact any fully matched sensitive patterns
        redacted = self._redact_full_matches(combined)

        # Step 2: Determine if the tail of the string contains a partial token prefix
        split_idx = self._find_candidate_boundary_split(redacted)

        if split_idx < len(redacted):
            # Tail could be part of a split sensitive token: buffer it for the next chunk
            emitted_text = redacted[:split_idx]
            self._carry_over = redacted[split_idx:]
            return emitted_text

        # No partial prefix detected: emit entire text
        return redacted

    def flush(self) -> str:
        """Flush remaining carry-over buffer and decoder state at stream completion.

        Performs final redaction on buffered tail and returns remaining text.
        """
        final_decoded = self._utf8_decoder.decode(b"", final=True)
        if final_decoded:
            self._carry_over += final_decoded

        if not self._carry_over:
            return ""

        final_text = self._redact_full_matches(self._carry_over)
        self._carry_over = ""
        return final_text
