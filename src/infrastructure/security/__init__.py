"""Infrastructure security module export."""

from infrastructure.security.streaming_dfa_automaton import (
    DEFAULT_PREFIX_PATTERNS,
    DEFAULT_SENSITIVE_PATTERNS,
    StreamingDFAAutomaton,
)

__all__ = [
    "DEFAULT_PREFIX_PATTERNS",
    "DEFAULT_SENSITIVE_PATTERNS",
    "StreamingDFAAutomaton",
]
