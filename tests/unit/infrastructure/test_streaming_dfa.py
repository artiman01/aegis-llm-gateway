"""Unit tests for Stateful Streaming Token-Boundary DFA Automaton."""

from __future__ import annotations

import pytest

from infrastructure.security.streaming_dfa_automaton import StreamingDFAAutomaton


@pytest.fixture
def dfa() -> StreamingDFAAutomaton:
    """Fixture providing a default StreamingDFAAutomaton instance."""
    return StreamingDFAAutomaton()


class TestStreamingDFAAutomaton:
    """Mathematical and state transition tests for token-boundary DFA redaction."""

    def test_single_chunk_full_match_redaction(self, dfa: StreamingDFAAutomaton) -> None:
        """Verify standard redaction of sensitive tokens within a single chunk."""
        chunk = "Here is an OpenAI secret: sk-1234567890abcdef1234 and AWS AKIAIOSFODNN7EXAMPLE."
        processed = dfa.process_chunk(chunk)
        flushed = dfa.flush()
        full_output = processed + flushed

        assert "sk-1234567890abcdef1234" not in full_output
        assert "AKIAIOSFODNN7EXAMPLE" not in full_output
        assert "[REDACTED]" in full_output

    def test_cross_chunk_openai_api_key_split(self, dfa: StreamingDFAAutomaton) -> None:
        """Verify detection and redaction of an API key cut across SSE chunks."""
        # Simulated BPE split between "sk-pr" and "oj-1234567890testabc"
        chunk1 = "The model key is sk-pr"
        chunk2 = "oj-1234567890testabc is active."

        out1 = dfa.process_chunk(chunk1)
        out2 = dfa.process_chunk(chunk2)
        flushed = dfa.flush()

        combined = out1 + out2 + flushed

        assert "sk-proj-1234567890testabc" not in combined
        assert "sk-pr" not in combined
        assert combined == "The model key is [REDACTED] is active."

    def test_cross_chunk_credit_card_split_three_parts(self, dfa: StreamingDFAAutomaton) -> None:
        """Verify redaction of credit card numbers split across multiple streaming boundaries."""
        chunks = [
            "Customer card: 4111-2222-",
            "3333-",
            "4444 completed transaction.",
        ]

        emitted: list[str] = []
        for ch in chunks:
            emitted.append(dfa.process_chunk(ch))
        emitted.append(dfa.flush())

        full_result = "".join(emitted)

        assert "4111-2222-3333-4444" not in full_result
        assert full_result == "Customer card: [REDACTED] completed transaction."

    def test_cross_chunk_aws_key_split(self, dfa: StreamingDFAAutomaton) -> None:
        """Verify redaction of AWS access key split at prefix boundary."""
        chunks = [
            "Config: key=AKIA",
            "1234567890ABCDEF and region=us-east-1",
        ]

        emitted: list[str] = []
        for ch in chunks:
            emitted.append(dfa.process_chunk(ch))
        emitted.append(dfa.flush())

        full_result = "".join(emitted)

        assert "AKIA1234567890ABCDEF" not in full_result
        assert full_result == "Config: key=[REDACTED] and region=us-east-1"

    def test_benign_text_stream_zero_buffering_latency(self, dfa: StreamingDFAAutomaton) -> None:
        """Verify normal stream passes chunks immediately without buffering delay."""
        chunks = [
            "The quick ",
            "brown fox ",
            "jumps over ",
            "the lazy dog.",
        ]

        emitted = [dfa.process_chunk(ch) for ch in chunks]
        final_flush = dfa.flush()

        # Each chunk without sensitive token prefixes should emit immediately
        assert emitted == chunks
        assert final_flush == ""
        assert dfa.carry_over == ""

    def test_flush_emits_unmatched_prefix_at_stream_end(self, dfa: StreamingDFAAutomaton) -> None:
        """Verify uncompleted prefixes are cleanly flushed at stream completion."""
        # "sk-" at the very end of stream that does not form a full key
        chunk = "This was just sk-"
        out = dfa.process_chunk(chunk)
        flushed = dfa.flush()

        assert out == "This was just "
        assert flushed == "sk-"
        assert dfa.carry_over == ""

    def test_custom_redaction_replacement(self) -> None:
        """Verify custom replacement string is respected."""
        custom_dfa = StreamingDFAAutomaton(replacement="***SENSITIVE***")
        chunk = "Secret: sk-abcdef1234567890"

        out = custom_dfa.process_chunk(chunk) + custom_dfa.flush()
        assert "***SENSITIVE***" in out
        assert "[REDACTED]" not in out

    def test_reset_clears_internal_state(self, dfa: StreamingDFAAutomaton) -> None:
        """Verify reset() clears internal carry-over buffer."""
        dfa.process_chunk("Prefix sk-")
        assert dfa.carry_over != ""

        dfa.reset()
        assert dfa.carry_over == ""
        assert dfa.flush() == ""

    def test_process_bytes_split_multibyte_utf8_cyrillic(self, dfa: StreamingDFAAutomaton) -> None:
        """Verify raw bytes split across multibyte UTF-8 boundary decode and redact seamlessly."""
        text = "Привет, вот ключ: sk-proj-1234567890abcdef1234"
        raw_bytes = text.encode("utf-8")

        # In UTF-8, 'р' is 2 bytes: 0xD1 0x80. Split right between those two bytes!
        split_point = 3  # Inside Cyrillic 'р'
        chunk1 = raw_bytes[:split_point]
        chunk2 = raw_bytes[split_point:]

        out1 = dfa.process_bytes(chunk1)
        out2 = dfa.process_bytes(chunk2)
        flushed = dfa.flush()

        combined = out1 + out2 + flushed
        assert combined == "Привет, вот ключ: [REDACTED]"
        assert "sk-proj" not in combined
