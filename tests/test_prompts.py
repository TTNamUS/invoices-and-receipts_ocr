"""Unit tests for the message builder."""

from invoice_extractor.llm.prompts import build_messages


def test_build_messages_includes_query():
    messages = build_messages("UNIQUE_OCR_MARKER")
    assert "UNIQUE_OCR_MARKER" in messages[-1]["content"]
