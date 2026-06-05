"""Unit tests for the few-shot message builder."""

from invoice_extractor.llm.prompts import build_messages


def test_build_messages_structure():
    messages = build_messages("Invoice no: 12345678")
    # 2 few-shot pairs (4 turns) + 1 actual query = 5 messages
    assert len(messages) == 5
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user", "assistant", "user"]


def test_build_messages_includes_query():
    messages = build_messages("UNIQUE_OCR_MARKER")
    assert "UNIQUE_OCR_MARKER" in messages[-1]["content"]


def test_few_shot_answers_are_json_like():
    import json

    messages = build_messages("x")
    # The two assistant turns should be parseable JSON objects.
    for idx in (1, 3):
        parsed = json.loads(messages[idx]["content"])
        assert parsed["document_type"] == "invoice"
