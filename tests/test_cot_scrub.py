"""Assignment-compliance: SSE wire must NEVER carry chain-of-thought.

The Hindi smoke run (2026-05-22) exposed that the synthesizer occasionally
emits `<think>...</think>` blocks (DeepSeek R1 / Gemini extended-thinking).
Our `_COT_PATTERNS` regex only covered `<thinking>` and `<thought>` —
missing the short-form `<think>`. These tests lock both the regex coverage
and the inline scrub behaviour so a regression can't ship.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    _COT_PATTERNS,
    _contains_cot,
    _scrub_cot_in_payload,
    _strip_cot_from_text,
)


# ── pattern coverage ──────────────────────────────────────────────────────


def test_pattern_catches_short_think_tag():
    assert _COT_PATTERNS.search("<think>foo</think>") is not None


def test_pattern_catches_long_thinking_tag():
    assert _COT_PATTERNS.search("<thinking>foo</thinking>") is not None


def test_pattern_catches_thought_tag():
    assert _COT_PATTERNS.search("<thought>foo</thought>") is not None


def test_pattern_catches_envelope_keys():
    for key in (
        "thought_summary",
        "thought_tokens",
        "reasoning_content",
        "reasoning_tokens",
        "redacted_thinking",
    ):
        assert _COT_PATTERNS.search(key) is not None, key


def test_pattern_is_case_insensitive():
    assert _COT_PATTERNS.search("<THINK>foo</THINK>") is not None
    assert _COT_PATTERNS.search("<Thinking>foo</Thinking>") is not None


# ── inline scrub ──────────────────────────────────────────────────────────


def test_strip_full_block():
    text = "Answer prefix <think>internal reasoning here</think> answer suffix."
    out = _strip_cot_from_text(text)
    assert "<think>" not in out
    assert "internal reasoning" not in out
    assert "Answer prefix" in out
    assert "answer suffix" in out


def test_strip_full_block_long_form():
    text = "Before <thinking>secret</thinking> After"
    out = _strip_cot_from_text(text)
    assert out == "Before  After"


def test_strip_open_only_mid_stream():
    # Mid-stream chunk that contains the open tag but no close yet — must
    # drop everything from the open tag onwards.
    text = "Visible text <think>partial reasoning never closed in this chunk"
    out = _strip_cot_from_text(text)
    assert "Visible text" in out
    assert "<think>" not in out
    assert "partial reasoning" not in out


def test_strip_close_only_after_open_consumed():
    # Earlier chunk consumed the open tag (already scrubbed); this chunk
    # carries the close tag with trailing legitimate text.
    text = "tail of reasoning </think> Real answer continues here."
    out = _strip_cot_from_text(text)
    assert "Real answer continues here." in out
    assert "</think>" not in out
    assert "tail of reasoning" not in out


def test_strip_idempotent_on_clean_text():
    text = "Just a normal answer with no reasoning tags."
    assert _strip_cot_from_text(text) == text


# ── payload integration ───────────────────────────────────────────────────


def test_scrub_answer_delta_text_field():
    payload = {
        "step": "generating",
        "label": "Generating answer with citations",
        "type": "answer_delta",
        "data": {"text": "Pre <think>x</think> Post"},
    }
    cleaned = _scrub_cot_in_payload(payload)
    assert cleaned["data"]["text"] == "Pre  Post"
    # Original untouched.
    assert payload["data"]["text"] == "Pre <think>x</think> Post"


def test_scrub_legacy_generating_string_payload():
    payload = {
        "step": "generating",
        "label": "Generating answer with citations",
        "data": "Hello <think>secret</think> world",
    }
    cleaned = _scrub_cot_in_payload(payload)
    assert cleaned["data"] == "Hello  world"


def test_scrub_leaves_structured_data_alone():
    payload = {
        "step": "phase_finished",
        "label": "ok",
        "type": "phase_finished",
        "data": {"name": "planning", "duration_ms": 42},
    }
    cleaned = _scrub_cot_in_payload(payload)
    assert cleaned == payload  # untouched


def test_contains_cot_after_scrub_returns_false_for_clean_payload():
    payload = {"step": "generating", "data": {"text": "Pre <think>x</think> Post"}}
    cleaned = _scrub_cot_in_payload(payload)
    assert not _contains_cot(cleaned)


def test_contains_cot_still_catches_envelope_after_scrub():
    # Envelope keys aren't text content — the scrub doesn't try to remove
    # them; the contains-check drops the whole frame for safety.
    payload = {"step": "generating", "data": {"reasoning_content": "x"}}
    assert _contains_cot(payload)


def test_scrub_recursive_strips_done_answer_field():
    """The `done` event carries the full answer in data.answer. The scrub
    must remove `<think>...</think>` from there so the contains-check below
    doesn't drop the entire done frame."""
    payload = {
        "step": "done",
        "label": "done",
        "data": {
            "turn_id": "t1",
            "answer": "<think>plan</think>The repo rate is 5.25%.",
            "urls": ["https://example.com"],
        },
    }
    cleaned = _scrub_cot_in_payload(payload)
    assert cleaned["data"]["answer"] == "The repo rate is 5.25%."
    assert not _contains_cot(cleaned)
    # Other fields are preserved.
    assert cleaned["data"]["urls"] == ["https://example.com"]
    assert cleaned["data"]["turn_id"] == "t1"


def test_scrub_recursive_walks_nested_lists():
    payload = {
        "step": "done",
        "data": {
            "messages": [
                {"text": "<think>x</think>hi"},
                {"text": "bye"},
            ],
        },
    }
    cleaned = _scrub_cot_in_payload(payload)
    assert cleaned["data"]["messages"][0]["text"] == "hi"
    assert cleaned["data"]["messages"][1]["text"] == "bye"
