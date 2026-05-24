"""Tests for the streaming CoT/reasoning-tag filter.

Background: users reported seeing <think>...</think> reasoning leaking into
the SSE pipeline when Sarvam-m or Cerebras-routed Qwen synthesized answers.
The filter strips these blocks at the chain level. Tests cover:
  - whole-chunk reasoning is removed
  - reasoning split across chunk boundaries is removed
  - the closing tag landing in a later chunk doesn't leak head text
  - genuine `<` characters in answer text survive (e.g. "5 < 10")
  - truncated streams (no closing tag) drop the partial CoT rather than leak
"""
from __future__ import annotations

import pytest

from utils.cot_filter import CotStreamFilter


def _drive(filter_: CotStreamFilter, chunks: list[str]) -> str:
    out: list[str] = []
    for c in chunks:
        out.append(filter_.feed(c))
    out.append(filter_.flush())
    return "".join(out)


def test_strips_single_chunk_think_block():
    f = CotStreamFilter()
    assert _drive(f, ["<think>plan it out</think>The capital is Paris."]) == "The capital is Paris."


def test_strips_thinking_block_variant():
    f = CotStreamFilter()
    assert _drive(f, ["<thinking>x</thinking>Answer."]) == "Answer."


def test_strips_when_open_and_close_in_different_chunks():
    f = CotStreamFilter()
    out = _drive(f, ["<think>", "step 1", " step 2", "</think>Final answer."])
    assert out == "Final answer."


def test_open_tag_split_across_chunks_does_not_leak_partial():
    """If `<thi` arrives, then `nk>...</think>real`, the filter must not have
    emitted `<thi` to the user."""
    f = CotStreamFilter()
    out = _drive(f, ["<thi", "nk>scratch</think>real"])
    assert out == "real"


def test_close_tag_split_across_chunks():
    f = CotStreamFilter()
    out = _drive(f, ["<think>scratch</thi", "nk>done"])
    assert out == "done"


def test_preserves_legitimate_angle_brackets_in_answer():
    """An angle bracket in the answer ('5 < 10') must survive once we know it
    is NOT the start of a reasoning tag."""
    f = CotStreamFilter()
    out = _drive(f, ["The result is 5 < 10 and 20 > 1."])
    assert out == "The result is 5 < 10 and 20 > 1."


def test_drops_unterminated_think_block_on_flush():
    """If the stream ends mid-CoT (model truncated), we must NOT flush the
    partial reasoning to the user."""
    f = CotStreamFilter()
    out = _drive(f, ["Answer is X.<think>oh wait the answer is actu"])
    assert out == "Answer is X."


def test_text_before_think_block_is_preserved():
    f = CotStreamFilter()
    out = _drive(f, ["Preamble. <think>scratch</think> Answer."])
    assert out == "Preamble.  Answer."


def test_multiple_think_blocks_in_one_stream():
    f = CotStreamFilter()
    out = _drive(f, ["<think>a</think>One.<think>b</think>Two."])
    assert out == "One.Two."


def test_empty_input_is_safe():
    f = CotStreamFilter()
    assert _drive(f, []) == ""
    assert _drive(f, ["", ""]) == ""


def test_character_by_character_streaming():
    """Worst-case chunking: every character its own chunk. Filter must still
    catch the reasoning block."""
    f = CotStreamFilter()
    payload = "<think>scratch</think>OK."
    out = _drive(f, list(payload))
    assert out == "OK."
