"""Tests for B5 quote-anchored citation popovers."""
from __future__ import annotations

import pytest

from agent.citation_guard import _extract_anchor_quote, build_cite_quote_map


CHUNK = (
    "The Reserve Bank of India raised the repo rate to 6.5 percent in February. "
    "Inflation moderated to 5.1 percent year-on-year by March. "
    "The MPC voted 5-1 in favor of the hike. "
    "Officials cited persistent food price pressures."
)


@pytest.mark.unit
def test_extract_anchor_quote_matches_claim_sentence():
    claim = "the central bank raised the repo rate to 6.5 percent in February."
    quote = _extract_anchor_quote(claim, CHUNK)
    assert quote is not None
    assert "repo rate to 6.5 percent" in quote.lower()
    assert "inflation moderated" not in quote.lower()


@pytest.mark.unit
def test_extract_anchor_quote_returns_different_sentence_for_different_claim():
    claim = "year-on-year inflation eased to 5.1 percent by March."
    quote = _extract_anchor_quote(claim, CHUNK)
    assert quote is not None
    assert "inflation moderated to 5.1 percent" in quote.lower()


@pytest.mark.unit
def test_extract_anchor_quote_falls_back_to_chunk_head_on_no_overlap():
    claim = "unrelated topic about ocean currents and tides."
    quote = _extract_anchor_quote(claim, CHUNK, max_chars=80)
    assert quote is not None
    # Fallback returns the leading sentence-bounded prefix.
    assert quote.lower().startswith("the reserve bank of india")


@pytest.mark.unit
def test_extract_anchor_quote_handles_empty_inputs():
    assert _extract_anchor_quote("", CHUNK) is None
    assert _extract_anchor_quote("some claim", "") is None


@pytest.mark.unit
def test_extract_anchor_quote_respects_max_chars():
    claim = "the central bank raised the repo rate to 6.5 percent in February."
    quote = _extract_anchor_quote(claim, CHUNK, max_chars=40)
    assert quote is not None
    assert len(quote) <= 41  # +1 for the ellipsis appended on truncation


@pytest.mark.unit
def test_build_cite_quote_map_indexes_by_doc_id():
    answer = (
        "The RBI raised the repo rate to 6.5 percent in February. [doc_1] "
        "Inflation moderated to 5.1 percent year-on-year by March. [doc_2]"
    )
    doc_map = {
        "doc_1": ("RBI Policy", "https://rbi.org.in/policy", "rbi.org.in"),
        "doc_2": ("Inflation Report", "https://stats.gov/cpi", "stats.gov"),
    }
    snippet_lookup = {"doc_1": CHUNK, "doc_2": CHUNK}
    out = build_cite_quote_map(answer, doc_map, snippet_lookup)
    assert set(out.keys()) == {"doc_1", "doc_2"}
    assert "repo rate to 6.5 percent" in out["doc_1"].lower()
    assert "inflation moderated to 5.1 percent" in out["doc_2"].lower()


@pytest.mark.unit
def test_build_cite_quote_map_handles_empty_inputs():
    assert build_cite_quote_map("", {}, {}) == {}
    assert build_cite_quote_map("no citations here.", {}, {"doc_1": CHUNK}) == {}


@pytest.mark.unit
def test_build_cite_quote_map_skips_missing_snippets():
    answer = "Claim sentence about something. [doc_1]"
    doc_map = {"doc_1": ("T", "https://u", "u")}
    out = build_cite_quote_map(answer, doc_map, {})  # no snippets
    assert out == {}


@pytest.mark.unit
def test_extract_anchor_quote_fallback_on_short_paraphrase_with_no_overlap():
    """Very short paraphrased claim with zero token overlap should still
    yield a non-empty quote (deterministic head-of-chunk fallback) so the
    UI popover renders SOMETHING grounded in the source."""
    claim = "Yes."  # no content tokens at all after stopword/length filter
    quote = _extract_anchor_quote(claim, CHUNK, max_chars=120)
    assert quote is not None
    assert len(quote) > 0
    # Fallback prefers sentence-boundary cut at the leading sentence.
    assert quote.lower().startswith("the reserve bank of india")


@pytest.mark.unit
def test_extract_anchor_quote_fallback_truncates_cleanly_when_no_sentence_split():
    """A chunk with no sentence terminators should still return a clean,
    bounded prefix (whitespace-cut, ellipsis appended on truncation)."""
    chunk = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
    claim = "completely unrelated narwhal kingdom"
    quote = _extract_anchor_quote(claim, chunk, max_chars=30)
    assert quote is not None
    # Truncated at last space within first 30 chars, ellipsis appended.
    assert quote.endswith("…")
    assert len(quote) <= 31
