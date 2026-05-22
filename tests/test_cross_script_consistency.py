"""Tests for C4 cross-script consistency metric.

Pure-Python pair-level aggregator over eval result rows grouped by
``concept_id``. No LLM call, no DB — just set arithmetic over cited URLs
and extracted entities.
"""
from __future__ import annotations

import pytest

from eval.judge import (
    _normalize_url,
    compute_cross_script_consistency,
)


def _row(qid: str, lang: str, concept_id: str, urls: list[str], entities: list[str]) -> dict:
    return {
        "question_id": qid,
        "language": lang,
        "concept_id": concept_id,
        "cited_urls": urls,
        "entities": entities,
    }


def test_perfect_overlap_returns_one():
    """Same URLs and entities in both languages → Jaccard = 1.0."""
    rows = [
        _row("F-4", "en", "capital-of-india",
             ["https://en.wikipedia.org/wiki/New_Delhi"],
             ["new delhi"]),
        _row("HI-7", "hi", "capital-of-india",
             ["https://en.wikipedia.org/wiki/New_Delhi"],
             ["new delhi"]),
    ]
    out = compute_cross_script_consistency(rows)
    assert out["n_pairs"] == 1
    assert out["n_topics"] == 1
    assert out["mean_citation_overlap"] == pytest.approx(1.0)
    assert out["mean_entity_overlap"] == pytest.approx(1.0)
    assert out["pairs"][0]["lang_a"] == "en"
    assert out["pairs"][0]["lang_b"] == "hi"


def test_zero_overlap_returns_zero():
    """Disjoint URL + entity sets → Jaccard = 0.0."""
    rows = [
        _row("F-5", "en", "current-pm-of-india",
             ["https://example.com/a"],
             ["alice"]),
        _row("HI-8", "hi", "current-pm-of-india",
             ["https://different.org/b"],
             ["bob"]),
    ]
    out = compute_cross_script_consistency(rows)
    assert out["n_pairs"] == 1
    assert out["mean_citation_overlap"] == pytest.approx(0.0)
    assert out["mean_entity_overlap"] == pytest.approx(0.0)


def test_partial_overlap_matches_expected_jaccard():
    """|A ∩ B| / |A ∪ B|: 1 shared URL out of 3 unique → 1/3.
    Domain normalization: www. prefix + trailing slash + query string
    are all stripped, so the two URLs collapse to the same normalized form."""
    rows = [
        _row("F-6", "en", "finance-minister-of-india",
             [
                 "https://www.rbi.org.in/announcement/",   # normalizes
                 "https://example.com/x",
             ],
             ["nirmala sitharaman", "rbi"]),
        _row("HI-2", "hi", "finance-minister-of-india",
             [
                 "https://rbi.org.in/announcement?utm=foo",  # same after norm
                 "https://other.com/y",
             ],
             ["nirmala sitharaman", "delhi"]),
    ]
    out = compute_cross_script_consistency(rows)
    # URLs: {rbi.org.in/announcement, example.com/x} ∩ {rbi.org.in/announcement, other.com/y}
    #     = {rbi.org.in/announcement}; union has 3 → 1/3.
    assert out["pairs"][0]["citation_overlap"] == pytest.approx(1 / 3)
    # Entities: {nirmala sitharaman, rbi} ∩ {nirmala sitharaman, delhi}
    #         = {nirmala sitharaman}; union has 3 → 1/3.
    assert out["pairs"][0]["entity_overlap"] == pytest.approx(1 / 3)


def test_single_language_group_returns_na():
    """Topic with only one language present → no pair emitted; metric is N/A."""
    rows = [
        _row("F-4", "en", "capital-of-india",
             ["https://en.wikipedia.org/wiki/New_Delhi"],
             ["new delhi"]),
    ]
    out = compute_cross_script_consistency(rows)
    assert out["n_pairs"] == 0
    assert out["n_topics"] == 0
    assert out["mean_citation_overlap"] is None
    assert out["mean_entity_overlap"] is None
    assert out["per_topic"] == {}
    assert out["pairs"] == []


def test_three_languages_emit_three_pairs():
    """N languages on one topic → C(N,2) pairs."""
    rows = [
        _row("F-4", "en", "capital-of-india",
             ["https://en.wikipedia.org/wiki/New_Delhi"], ["new delhi"]),
        _row("HI-7", "hi", "capital-of-india",
             ["https://en.wikipedia.org/wiki/New_Delhi"], ["new delhi"]),
        _row("TA-1", "ta", "capital-of-india",
             ["https://en.wikipedia.org/wiki/New_Delhi"], ["new delhi"]),
    ]
    out = compute_cross_script_consistency(rows)
    assert out["n_pairs"] == 3  # C(3,2)
    assert out["n_topics"] == 1


def test_falls_back_to_answer_markdown_links_when_no_explicit_urls():
    """When ``cited_urls`` is absent, URLs are scraped from the agent_answer
    markdown links produced by citation_guard."""
    rows = [
        {
            "question_id": "F-4", "language": "en", "concept_id": "capital-of-india",
            "agent_answer": "Delhi is the capital [Wikipedia — en.wikipedia.org](https://en.wikipedia.org/wiki/New_Delhi).",
        },
        {
            "question_id": "HI-7", "language": "hi", "concept_id": "capital-of-india",
            "agent_answer": "नई दिल्ली राजधानी है [विकी](https://en.wikipedia.org/wiki/New_Delhi).",
        },
    ]
    out = compute_cross_script_consistency(rows)
    assert out["pairs"][0]["citation_overlap"] == pytest.approx(1.0)


def test_normalize_url_strips_www_query_and_fragment():
    assert _normalize_url("https://www.RBI.org.in/path/?utm=1#x") == "https://rbi.org.in/path"
    # Root-only slash is preserved (only trailing slashes on non-root paths strip).
    assert _normalize_url("http://Example.com/") == "http://example.com/"
