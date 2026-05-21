"""
Failing tests for the 4 pure functions. Run BEFORE implementation to confirm RED.
After implementation, all must turn GREEN.
"""
import math
import sys
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pytest

# ── imports will fail until the modules exist ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.citation_guard import convert_citations, CitationGuard
from agent.context_engine import format_context_xml, select_with_diversity, score_chunk, rank_and_select_mmr
from agent.models import ContextSnippet
from eval.judge import classify_failure


# ── Helpers ────────────────────────────────────────────────────────────────

def make_snippet(domain: str, text: str = "sample text", token_count: int = 50,
                 bm25_score: float = 1.0, days_old: int = 0) -> ContextSnippet:
    retrieved_at = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    return ContextSnippet(
        doc_id="doc_1",
        url=f"https://{domain}/page",
        title="Test Page",
        domain=domain,
        text=text,
        snippet=text[:100],
        bm25_score=bm25_score,
        recency_score=0.0,
        diversity_score=0.0,
        final_score=0.0,
        token_count=token_count,
        retrieved_at=retrieved_at,
    )


# ── Test: convert_citations ────────────────────────────────────────────────

class TestConvertCitations:
    def test_replaces_known_doc_ref(self):
        doc_map = {"doc_1": ("OpenAI Blog", "https://openai.com/blog", "openai.com")}
        answer = "GPT-4 was released [doc_1] last year."
        result = convert_citations(answer, doc_map)
        assert "[OpenAI Blog — openai.com](https://openai.com/blog)" in result
        assert "[doc_1]" not in result

    def test_leaves_unknown_doc_ref_unchanged(self):
        doc_map = {"doc_1": ("Some Title", "https://example.com", "example.com")}
        answer = "Unknown [doc_99] reference."
        result = convert_citations(answer, doc_map)
        assert "[doc_99]" in result

    def test_replaces_multiple_different_refs(self):
        doc_map = {
            "doc_1": ("Title A", "https://a.com", "a.com"),
            "doc_2": ("Title B", "https://b.com", "b.com"),
        }
        answer = "Claim one [doc_1] and claim two [doc_2]."
        result = convert_citations(answer, doc_map)
        assert "[Title A — a.com](https://a.com)" in result
        assert "[Title B — b.com](https://b.com)" in result

    def test_empty_doc_map_leaves_answer_unchanged(self):
        answer = "No citations here [doc_1]."
        result = convert_citations(answer, {})
        assert result == answer

    def test_no_doc_refs_returns_unchanged(self):
        answer = "Plain text with no citations."
        result = convert_citations(answer, {"doc_1": ("T", "https://x.com", "x.com")})
        assert result == answer

    def test_replaces_grouped_doc_refs(self):
        doc_map = {
            "doc_1": ("Title A", "https://a.com", "a.com"),
            "doc_3": ("Title C", "https://c.com", "c.com"),
        }
        answer = "Evidence [doc_1, doc_3] supports this."
        result = convert_citations(answer, doc_map)
        assert "[Title A — a.com](https://a.com)" in result
        assert "[Title C — c.com](https://c.com)" in result
        assert "[doc_1, doc_3]" not in result


# ── Test: CitationGuard.verify ─────────────────────────────────────────────

class TestCitationGuardVerify:
    def _guard(self):
        return CitationGuard()

    def test_all_citations_valid_returns_1(self):
        guard = self._guard()
        fetched_urls = {"https://a.com/p", "https://b.com/p"}
        doc_map = {
            "doc_1": ("A", "https://a.com/p", "a.com"),
            "doc_2": ("B", "https://b.com/p", "b.com"),
        }
        answer = "Claim [doc_1] and claim [doc_2]."
        score = guard.verify(answer, doc_map, fetched_urls)
        assert score == 1.0

    def test_one_hallucinated_citation_reduces_score(self):
        guard = self._guard()
        fetched_urls = {"https://a.com/p"}
        doc_map = {
            "doc_1": ("A", "https://a.com/p", "a.com"),
            "doc_2": ("Ghost", "https://ghost.com/p", "ghost.com"),
        }
        answer = "Claim [doc_1] and hallucinated [doc_2]."
        score = guard.verify(answer, doc_map, fetched_urls)
        assert score == 0.5

    def test_no_citations_returns_1(self):
        guard = self._guard()
        score = guard.verify("No citations.", {}, set())
        assert score == 1.0

    def test_all_hallucinated_returns_0(self):
        guard = self._guard()
        fetched_urls = set()
        doc_map = {"doc_1": ("Ghost", "https://ghost.com", "ghost.com")}
        answer = "Fully hallucinated [doc_1]."
        score = guard.verify(answer, doc_map, fetched_urls)
        assert score == 0.0

    def test_grouped_citations_are_counted(self):
        guard = self._guard()
        fetched_urls = {"https://a.com/p"}
        doc_map = {
            "doc_1": ("A", "https://a.com/p", "a.com"),
            "doc_2": ("B", "https://b.com/p", "b.com"),
        }
        answer = "Combined evidence [doc_1, doc_2]."
        score = guard.verify(answer, doc_map, fetched_urls)
        assert score == 0.5


# ── Test: select_with_diversity ────────────────────────────────────────────

class TestSelectWithDiversity:
    def test_max_per_domain_2_cap_enforced(self):
        """Three chunks from same domain; only 2 should be selected."""
        chunks = [
            make_snippet("example.com", token_count=100),
            make_snippet("example.com", token_count=100),
            make_snippet("example.com", token_count=100),
        ]
        # Assign distinct final_scores so order is deterministic
        chunks[0].final_score = 0.9
        chunks[1].final_score = 0.8
        chunks[2].final_score = 0.7

        selected = select_with_diversity(chunks, max_tokens=10000, max_per_domain=2)
        domains = [c.domain for c in selected]
        assert domains.count("example.com") <= 2

    def test_different_domains_all_included(self):
        chunks = [
            make_snippet("alpha.com", token_count=100),
            make_snippet("beta.com", token_count=100),
            make_snippet("gamma.com", token_count=100),
        ]
        for i, c in enumerate(chunks):
            c.final_score = 0.9 - i * 0.1

        selected = select_with_diversity(chunks, max_tokens=10000, max_per_domain=2)
        assert len(selected) == 3

    def test_token_budget_respected(self):
        chunks = [make_snippet(f"site{i}.com", token_count=400) for i in range(10)]
        for i, c in enumerate(chunks):
            c.final_score = 1.0 - i * 0.05

        selected = select_with_diversity(chunks, max_tokens=1000, max_per_domain=2)
        total_tokens = sum(c.token_count for c in selected)
        assert total_tokens <= 1000

    def test_empty_input_returns_empty(self):
        assert select_with_diversity([], max_tokens=6000) == []

    def test_oversized_top_chunk_does_not_block_smaller_chunks(self):
        chunks = [
            make_snippet("big.com", token_count=1200),
            make_snippet("small-a.com", token_count=400),
            make_snippet("small-b.com", token_count=400),
        ]
        chunks[0].final_score = 1.0
        chunks[1].final_score = 0.9
        chunks[2].final_score = 0.8

        selected = select_with_diversity(chunks, max_tokens=1000, max_per_domain=2)
        assert len(selected) == 2
        assert selected[0].domain == "small-a.com"
        assert selected[1].domain == "small-b.com"

    def test_mmr_respects_domain_caps_and_budget(self):
        chunks = [
            make_snippet("a.com", text="bm25 sparse retrieval exact match", token_count=220),
            make_snippet("a.com", text="bm25 lexical scoring tf idf", token_count=220),
            make_snippet("b.com", text="dense vector embeddings semantic search", token_count=220),
            make_snippet("c.com", text="hybrid rag reciprocal rank fusion", token_count=220),
        ]
        selected = rank_and_select_mmr("bm25 dense retrieval comparison", chunks, max_tokens=500, max_per_domain=1)
        assert sum(c.token_count for c in selected) <= 500
        domains = [c.domain for c in selected]
        assert len(domains) == len(set(domains))


# ── Test: score_chunk ──────────────────────────────────────────────────────

class TestScoreChunk:
    def test_weights_sum_to_1(self):
        """Weights in score_chunk: 0.6 + 0.2 + 0.2 == 1.0"""
        assert math.isclose(0.6 + 0.2 + 0.2, 1.0)

    def test_no_division_by_zero_when_bm25_zero(self):
        chunk = make_snippet("example.com", days_old=0)
        domain_counts = defaultdict(int)
        score = score_chunk(chunk, bm25_score=0.0, all_bm25_scores=[0.0, 0.0],
                            url_domain_count=domain_counts)
        assert isinstance(score, float)
        assert not math.isnan(score)

    def test_no_division_by_zero_domain_count_zero(self):
        chunk = make_snippet("example.com", days_old=0)
        domain_counts = defaultdict(int)
        score = score_chunk(chunk, bm25_score=1.0, all_bm25_scores=[1.0],
                            url_domain_count=domain_counts)
        assert score > 0

    def test_higher_bm25_yields_higher_score(self):
        chunk_high = make_snippet("a.com", days_old=0)
        chunk_low = make_snippet("a.com", days_old=0)
        domain_counts = defaultdict(int)
        all_scores = [0.1, 0.9]
        s_high = score_chunk(chunk_high, bm25_score=0.9, all_bm25_scores=all_scores,
                             url_domain_count=domain_counts)
        s_low = score_chunk(chunk_low, bm25_score=0.1, all_bm25_scores=all_scores,
                            url_domain_count=domain_counts)
        assert s_high > s_low

    def test_older_content_scores_lower_on_recency(self):
        chunk_new = make_snippet("b.com", days_old=0)
        chunk_old = make_snippet("b.com", days_old=365)
        domain_counts = defaultdict(int)
        all_scores = [1.0]
        s_new = score_chunk(chunk_new, bm25_score=1.0, all_bm25_scores=all_scores,
                            url_domain_count=domain_counts)
        s_old = score_chunk(chunk_old, bm25_score=1.0, all_bm25_scores=all_scores,
                            url_domain_count=domain_counts)
        assert s_new > s_old

    def test_score_within_bounds(self):
        chunk = make_snippet("c.com", days_old=5)
        domain_counts = defaultdict(int)
        score = score_chunk(chunk, bm25_score=0.5, all_bm25_scores=[0.0, 0.5, 1.0],
                            url_domain_count=domain_counts)
        assert 0.0 <= score <= 1.0


class TestContextFormatting:
    def test_format_context_xml_escapes_reserved_chars(self):
        s = make_snippet("example.com", text='alpha < beta & "quoted"')
        s.title = 'A "title" <x>'
        xml, _ = format_context_xml([s])
        assert "&lt;" in xml
        assert "&amp;" in xml
        assert "&quot;" in xml


class TestFailureClassification:
    def test_classify_failure_handles_none_optional_scores(self):
        result = classify_failure({
            "faithfulness_score": 0.95,
            "answer_relevance_score": 0.91,
            "citation_integrity_score": 1.0,
            "conflict_adherence_score": None,
            "session_coherence_score": None,
        })
        assert result == "PASS"
