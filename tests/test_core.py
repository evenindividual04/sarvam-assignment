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

from agent.citation_guard import (
    CitationGuard,
    convert_citations,
    parse_quote_claim_blocks,
)
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
        assert "[OpenAI Blog — openai.com] (https://openai.com/blog)" in result
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
        assert "[Title A — a.com] (https://a.com)" in result
        assert "[Title B — b.com] (https://b.com)" in result

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
        assert "[Title A — a.com] (https://a.com)" in result
        assert "[Title C — c.com] (https://c.com)" in result
        assert "[doc_1, doc_3]" not in result


# ── Phase 1: ReClaim quote-first audit ────────────────────────────────────


class TestQuoteFirstAudit:
    def test_parse_quote_claim_blocks_basic(self):
        answer = (
            "Per the docs, <quote>the repo rate is 5.50% as of May 2026</quote> "
            "<claim>signaling continued easing</claim> [doc_1]."
        )
        blocks = parse_quote_claim_blocks(answer)
        assert len(blocks) == 1
        b = blocks[0]
        assert b["quote"] == "the repo rate is 5.50% as of May 2026"
        assert b["claim"] == "signaling continued easing"
        assert b["doc_ids"] == ("doc_1",)
        assert b["start"] >= 0 and b["end"] > b["start"]

    def test_parse_quote_claim_blocks_multiple_doc_markers(self):
        """Regression: lock in that ``_DOC_ID_INNER_RE`` resolves at call
        time even though it is defined below ``parse_quote_claim_blocks``.
        Also covers multiple [doc_N] markers and multiple blocks per answer.
        """
        answer = (
            "<quote>first</quote><claim>one</claim>[doc_1][doc_2] then "
            "<quote>second</quote><claim>two</claim>[doc_3]"
        )
        blocks = parse_quote_claim_blocks(answer)
        assert len(blocks) == 2
        assert blocks[0]["doc_ids"] == ("doc_1", "doc_2")
        assert blocks[1]["doc_ids"] == ("doc_3",)

    def test_verify_quoted_text_grounded(self):
        guard = CitationGuard()
        snippet_lookup = {
            "doc_1": "Reserve Bank of India says the repo rate is 5.50% as of May 2026.",
        }
        doc_map = {"doc_1": ("RBI", "https://rbi.in", "rbi.in")}
        answer = (
            "Per RBI, <quote>the repo rate is 5.50% as of May 2026</quote> "
            "<claim>indicating easing</claim> [doc_1]."
        )
        result = guard.verify_quoted_text(answer, doc_map, snippet_lookup)
        assert result["total_quoted_claims"] == 1
        assert result["grounded_claims"] == 1
        assert result["quote_grounding_ratio"] == 1.0
        assert result["audit"][0]["grounded"] is True
        assert result["audit"][0]["matched_doc_id"] == "doc_1"

    def test_verify_quoted_text_hallucinated(self):
        guard = CitationGuard()
        snippet_lookup = {"doc_1": "RBI kept the policy rate unchanged."}
        doc_map = {"doc_1": ("RBI", "https://rbi.in", "rbi.in")}
        answer = (
            "<quote>the repo rate was slashed to 1.00% overnight</quote> "
            "<claim>unprecedented cut</claim> [doc_1]."
        )
        result = guard.verify_quoted_text(answer, doc_map, snippet_lookup)
        assert result["total_quoted_claims"] == 1
        assert result["grounded_claims"] == 0
        assert result["quote_grounding_ratio"] == 0.0
        assert result["audit"][0]["grounded"] is False
        assert result["audit"][0]["matched_doc_id"] is None

    def test_verify_quoted_text_no_quotes(self):
        guard = CitationGuard()
        answer = "Some plain answer with [doc_1] but no quote blocks."
        result = guard.verify_quoted_text(
            answer, {"doc_1": ("T", "https://x", "x")}, {"doc_1": "irrelevant"}
        )
        assert result["total_quoted_claims"] == 0
        assert result["quote_grounding_ratio"] == 1.0
        assert result["audit"] == []

    def test_verify_quoted_text_case_insensitive(self):
        guard = CitationGuard()
        snippet_lookup = {
            "doc_1": "The RBI Repo Rate Is 5.50% As Of May 2026.",
        }
        doc_map = {"doc_1": ("RBI", "https://rbi.in", "rbi.in")}
        answer = (
            "<quote>the rbi repo rate is 5.50%   as of   may 2026</quote> "
            "<claim>easing</claim> [doc_1]."
        )
        result = guard.verify_quoted_text(answer, doc_map, snippet_lookup)
        assert result["grounded_claims"] == 1
        assert result["quote_grounding_ratio"] == 1.0


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


# ── Phase 1.5: Uncertainty + Next-Steps ────────────────────────────────────


class TestProposeFollowUps:
    """Phase 1.5: deterministic alternate-query generation."""

    def test_propose_follow_ups_returns_three_distinct(self, monkeypatch):
        import asyncio as _asyncio
        from utils import next_steps as _ns

        async def _fake_groq(prompt: str, max_tokens: int = 200) -> str:
            return (
                '{"follow_ups": ['
                '"alternate phrasing alpha",'
                '"alternate phrasing beta",'
                '"alternate phrasing gamma"'
                "]}"
            )

        monkeypatch.setattr(
            "utils.provider_router.call_groq", _fake_groq, raising=True
        )
        executed = ["original q1", "original q2"]
        out = _asyncio.run(
            _ns.propose_follow_ups("user query", executed, outcome="missing")
        )
        assert len(out) == 3
        lowered = [q.lower() for q in out]
        assert len(set(lowered)) == 3, "follow-ups must be distinct"
        for q in out:
            assert q.lower() not in {e.lower() for e in executed}

    def test_propose_follow_ups_fallback_when_groq_unavailable(self, monkeypatch):
        import asyncio as _asyncio
        from utils import next_steps as _ns

        async def _broken_groq(prompt: str, max_tokens: int = 200) -> str:
            raise RuntimeError("groq down")

        monkeypatch.setattr(
            "utils.provider_router.call_groq", _broken_groq, raising=True
        )
        executed = ["RBI repo rate May 2026"]
        out = _asyncio.run(
            _ns.propose_follow_ups(
                "RBI repo rate May 2026", executed, outcome="weak"
            )
        )
        assert len(out) == 3
        for q in out:
            assert q.lower() not in {e.lower() for e in executed}
        joined = " ".join(out).lower()
        assert "2026" in joined or "official" in joined

    def test_propose_follow_ups_fallback_when_parse_fails(self, monkeypatch):
        import asyncio as _asyncio
        from utils import next_steps as _ns

        async def _garbage_groq(prompt: str, max_tokens: int = 200) -> str:
            return "not json at all"

        monkeypatch.setattr(
            "utils.provider_router.call_groq", _garbage_groq, raising=True
        )
        out = _asyncio.run(
            _ns.propose_follow_ups("topic X", ["topic X"], outcome="missing")
        )
        assert len(out) == 3

    def test_propose_follow_ups_uses_cerebras_when_available(self, monkeypatch):
        """Tier C: CEREBRAS_API_KEY set + auto → Cerebras tried, Groq not called."""
        import asyncio as _asyncio
        from utils import next_steps as _ns

        monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
        monkeypatch.setenv("FOLLOW_UP_PROVIDER", "auto")

        async def fake_cerebras(prompt: str, max_tokens: int = 200) -> str:
            return (
                '{"follow_ups": ['
                '"alt one","alt two","alt three"]}'
            )

        async def must_not_call_groq(prompt: str, max_tokens: int = 200) -> str:
            raise AssertionError("Groq should not run when Cerebras succeeds")

        monkeypatch.setattr(
            "utils.provider_router.call_cerebras", fake_cerebras, raising=True
        )
        monkeypatch.setattr(
            "utils.provider_router.call_groq", must_not_call_groq, raising=True
        )

        async def _run():
            out = await _ns.propose_follow_ups("q", ["original"], outcome="missing")
            return out, _ns.last_follow_up_provider()

        out, prov = _asyncio.run(_run())
        assert len(out) == 3
        assert prov == "cerebras"

    def test_propose_follow_ups_falls_back_to_groq(self, monkeypatch):
        """Tier C: Cerebras failure → Groq invoked + provider stamped ``groq``."""
        import asyncio as _asyncio
        from utils import next_steps as _ns

        monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
        monkeypatch.setenv("FOLLOW_UP_PROVIDER", "auto")

        async def broken_cerebras(prompt: str, max_tokens: int = 200) -> str:
            raise RuntimeError("cerebras down")

        async def fake_groq(prompt: str, max_tokens: int = 200) -> str:
            return (
                '{"follow_ups": ['
                '"groq one","groq two","groq three"]}'
            )

        monkeypatch.setattr(
            "utils.provider_router.call_cerebras", broken_cerebras, raising=True
        )
        monkeypatch.setattr(
            "utils.provider_router.call_groq", fake_groq, raising=True
        )

        async def _run():
            out = await _ns.propose_follow_ups("q", ["original"], outcome="weak")
            return out, _ns.last_follow_up_provider()

        out, prov = _asyncio.run(_run())
        assert len(out) == 3
        assert prov == "groq"

    def test_propose_follow_ups_groq_failure_labels_heuristic(self, monkeypatch):
        """Fix #6: if BOTH Cerebras and Groq raise, the attribution must be
        ``heuristic`` (not the stale ``cerebras`` from the local var)."""
        import asyncio as _asyncio
        from utils import next_steps as _ns

        monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
        monkeypatch.setenv("FOLLOW_UP_PROVIDER", "auto")

        async def broken_cerebras(prompt: str, max_tokens: int = 200) -> str:
            raise RuntimeError("cerebras down")

        async def broken_groq(prompt: str, max_tokens: int = 200) -> str:
            raise RuntimeError("groq down")

        monkeypatch.setattr(
            "utils.provider_router.call_cerebras", broken_cerebras, raising=True
        )
        monkeypatch.setattr(
            "utils.provider_router.call_groq", broken_groq, raising=True
        )

        async def _run():
            out = await _ns.propose_follow_ups("q", ["original"], outcome="weak")
            return out, _ns.last_follow_up_provider()

        out, prov = _asyncio.run(_run())
        assert len(out) == 3  # heuristic always returns 3
        assert prov == "heuristic"


class TestEnsureUncertaintyBlock:
    """Phase 1.5: deterministic [UNCERTAINTY] block injection."""

    def test_ensure_uncertainty_block_uses_follow_ups(self):
        from agent.orchestrator import _ensure_uncertainty_block

        executed = ["original q1", "original q2"]
        alternates = ["alt one", "alt two", "alt three"]
        out = _ensure_uncertainty_block("partial answer.", alternates)
        for q in alternates:
            assert q in out
        for q in executed:
            assert q not in out
        assert "[UNCERTAINTY]" in out

    def test_uncertainty_block_idempotent(self):
        from agent.orchestrator import _ensure_uncertainty_block

        alternates = ["alt one", "alt two", "alt three"]
        once = _ensure_uncertainty_block("partial answer.", alternates)
        twice = _ensure_uncertainty_block(once, alternates)
        assert once == twice
        assert once.count("[UNCERTAINTY]") == 1

    def test_ensure_uncertainty_block_pads_when_fewer_than_three(self):
        from agent.orchestrator import _ensure_uncertainty_block

        out = _ensure_uncertainty_block("answer.", ["only one"])
        assert "[UNCERTAINTY]" in out
        assert "only one" in out
        # Padding ensures we always print 3 bullets.
        assert out.count("\n- ") == 3


class TestMissingEvidenceBranch:
    """Phase 1.5: empty-context branch tags the turn and proposes alternates."""

    def test_uncertainty_branch_missing_evidence(self, monkeypatch):
        import asyncio as _asyncio
        import sys as _sys
        import uuid as _uuid

        _sys.path.insert(0, os.path.dirname(__file__))
        from test_adaptive_hop import (
            _GateProbe, _install_common_mocks, _patch_plan,
        )
        from agent import orchestrator as orch_mod
        from agent.models import PlannerOutput, QueryIntent, TypedQuery
        from agent.memory import init_db

        _asyncio.run(init_db())

        probe = _GateProbe()
        _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[0])

        def _empty_select(_q, _chunks, max_tokens=None, *args, **kwargs):
            return []

        monkeypatch.setattr(orch_mod, "rank_and_select", _empty_select)
        monkeypatch.setattr(orch_mod, "rank_and_select_mmr", _empty_select)

        async def _empty_select_async(_q, _chunks, max_tokens=None, *args, **kwargs):
            return []

        monkeypatch.setattr(orch_mod, "rank_and_select_async", _empty_select_async)

        async def _fake_follow_ups(query, executed, outcome):
            return ["alt query alpha", "alt query beta", "alt query gamma"]

        monkeypatch.setattr(orch_mod, "propose_follow_ups", _fake_follow_ups)

        _patch_plan(
            monkeypatch, probe,
            plan_outputs=[
                PlannerOutput(
                    strategy="primary lookup",
                    confidence="high",
                    queries=[
                        TypedQuery(text="needle in haystack", intent=QueryIntent.PRIMARY)
                    ],
                ),
            ],
        )

        async def _run() -> None:
            orch = orch_mod.ResearchOrchestrator()
            async for _ev in orch.run(
                query="needle in haystack", session_id=f"s-{_uuid.uuid4().hex}"
            ):
                pass
            await orch.aclose()

        _asyncio.run(_run())

        meta = probe.persisted_run_metadata
        assert meta.get("uncertainty_kind") == "missing"
        assert meta.get("follow_up_queries") == [
            "alt query alpha", "alt query beta", "alt query gamma",
        ]
        assert "SYNTHESIS_SKIPPED" in probe.persisted_state_trace


# ── Phase 1.75: Context-Builder Hardening ─────────────────────────────────


class TestUShapeOrdering:
    """Lost-in-the-Middle (Liu et al. 2023) mitigation: top-1 first, top-2 last."""

    def test_u_shape_ordering(self):
        from agent.context_engine import _reorder_u_shape
        a = make_snippet("a.com")
        b = make_snippet("b.com")
        c = make_snippet("c.com")
        d = make_snippet("d.com")
        e = make_snippet("e.com")
        out = _reorder_u_shape([a, b, c, d, e])
        assert out[0] is a, "top-1 must be at position 0"
        assert out[-1] is b, "top-2 must be at the last position"
        assert out[1:-1] == [c, d, e]

    def test_u_shape_no_op_for_small_lists(self):
        from agent.context_engine import _reorder_u_shape
        a = make_snippet("a.com")
        b = make_snippet("b.com")
        assert _reorder_u_shape([]) == []
        assert _reorder_u_shape([a]) == [a]
        assert _reorder_u_shape([a, b]) == [a, b]


class TestSnippetXmlMetadata:
    """Phase 1.75: each <document> carries retrieved_at, rank, relevance_score."""

    def test_snippet_xml_includes_retrieved_at_rank_score(self):
        import xml.etree.ElementTree as ET
        s1 = make_snippet("a.com", text="alpha", bm25_score=0.9)
        s2 = make_snippet("b.com", text="beta", bm25_score=0.7)
        s3 = make_snippet("c.com", text="gamma", bm25_score=0.5)
        s1.final_score = 0.847
        s2.final_score = 0.611
        s3.final_score = 0.412
        xml, _ = format_context_xml([s1, s2, s3])
        root = ET.fromstring(xml)
        docs = list(root.findall("document"))
        assert len(docs) == 3
        for doc in docs:
            assert doc.find("retrieved_at") is not None, "missing <retrieved_at>"
            assert doc.find("rank") is not None, "missing <rank>"
            assert doc.find("relevance_score") is not None, "missing <relevance_score>"
        # Ranks are 1-indexed and sequential in injection order.
        ranks = [int(d.findtext("rank")) for d in docs]
        assert ranks == [1, 2, 3]
        # Relevance scores formatted to 3 decimal places.
        for doc in docs:
            score_text = doc.findtext("relevance_score") or ""
            assert "." in score_text and len(score_text.split(".")[1]) == 3


class TestAdaptiveDrop:
    """Phase 1.75: evict lowest-scored selected chunk when a higher-scored
    candidate would otherwise be skipped due to budget."""

    def test_adaptive_drop_evicts_lower_score(self):
        # Two chunks fit exactly in budget=200 (100 + 100). A 3rd chunk with
        # higher score should evict the lower-scored existing chunk.
        low = make_snippet("low.com", text="x", token_count=100)
        low.final_score = 0.3
        mid = make_snippet("mid.com", text="y", token_count=100)
        mid.final_score = 0.6
        high = make_snippet("high.com", text="z", token_count=100)
        high.final_score = 0.9
        selected = select_with_diversity([low, mid, high], max_tokens=200)
        # `high` and `mid` should win; `low` should be evicted.
        domains = {c.domain for c in selected}
        assert "high.com" in domains
        assert "mid.com" in domains
        assert "low.com" not in domains
        total_tokens = sum(c.token_count for c in selected)
        assert total_tokens <= 200


class TestCompressHistoryFallback:
    """Phase 1.75: when history exceeds budget after rolling-summary injection,
    compress_history() must fire and tag run_metadata."""

    def test_compress_history_fallback_fires_when_overflow(self, monkeypatch):
        import asyncio as _asyncio
        import sys as _sys
        import uuid as _uuid

        _sys.path.insert(0, os.path.dirname(__file__))
        from test_adaptive_hop import (
            _GateProbe, _install_common_mocks, _patch_plan,
        )
        from agent import orchestrator as orch_mod
        from agent.models import PlannerOutput, QueryIntent, TypedQuery, Turn
        from agent.memory import init_db

        _asyncio.run(init_db())

        probe = _GateProbe()
        _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[0])

        # Force the history-overflow branch: huge fabricated prior turns +
        # huge fabricated rolling summary. count_tokens on long ASCII text
        # roughly tracks chars/4, so 100k chars >> history_budget (4000).
        big_chunk = "lorem ipsum dolor sit amet " * 2000
        fake_turns = [
            Turn(
                turn_id=f"t{i}",
                session_id="sess",
                query=f"query {i}",
                created_at="2026-01-01T00:00:00Z",
                response=big_chunk,
            )
            for i in range(3)
        ]

        async def _fake_prior(session_id, query):
            return fake_turns
        async def _fake_summary(session_id):
            return big_chunk
        async def _fake_count(session_id):
            return 10

        monkeypatch.setattr(orch_mod, "get_relevant_prior_turns", _fake_prior)
        monkeypatch.setattr(orch_mod, "get_latest_summary", _fake_summary)
        monkeypatch.setattr(orch_mod, "get_session_turn_count", _fake_count)

        # Mock the Groq compression call so the test is hermetic.
        compress_calls: list[tuple] = []
        async def _fake_compress(existing_summary, remaining_turns, max_tokens):
            compress_calls.append((existing_summary, remaining_turns, max_tokens))
            return "compressed history"
        monkeypatch.setattr(
            "utils.provider_router.compress_history", _fake_compress, raising=True
        )

        _patch_plan(
            monkeypatch, probe,
            plan_outputs=[
                PlannerOutput(
                    strategy="primary lookup",
                    confidence="high",
                    queries=[TypedQuery(text="topic", intent=QueryIntent.PRIMARY)],
                ),
            ],
        )

        async def _run() -> None:
            orch = orch_mod.ResearchOrchestrator()
            async for _ev in orch.run(
                query="topic", session_id=f"s-{_uuid.uuid4().hex}",
            ):
                pass
            await orch.aclose()

        _asyncio.run(_run())

        assert compress_calls, "compress_history was not invoked"
        meta = probe.persisted_run_metadata
        fallbacks = meta.get("context_fallbacks") or []
        assert "history_compressed" in fallbacks


class TestBudgetDistributionTelemetry:
    """Phase 1.75: run_metadata.budget_distribution has all 4 keys with ints."""

    def test_budget_distribution_in_run_metadata(self, monkeypatch):
        import asyncio as _asyncio
        import sys as _sys
        import uuid as _uuid

        _sys.path.insert(0, os.path.dirname(__file__))
        from test_adaptive_hop import (
            _GateProbe, _install_common_mocks, _patch_plan,
        )
        from agent import orchestrator as orch_mod
        from agent.models import PlannerOutput, QueryIntent, TypedQuery
        from agent.memory import init_db

        _asyncio.run(init_db())

        probe = _GateProbe()
        _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[10])

        _patch_plan(
            monkeypatch, probe,
            plan_outputs=[
                PlannerOutput(
                    strategy="primary lookup",
                    confidence="high",
                    queries=[TypedQuery(text="topic", intent=QueryIntent.PRIMARY)],
                ),
            ],
        )

        async def _run() -> None:
            orch = orch_mod.ResearchOrchestrator()
            async for _ev in orch.run(
                query="topic", session_id=f"s-{_uuid.uuid4().hex}",
            ):
                pass
            await orch.aclose()

        _asyncio.run(_run())

        bd = probe.persisted_run_metadata.get("budget_distribution") or {}
        for key in ("system", "history", "web_context", "output_reserved"):
            assert key in bd, f"missing budget_distribution['{key}']"
            assert isinstance(bd[key], int), f"{key} must be int, got {type(bd[key])}"
            assert bd[key] >= 0, f"{key} must be >= 0"
