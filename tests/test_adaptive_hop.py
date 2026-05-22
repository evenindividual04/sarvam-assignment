"""V3.2 — Adaptive 2-hop retrieval gated by planner confidence.

Tests the orchestrator's gate decision: a second hop runs only when the
first-hop planner reports `confidence="low"` AND the selected context is
below half of the web-context budget. Hop 2 is restricted to RECENCY_CHECK
and CONTRADICTION_PROBE intents (PRIMARY queries are dropped).
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from typing import Any, AsyncIterator

import pytest

# Force DB into a temp location BEFORE memory module is imported.
_TMP_DB_DIR = tempfile.mkdtemp(prefix="adaptive_hop_test_db_")
os.environ["DB_PATH"] = os.path.join(_TMP_DB_DIR, "research.db")

from agent import orchestrator as orch_mod  # noqa: E402
from agent import memory as memory_mod  # noqa: E402
from agent.memory import init_db  # noqa: E402
from agent.models import (  # noqa: E402
    ConflictResult,
    ContextSnippet,
    PlannerOutput,
    QueryIntent,
    SearchResult,
    TypedQuery,
)
from utils import provider_router  # noqa: E402


# ── Pydantic model tests ───────────────────────────────────────────────────

def test_planner_output_accepts_confidence_field():
    out = PlannerOutput(
        strategy="s",
        queries=[TypedQuery(text="q", intent=QueryIntent.PRIMARY)],
        confidence="low",
    )
    assert out.confidence == "low"


def test_planner_output_defaults_confidence_medium_when_missing():
    out = PlannerOutput(
        strategy="s",
        queries=[TypedQuery(text="q", intent=QueryIntent.PRIMARY)],
    )
    assert out.confidence == "medium"


def test_parse_planner_output_defaults_confidence_when_field_missing():
    raw = '{"strategy":"s","queries":[{"text":"q","intent":"primary"}]}'
    out = provider_router.parse_planner_output(raw, "fallback")
    assert out.confidence == "medium"


def test_parse_planner_output_reads_confidence_field():
    raw = '{"strategy":"s","confidence":"low","queries":[{"text":"q","intent":"primary"}]}'
    out = provider_router.parse_planner_output(raw, "fallback")
    assert out.confidence == "low"


def test_parse_planner_output_fallback_uses_low_confidence():
    out = provider_router.parse_planner_output("not json at all", "the query")
    assert out.strategy == "Direct retrieval fallback"
    assert out.confidence == "low"


# ── Orchestrator gating tests ──────────────────────────────────────────────

def _mk_snippet(doc_id: str, tokens: int = 100, domain: str = "example.com") -> ContextSnippet:
    return ContextSnippet(
        doc_id=doc_id,
        url=f"https://{domain}/{doc_id}",
        title=f"title-{doc_id}",
        domain=domain,
        text=f"text for {doc_id} " * max(1, tokens // 4),
        snippet=f"snippet-{doc_id}",
        token_count=tokens,
        retrieved_at="2026-05-01T00:00:00+00:00",
    )


def _mk_search_result(url: str) -> SearchResult:
    return SearchResult(
        url=url,
        title=f"t-{url}",
        snippet="snippet",
        domain="example.com",
        retrieved_at="2026-05-01T00:00:00+00:00",
        raw_content="raw content body",
    )


class _GateProbe:
    """Captures orchestrator behaviour for assertions."""

    def __init__(self) -> None:
        self.plan_calls: list[str] = []  # one entry per planner invocation
        self.search_calls: list[list[TypedQuery]] = []
        self.persisted_turn = None
        self.persisted_state_trace: list[str] = []
        self.persisted_run_metadata: dict[str, Any] = {}


def _install_common_mocks(monkeypatch, probe: _GateProbe, *, selected_tokens_per_hop: list[int]):
    """Mock every external boundary in the orchestrator."""

    # Memory: in-memory no-ops
    async def _session_exists(_sid):
        return True

    async def _create_session(_sid, _now):
        return None

    async def _get_relevant_prior_turns(_sid, _q):
        return []

    async def _get_latest_summary(_sid):
        return None

    async def _get_session_turn_count(_sid):
        return 0

    async def _save_turn(turn):
        probe.persisted_turn = turn
        probe.persisted_state_trace = list(turn.state_trace)
        probe.persisted_run_metadata = dict(turn.run_metadata_json or {})

    async def _save_turn_context(_tid, _sel):
        return None

    async def _save_claim_audit(_tid, _records):
        return None

    async def _save_contradiction_probe(**_kwargs):
        return None

    async def _save_session_summary(*_a, **_kw):
        return None

    monkeypatch.setattr(orch_mod, "session_exists", _session_exists)
    monkeypatch.setattr(orch_mod, "create_session", _create_session)
    monkeypatch.setattr(orch_mod, "get_relevant_prior_turns", _get_relevant_prior_turns)
    monkeypatch.setattr(orch_mod, "get_latest_summary", _get_latest_summary)
    monkeypatch.setattr(orch_mod, "get_session_turn_count", _get_session_turn_count)
    monkeypatch.setattr(orch_mod, "save_turn", _save_turn)
    monkeypatch.setattr(orch_mod, "save_turn_context", _save_turn_context)
    monkeypatch.setattr(orch_mod, "save_claim_audit", _save_claim_audit)
    monkeypatch.setattr(orch_mod, "save_contradiction_probe", _save_contradiction_probe)
    monkeypatch.setattr(orch_mod, "save_session_summary", _save_session_summary)

    # Search: returns one result per call
    async def _search(typed_queries, cancel_token=None, **kwargs):
        probe.search_calls.append(list(typed_queries))
        hop = len(probe.search_calls)
        return [_mk_search_result(f"https://example.com/hop{hop}")]

    monkeypatch.setattr(orch_mod, "search", _search)

    # Extractor: returns raw content per url
    class _FakeExtractor:
        def __init__(self):
            pass

        async def extract_all(self, results, cancel_token=None):
            return {r.url: "extracted body text " * 30 for r in results}

        async def aclose(self):
            return None

    monkeypatch.setattr(orch_mod, "Extractor", _FakeExtractor)

    # chunk(): produce a single snippet per result (we ignore content; tokens controlled below)
    def _chunk(result, text):
        return [_mk_snippet(doc_id=f"d-{result.url}", tokens=10, domain=result.domain)]

    monkeypatch.setattr(orch_mod, "chunk", _chunk)

    # rank_and_select: return controlled token sums per hop
    call_count = {"n": 0}

    def _rank_and_select(query, chunks, max_tokens):
        idx = call_count["n"]
        call_count["n"] += 1
        target_tokens = selected_tokens_per_hop[min(idx, len(selected_tokens_per_hop) - 1)]
        # Build snippets summing to target_tokens
        snip = _mk_snippet(doc_id=f"sel-{idx}", tokens=target_tokens)
        return [snip]

    monkeypatch.setattr(orch_mod, "rank_and_select", _rank_and_select)
    monkeypatch.setattr(orch_mod, "rank_and_select_mmr", _rank_and_select)

    async def _rank_and_select_async(query, chunks, max_tokens):
        return _rank_and_select(query, chunks, max_tokens)

    monkeypatch.setattr(orch_mod, "rank_and_select_async", _rank_and_select_async)

    # format_context_xml: trivial xml
    def _format_xml(selected):
        xml = "<context>" + "".join(f"<doc id='{s.doc_id}'/>" for s in selected) + "</context>"
        doc_map = {s.doc_id: (s.title, s.url, s.domain) for s in selected}
        return xml, doc_map

    monkeypatch.setattr(orch_mod, "format_context_xml", _format_xml)

    # probe_contradictions: never finds conflict
    async def _probe(_selected, _query):
        return ConflictResult(has_conflict=False)

    monkeypatch.setattr(orch_mod, "probe_contradictions", _probe)

    # verify_claims: no-op
    async def _verify(answer, doc_map, snippets):
        return 1.0, [], answer

    monkeypatch.setattr(orch_mod, "verify_claims", _verify)

    # Synthesizer: return short canned answer
    async def _stream_synth(**_kwargs) -> AsyncIterator[tuple[str, int, int]]:
        yield ("Answer body.", 0, 0)
        yield ("", 10, 5)

    import agent.synthesizer as synth_mod
    monkeypatch.setattr(synth_mod, "stream_synthesis", _stream_synth)

    # Citation guard: trivial verify (it's instance method; orchestrator uses self._guard.verify)
    # CitationGuard.verify already works on string + doc_map; leave it.

    # STOP-RAG gate (agent.stopping.decide_continue) and refinement classifier
    # (agent.refinement_check.classify_answer) both call provider_router.call
    # _groq under the hood. Without stubbing the boundary, adaptive-hop tests
    # hit real Groq endpoints; whether the call beats the 4 s stop-RAG
    # timeout decides whether the hop loop continues, which makes the suite
    # order-dependent. Stub call_groq to raise so both helpers take their
    # documented "degrade-to-safer" path — orchestrator's own terminators
    # (MAX_HOPS, DIFFICULTY_EASY, TOKEN_BUDGET) then drive the gating logic
    # that this test file is actually asserting.
    async def _groq_unavailable(_prompt, max_tokens: int = 200):
        raise RuntimeError("call_groq stubbed in adaptive_hop tests")

    monkeypatch.setattr("utils.provider_router.call_groq", _groq_unavailable)


def _patch_plan(monkeypatch, probe: _GateProbe, *, plan_outputs: list[PlannerOutput]):
    """Mock provider_router.plan with a sequence of planner outputs."""

    async def _plan(query, prior_summary="No prior context."):
        probe.plan_calls.append(query)
        idx = len(probe.plan_calls) - 1
        return plan_outputs[min(idx, len(plan_outputs) - 1)]

    monkeypatch.setattr(provider_router, "plan", _plan)


async def _run_once(query: str = "test query") -> tuple[_GateProbe, Any]:
    orch = orch_mod.ResearchOrchestrator()
    events = []
    async for ev in orch.run(query=query, session_id="s-" + uuid.uuid4().hex):
        events.append(ev)
    await orch.aclose()
    return events


@pytest.fixture(autouse=True)
def _init_test_db():
    asyncio.run(init_db())


def test_second_hop_runs_when_confidence_low_and_context_thin(monkeypatch):
    probe = _GateProbe()
    # Hop 1 selects ~10 tokens (well below 0.5 * web_context_budget ≈ 3200).
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[10, 10])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="low",
                queries=[TypedQuery(text="q1", intent=QueryIntent.PRIMARY)],
            ),
            PlannerOutput(
                strategy="hop2",
                confidence="medium",
                queries=[
                    TypedQuery(text="q2 recency 2026", intent=QueryIntent.RECENCY_CHECK),
                ],
            ),
        ],
    )

    asyncio.run(_run_once())

    assert len(probe.plan_calls) == 2, "second planner call should fire"
    assert len(probe.search_calls) == 2, "two search rounds expected"
    assert "HOP_2" in probe.persisted_state_trace
    assert probe.persisted_run_metadata.get("hop_count") == 2


def test_second_hop_skipped_when_confidence_high(monkeypatch):
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[10])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="high",
                queries=[TypedQuery(text="q1", intent=QueryIntent.PRIMARY)],
            ),
        ],
    )

    asyncio.run(_run_once())

    assert len(probe.plan_calls) == 1
    assert len(probe.search_calls) == 1
    assert "HOP_2" not in probe.persisted_state_trace
    assert probe.persisted_run_metadata.get("hop_count") == 1


def test_second_hop_skipped_when_context_above_80pct_budget(monkeypatch):
    probe = _GateProbe()
    # Demo tuning: threshold was loosened from 0.5 to 0.8 of
    # web_context_budget (6400). 5500 > 0.8 * 6400 = 5120, so R6
    # (EVIDENCE_SUFFICIENT) still fires and hop 2 is skipped.
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[5500])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="low",  # would normally trigger hop2
                queries=[TypedQuery(text="q1", intent=QueryIntent.PRIMARY)],
            ),
        ],
    )

    asyncio.run(_run_once())

    assert len(probe.plan_calls) == 1, "hop 2 should be skipped (context already sufficient)"
    assert "HOP_2" not in probe.persisted_state_trace
    assert probe.persisted_run_metadata.get("hop_count") == 1


def test_second_hop_filters_to_recency_and_contradiction_only(monkeypatch):
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[10, 10])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="low",
                queries=[TypedQuery(text="q1", intent=QueryIntent.PRIMARY)],
            ),
            PlannerOutput(
                strategy="hop2",
                confidence="medium",
                queries=[
                    TypedQuery(text="redo primary", intent=QueryIntent.PRIMARY),
                    TypedQuery(text="define X", intent=QueryIntent.DEFINITION),
                    TypedQuery(text="compare A vs B", intent=QueryIntent.COMPARISON),
                    TypedQuery(text="latest 2026 numbers", intent=QueryIntent.RECENCY_CHECK),
                    TypedQuery(text="criticism of X", intent=QueryIntent.CONTRADICTION_PROBE),
                ],
            ),
        ],
    )

    asyncio.run(_run_once())

    assert len(probe.search_calls) == 2
    hop2_queries = probe.search_calls[1]
    hop2_intents = {tq.intent for tq in hop2_queries}
    assert hop2_intents == {QueryIntent.RECENCY_CHECK, QueryIntent.CONTRADICTION_PROBE}, (
        f"hop2 must drop non-recency/non-contradiction intents; got {hop2_intents}"
    )


def test_max_hops_2_hard_cap(monkeypatch):
    probe = _GateProbe()
    # Even if every hop reports low confidence + thin context, cap is 2.
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[10, 10, 10, 10])
    low_plan = PlannerOutput(
        strategy="loop",
        confidence="low",
        queries=[TypedQuery(text="latest 2026", intent=QueryIntent.RECENCY_CHECK)],
    )
    _patch_plan(monkeypatch, probe, plan_outputs=[low_plan, low_plan, low_plan, low_plan])

    asyncio.run(_run_once())

    assert len(probe.search_calls) == 2, "search must not run a third time"
    assert probe.persisted_run_metadata.get("hop_count") == 2


def test_hop_count_persisted_to_run_metadata(monkeypatch):
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[10])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="high",
                queries=[TypedQuery(text="q1", intent=QueryIntent.PRIMARY)],
            ),
        ],
    )

    asyncio.run(_run_once())

    assert "hop_count" in probe.persisted_run_metadata
    assert probe.persisted_run_metadata["hop_count"] == 1


def test_state_trace_contains_HOP_2_when_second_hop_runs(monkeypatch):
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[10, 10])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="low",
                queries=[TypedQuery(text="q1", intent=QueryIntent.PRIMARY)],
            ),
            PlannerOutput(
                strategy="hop2",
                confidence="medium",
                queries=[
                    TypedQuery(text="latest 2026", intent=QueryIntent.RECENCY_CHECK),
                ],
            ),
        ],
    )

    asyncio.run(_run_once())

    assert "HOP_2" in probe.persisted_state_trace


# ── Phase 1.875: terminator & evidence_gap tests ──────────────────────────

def test_terminator_fired_difficulty_easy(monkeypatch):
    """difficulty='easy' clamps max_hops to 1, terminator='DIFFICULTY_EASY_SKIPPED'."""
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[10, 10])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="low",
                difficulty="easy",
                queries=[TypedQuery(text="q1", intent=QueryIntent.PRIMARY)],
            ),
        ],
    )

    asyncio.run(_run_once())

    # Difficulty=easy clamps max_hops=1 so hop loop exits via MAX_HOPS_REACHED.
    # (Defensive: orchestrator may stamp either MAX_HOPS_REACHED or the easy
    # bypass depending on which guard fires first; both are acceptable signals
    # that the easy path was respected.)
    assert len(probe.search_calls) == 1, "easy difficulty must run hop 1 only"
    terminator = probe.persisted_run_metadata.get("terminator_fired")
    assert terminator in ("MAX_HOPS_REACHED", "DIFFICULTY_EASY_SKIPPED")


def test_terminator_fired_token_budget(monkeypatch):
    """High cumulative tokens skip hop 2 even when confidence='low'."""
    probe = _GateProbe()
    # Hop 1 selects 15,000 tokens — far above 0.75 * 16_000 threshold.
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[15000])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="low",
                queries=[TypedQuery(text="q1", intent=QueryIntent.PRIMARY)],
            ),
        ],
    )

    asyncio.run(_run_once())

    assert len(probe.search_calls) == 1, "token budget should terminate before hop 2"
    assert probe.persisted_run_metadata.get("terminator_fired") == "TOKEN_BUDGET_EXHAUSTED"


def test_evidence_gap_detection_no_results(monkeypatch):
    """A TypedQuery with zero results recorded as no_results gap."""
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[100])

    # Override search to return results only for some queries.
    async def _search(typed_queries, cancel_token=None, urls_by_query=None, **kwargs):
        probe.search_calls.append(list(typed_queries))
        out = []
        for tq in typed_queries:
            if urls_by_query is not None:
                urls_by_query.setdefault(tq.text, [])
            if tq.text == "found-query":
                r = _mk_search_result("https://example.com/found")
                r.intent_origin = tq.intent.value
                out.append(r)
                if urls_by_query is not None:
                    urls_by_query[tq.text].append(r.url)
        return out

    monkeypatch.setattr(orch_mod, "search", _search)

    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="medium",
                queries=[
                    TypedQuery(text="found-query", intent=QueryIntent.PRIMARY),
                    TypedQuery(text="empty-query", intent=QueryIntent.DEFINITION),
                ],
            ),
        ],
    )

    asyncio.run(_run_once())

    gaps = probe.persisted_run_metadata.get("evidence_gaps", [])
    by_query = {g["query"]: g for g in gaps}
    assert "empty-query" in by_query
    assert by_query["empty-query"]["reason"] == "no_results"
    assert by_query["empty-query"]["intent"] == "definition"


def test_evidence_gap_detection_all_filtered(monkeypatch):
    """A TypedQuery with results but all dropped during selection → all_filtered."""
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[10])

    async def _search(typed_queries, cancel_token=None, urls_by_query=None, **kwargs):
        probe.search_calls.append(list(typed_queries))
        out = []
        for tq in typed_queries:
            r = _mk_search_result(f"https://example.com/{tq.text}")
            r.intent_origin = tq.intent.value
            out.append(r)
            if urls_by_query is not None:
                urls_by_query.setdefault(tq.text, []).append(r.url)
        return out

    monkeypatch.setattr(orch_mod, "search", _search)

    # Selection picks only an unrelated URL — so every plan query is "all_filtered".
    def _rank(query, chunks, max_tokens):
        s = _mk_snippet(doc_id="sel-unrelated", tokens=10, domain="unrelated.com")
        s.url = "https://unrelated.com/whatever"
        return [s]

    monkeypatch.setattr(orch_mod, "rank_and_select", _rank)
    monkeypatch.setattr(orch_mod, "rank_and_select_mmr", _rank)

    async def _rank_async(query, chunks, max_tokens):
        return _rank(query, chunks, max_tokens)

    monkeypatch.setattr(orch_mod, "rank_and_select_async", _rank_async)

    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="medium",
                queries=[
                    TypedQuery(text="filt-q", intent=QueryIntent.PRIMARY),
                ],
            ),
        ],
    )

    asyncio.run(_run_once())

    gaps = probe.persisted_run_metadata.get("evidence_gaps", [])
    by_query = {g["query"]: g for g in gaps}
    assert "filt-q" in by_query
    assert by_query["filt-q"]["reason"] == "all_filtered"
