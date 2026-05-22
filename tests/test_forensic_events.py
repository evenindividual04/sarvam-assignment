"""Forensic-differentiation backend events (Track A).

Covers the four event-type discriminators added to the orchestrator stream:

- ``hop_evidence``        — mechanical grounded/open ledger per hop
- ``source_contribution`` — token-share per URL of the final context
- ``source_role``         — LLM-classified role per URL (graceful degradation)
- ``terminator``          — explicit hop-loop stop reason, one per run

The grounded/open computation and token-share helpers are tested as pure
units; the end-to-end emission count is asserted against a fully-mocked
orchestrator run reusing the adaptive-hop test scaffold.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import uuid

import pytest

_TMP_DB_DIR = tempfile.mkdtemp(prefix="forensic_events_test_db_")
os.environ["DB_PATH"] = os.path.join(_TMP_DB_DIR, "research.db")

from agent import orchestrator as orch_mod  # noqa: E402
from agent import source_role as source_role_mod  # noqa: E402
from agent.context_engine import compute_token_contribution  # noqa: E402
from agent.memory import init_db  # noqa: E402
from agent.models import (  # noqa: E402
    EVT_HOP_EVIDENCE,
    EVT_SOURCE_CONTRIBUTION,
    EVT_SOURCE_ROLE,
    EVT_TERMINATOR,
    ContextSnippet,
    PlannerOutput,
    QueryIntent,
    TypedQuery,
)
from agent.orchestrator import _compute_hop_evidence  # noqa: E402
from agent.source_role import classify_source_roles  # noqa: E402

# Reuse the heavy-mock scaffold.
from tests.test_adaptive_hop import (  # noqa: E402
    _GateProbe,
    _install_common_mocks,
    _patch_plan,
)


@pytest.fixture(autouse=True)
def _init_test_db():
    asyncio.run(init_db())


@pytest.fixture(autouse=True)
def _clear_source_role_cache():
    source_role_mod._clear_cache()
    yield
    source_role_mod._clear_cache()


# ── Helpers ──────────────────────────────────────────────────────────────


def _mk_snippet(
    *,
    doc_id: str,
    url: str,
    text: str,
    tokens: int = 50,
    domain: str = "example.com",
    title: str = "T",
    final_score: float = 0.5,
) -> ContextSnippet:
    return ContextSnippet(
        doc_id=doc_id,
        url=url,
        title=title,
        domain=domain,
        text=text,
        snippet=text[:80],
        token_count=tokens,
        retrieved_at="2026-05-01T00:00:00+00:00",
        final_score=final_score,
    )


# ── Unit tests: _compute_hop_evidence ────────────────────────────────────


def test_hop_evidence_grounds_entity_from_criterion():
    """A criterion-token appearing in a chunk → grounded row with real doc_id."""
    snippets = [
        _mk_snippet(
            doc_id="doc_1",
            url="https://nasa.gov/artemis",
            text=(
                "Artemis II crew completed final integration tests in 2025 "
                "ahead of the lunar flyby mission. NASA confirmed the launch."
            ),
            final_score=0.7,
        ),
        _mk_snippet(
            doc_id="doc_2",
            url="https://example.com/other",
            text="Unrelated snippet about culinary practices.",
        ),
    ]
    criteria = ["Identify the Artemis launch year"]
    grounded, open_rows = _compute_hop_evidence(snippets, criteria)

    # Artemis token grounded against doc_1.
    artemis_rows = [g for g in grounded if g["token"].lower() == "artemis"]
    assert artemis_rows, f"expected 'Artemis' grounded, got {grounded}"
    assert artemis_rows[0]["doc_id"] == "doc_1"
    assert artemis_rows[0]["url"] == "https://nasa.gov/artemis"
    assert "Artemis" in (artemis_rows[0]["quote"] or "")
    assert len(artemis_rows[0]["quote"]) <= 200

    # The criterion was satisfied, so it should NOT appear in `open`.
    assert all(o["criterion"] != criteria[0] for o in open_rows)


def test_hop_evidence_open_criteria_set_difference():
    """Criterion whose tokens don't appear → open row with reason=no_evidence."""
    snippets = [
        _mk_snippet(
            doc_id="doc_1",
            url="https://nasa.gov/artemis",
            text="Artemis program update content here.",
        ),
    ]
    criteria = [
        "Identify the Artemis launch year",
        "Compare Boeing Starliner regulatory approvals",
    ]
    grounded, open_rows = _compute_hop_evidence(snippets, criteria)

    starliner = [o for o in open_rows if "Starliner" in o["criterion"]]
    assert starliner, f"expected Starliner criterion ungrounded, got {open_rows}"
    assert starliner[0]["reason"] == "no_evidence"


def test_hop_evidence_partial_reason_when_score_low():
    """Token matches but BM25 < 0.3 → criterion reported as 'partial'."""
    snippets = [
        _mk_snippet(
            doc_id="doc_1",
            url="https://example.com/a",
            text="Artemis brief mention only.",
            final_score=0.1,
        ),
    ]
    criteria = ["verify the Artemis status"]
    grounded, open_rows = _compute_hop_evidence(snippets, criteria)

    # Token is grounded but the criterion also lands in `open` as partial.
    assert any(g["token"].lower() == "artemis" for g in grounded)
    partial = [o for o in open_rows if o["reason"] == "partial"]
    assert partial, f"expected partial open row, got {open_rows}"


def test_hop_evidence_caps_grounded_at_eight():
    """Even with many matches, grounded is capped at 8 rows."""
    text = " ".join(f"Entity{i}" for i in range(20))
    snippets = [_mk_snippet(doc_id="d1", url="https://x.com/a", text=text)]
    criteria = [f"Track Entity{i} status" for i in range(20)]
    grounded, _ = _compute_hop_evidence(snippets, criteria)
    assert len(grounded) <= 8


# ── Unit tests: compute_token_contribution ───────────────────────────────


def test_source_contribution_shares_sum_to_one():
    snippets = [
        _mk_snippet(doc_id="d1", url="https://a.com/1", text="x", tokens=300),
        _mk_snippet(doc_id="d2", url="https://b.com/2", text="x", tokens=100),
        _mk_snippet(doc_id="d3", url="https://a.com/1", text="x", tokens=200),
    ]
    contribs, total = compute_token_contribution(snippets)
    assert total == 600
    # Two unique URLs; a.com/1 should aggregate 500 (300+200), b.com 100.
    by_url = {c["url"]: c for c in contribs}
    assert by_url["https://a.com/1"]["tokens"] == 500
    assert by_url["https://b.com/2"]["tokens"] == 100
    share_sum = sum(c["share"] for c in contribs)
    assert abs(share_sum - 1.0) < 0.01


def test_source_contribution_citations_zero_without_answer():
    snippets = [_mk_snippet(doc_id="d1", url="https://a.com", text="x", tokens=10)]
    contribs, _ = compute_token_contribution(snippets)
    assert contribs[0]["citations"] == 0


def test_source_contribution_citations_from_answer_markers():
    snippets = [
        _mk_snippet(doc_id="doc_1", url="https://a.com", text="x", tokens=10),
        _mk_snippet(doc_id="doc_2", url="https://b.com", text="x", tokens=10),
    ]
    answer = "Some claim [doc_1]. Another claim [doc_1] [doc_2]."
    contribs, _ = compute_token_contribution(snippets, answer=answer)
    by_url = {c["url"]: c for c in contribs}
    assert by_url["https://a.com"]["citations"] == 1
    assert by_url["https://b.com"]["citations"] == 1


def test_source_contribution_empty_input():
    contribs, total = compute_token_contribution([])
    assert contribs == []
    assert total == 0


# ── Unit tests: source_role classifier ───────────────────────────────────


def test_source_role_classifier_degrades_to_unclassified_on_error(monkeypatch):
    """Provider raising → every input URL gets unclassified/0.0."""
    snippets = [
        _mk_snippet(doc_id="d1", url="https://a.com/1", text="x"),
        _mk_snippet(doc_id="d2", url="https://b.com/2", text="x"),
    ]

    async def _boom(_prompt, max_tokens=400):
        raise RuntimeError("provider down")

    monkeypatch.setattr("utils.provider_router.call_groq", _boom)

    result = asyncio.run(classify_source_roles(snippets))
    assert set(result.keys()) == {"https://a.com/1", "https://b.com/2"}
    for role, conf in result.values():
        assert role == "unclassified"
        assert conf == 0.0


def test_source_role_classifier_parses_valid_response(monkeypatch):
    snippets = [
        _mk_snippet(doc_id="d1", url="https://gov.example/a", text="official text"),
        _mk_snippet(doc_id="d2", url="https://news.example/b", text="news text"),
    ]

    async def _ok(_prompt, max_tokens=400):
        return (
            '{"roles": ['
            '{"url": "https://gov.example/a", "role": "official", "confidence": 0.9},'
            '{"url": "https://news.example/b", "role": "news_event", "confidence": 0.7}'
            ']}'
        )

    monkeypatch.setattr("utils.provider_router.call_groq", _ok)

    result = asyncio.run(classify_source_roles(snippets))
    assert result["https://gov.example/a"] == ("official", 0.9)
    assert result["https://news.example/b"] == ("news_event", 0.7)


def test_source_role_classifier_caches_results(monkeypatch):
    snippets = [_mk_snippet(doc_id="d1", url="https://cache.test/a", text="x")]
    call_count = {"n": 0}

    async def _track(_prompt, max_tokens=400):
        call_count["n"] += 1
        return '{"roles": [{"url": "https://cache.test/a", "role": "official", "confidence": 0.8}]}'

    monkeypatch.setattr("utils.provider_router.call_groq", _track)

    asyncio.run(classify_source_roles(snippets))
    asyncio.run(classify_source_roles(snippets))  # second call → cache hit
    assert call_count["n"] == 1


# ── End-to-end: emission counts over a full orchestrator run ─────────────


async def _collect_events(query: str = "Artemis launch year") -> list:
    orch = orch_mod.ResearchOrchestrator()
    events = []
    async for ev in orch.run(query=query, session_id="s-" + uuid.uuid4().hex):
        events.append(ev)
    await orch.aclose()
    return events


def _events_of(events: list, etype: str) -> list:
    return [e for e in events if getattr(e, "event_type", None) == etype]


def _stub_source_role(monkeypatch):
    async def _noop(_chunks):
        return {}
    monkeypatch.setattr(
        "agent.source_role.classify_source_roles", _noop,
    )


def test_terminator_event_emitted_once_per_run(monkeypatch):
    """One full run → exactly one terminator event."""
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[4000])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="high",  # blocks hop-2
                success_criteria=["Verify Artemis launch year"],
                queries=[
                    TypedQuery(
                        text="q1",
                        intent=QueryIntent.PRIMARY,
                        rationale="probe canonical sources",
                    ),
                ],
            ),
        ],
    )
    _stub_source_role(monkeypatch)

    events = asyncio.run(_collect_events())
    terminators = _events_of(events, EVT_TERMINATOR)

    assert len(terminators) == 1, f"expected 1 terminator, got {len(terminators)}"
    payload = terminators[0].data
    assert payload["reason"] in {
        "MAX_HOPS_REACHED",
        "EVIDENCE_SUFFICIENT",
        "MARGINAL_GAIN_LOW",
        "NO_NEW_QUERIES",
        "BUDGET_EXHAUSTED",
        "CRITERIA_SATISFIED",
    }
    assert isinstance(payload["hop"], int) and payload["hop"] >= 1


def test_source_contribution_and_role_emitted_once_per_run(monkeypatch):
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[4000])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="high",
                queries=[
                    TypedQuery(
                        text="q1",
                        intent=QueryIntent.PRIMARY,
                        rationale="probe sources",
                    ),
                ],
            ),
        ],
    )
    _stub_source_role(monkeypatch)

    events = asyncio.run(_collect_events())
    contributions = _events_of(events, EVT_SOURCE_CONTRIBUTION)
    roles = _events_of(events, EVT_SOURCE_ROLE)

    assert len(contributions) == 1
    assert len(roles) == 1
    payload = contributions[0].data
    assert "contributions" in payload
    assert "total_tokens" in payload


def test_hop_evidence_emitted_per_hop(monkeypatch):
    """Single hop → exactly one hop_evidence event."""
    probe = _GateProbe()
    _install_common_mocks(monkeypatch, probe, selected_tokens_per_hop=[4000])
    _patch_plan(
        monkeypatch,
        probe,
        plan_outputs=[
            PlannerOutput(
                strategy="hop1",
                confidence="high",
                success_criteria=["Confirm Artemis launch"],
                queries=[
                    TypedQuery(text="q1", intent=QueryIntent.PRIMARY, rationale="r"),
                ],
            ),
        ],
    )
    _stub_source_role(monkeypatch)

    events = asyncio.run(_collect_events())
    hop_evidence = _events_of(events, EVT_HOP_EVIDENCE)
    assert len(hop_evidence) == 1
    payload = hop_evidence[0].data
    assert payload["hop"] == 1
    assert isinstance(payload["grounded"], list)
    assert isinstance(payload["open"], list)
