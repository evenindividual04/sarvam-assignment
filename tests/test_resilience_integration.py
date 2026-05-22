from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import httpx

from agent.models import QueryIntent, SearchResult, TypedQuery
from agent import search as search_mod
from utils import provider_router


def test_search_falls_back_when_parallel_fails(monkeypatch):
    async def fake_parallel(query, client, num_results=None):
        request = httpx.Request("POST", "https://api.parallel.ai/v1/search")
        response = httpx.Response(422, request=request)
        raise httpx.HTTPStatusError("parallel fail", request=request, response=response)

    async def fake_tavily(query, client):
        return [
            SearchResult(
                url="https://example.com/a",
                title="A",
                snippet="A",
                domain="example.com",
                retrieved_at="2026-01-01T00:00:00+00:00",
                raw_content="content",
            )
        ]

    async def fake_serper(query, client):
        return []

    monkeypatch.setattr(search_mod, "_search_parallel", fake_parallel)
    monkeypatch.setattr(search_mod, "_search_tavily", fake_tavily)
    monkeypatch.setattr(search_mod, "_search_serper", fake_serper)

    out = asyncio.run(search_mod.search([TypedQuery(text="x", intent=QueryIntent.PRIMARY)]))
    assert len(out) == 1
    assert out[0].url == "https://example.com/a"


def test_synthesize_falls_back_to_openrouter(monkeypatch):
    async def fake_gemini(prompt):
        raise Exception("503 Service Unavailable")
        yield  # pragma: no cover

    async def fake_openrouter(prompt):
        yield ("fallback answer [doc_1]", 0, 0)
        yield ("", 0, 0)

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    monkeypatch.setattr(provider_router, "_synthesize_gemini", fake_gemini)
    monkeypatch.setattr(provider_router, "_synthesize_openrouter", fake_openrouter)
    # V3.9: ensure providers between gemini and openrouter in the chain are
    # skipped (missing keys → pre-flight skip), so the test still asserts
    # "openrouter receives the fallback."
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)

    async def collect():
        chunks = []
        async for text, _, _ in provider_router.synthesize(
            query="q",
            context_xml="<context></context>",
            doc_map={"doc_1": ("t", "https://x.com", "x.com")},
        ):
            chunks.append(text)
        return "".join(chunks)

    text = asyncio.run(collect())
    assert "fallback answer" in text


def test_synthesize_yields_unavailable_when_chain_exhausted(monkeypatch):
    """V3.9: the new chain doesn't raise when all providers are unavailable —
    it yields a single user-facing error string. This is the contract change
    from the V2 behavior (which propagated the upstream exception)."""
    async def fake_gemini(prompt):
        raise Exception("503 Service Unavailable")
        yield  # pragma: no cover

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    # Force Ollama unreachable for the test (port 1 is closed everywhere).
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setattr(provider_router, "_synthesize_gemini", fake_gemini)

    async def collect():
        chunks = []
        async for text, _, _ in provider_router.synthesize(
            query="q",
            context_xml="<context></context>",
            doc_map={},
        ):
            chunks.append(text)
        return chunks

    chunks = asyncio.run(collect())
    assert len(chunks) == 1
    assert "unavailable" in chunks[0].lower()


def test_parse_planner_output_valid_json():
    raw = (
        '{"strategy":"Find official source then compare",'
        '"queries":[{"text":"q1","intent":"primary"},'
        '{"text":"q2","intent":"comparison"},'
        '{"text":"q3 2026","intent":"recency_check"}]}'
    )
    out = provider_router.parse_planner_output(raw, "fallback")
    assert out.strategy == "Find official source then compare"
    assert [q.text for q in out.queries] == ["q1", "q2", "q3 2026"]
    assert out.queries[0].intent == QueryIntent.PRIMARY


def test_parse_planner_output_fallback_on_invalid():
    out = provider_router.parse_planner_output("not json", "fallback query")
    assert out.strategy == "Direct retrieval fallback"
    assert len(out.queries) == 1
    assert out.queries[0].text == "fallback query"
    assert out.queries[0].intent == QueryIntent.PRIMARY


# ── Goal 6: CoT filter extension ──────────────────────────────────────────


def test_cot_filter_strips_deepseek_reasoning_content():
    """`reasoning_content` (DeepSeek R1) and `reasoning_tokens` must be filtered."""
    import main as main_mod

    payload = {"step": "generating", "data": {"reasoning_content": "Let me think..."}}
    assert main_mod._contains_cot(payload)

    payload = {"step": "generating", "data": {"reasoning_tokens": 1234}}
    assert main_mod._contains_cot(payload)


def test_cot_filter_strips_claude_thinking_blocks():
    """Anthropic-style `<thought>...</thought>` blocks must be filtered."""
    import main as main_mod

    payload = {
        "step": "generating",
        "data": "Here is some <thought>internal reasoning</thought> leaked in",
    }
    assert main_mod._contains_cot(payload)


def test_cot_filter_strips_redacted_thinking():
    """Claude API `redacted_thinking` content blocks must be filtered."""
    import main as main_mod

    payload = {"step": "generating", "data": {"type": "redacted_thinking", "data": "..."}}
    assert main_mod._contains_cot(payload)


def test_cot_filter_keeps_normal_payload_through():
    """Sanity: normal answer chunks are NOT flagged as CoT."""
    import main as main_mod

    payload = {"step": "generating", "data": "The repo rate is 6.5%. [doc_1]"}
    assert not main_mod._contains_cot(payload)


def test_cancellation_emits_single_error_event(monkeypatch, tmp_path):
    """Regression: cancel path used to emit TWO ``error/cancelled`` events
    (a legacy string-payload and a typed dict-payload). Now exactly one."""
    import importlib

    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "cancel-single.db"))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr("utils.env_check.validate", lambda: None, raising=False)

    import agent.memory as mem_mod
    importlib.reload(mem_mod)
    import agent.orchestrator as orch_mod
    importlib.reload(orch_mod)

    asyncio.run(mem_mod.init_db())

    from agent.models import PlannerOutput, QueryIntent, TypedQuery
    from utils.cancellation import CancellationToken
    from utils import provider_router as pr

    async def fake_plan(*a, **kw):
        return PlannerOutput(
            strategy="t",
            queries=[TypedQuery(text="q", intent=QueryIntent.PRIMARY)],
        )

    monkeypatch.setattr(pr, "plan", fake_plan)

    async def _drive() -> list:
        tok = CancellationToken()
        orch = orch_mod.ResearchOrchestrator()
        events: list = []
        gen = orch.run("q", "s-single", cancel_token=tok, turn_id="t-single")
        # Pull first event, then cancel.
        first = await gen.__anext__()
        events.append(first)
        tok.cancel()
        async for ev in gen:
            events.append(ev)
        return events

    events = asyncio.run(_drive())
    cancel_events = [e for e in events if e.step == "error" and e.label == "cancelled"]
    assert len(cancel_events) == 1, (
        f"Expected exactly one error/cancelled event, got {len(cancel_events)}: {cancel_events}"
    )
    # Payload should be a structured dict, not a bare string.
    assert isinstance(cancel_events[0].data, dict)
    assert cancel_events[0].data.get("message") == "Cancelled by user."


def test_eval_runner_timeout_is_recorded(monkeypatch, tmp_path):
    eval_runner = importlib.import_module("eval.eval_runner")

    dataset = [
        {"id": "T-1", "category": "factual", "query": "slow query", "is_multiturn": False}
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    class SlowAgent:
        async def run(self, query, session_id):
            await asyncio.sleep(2)
            if False:
                yield None

        async def aclose(self):
            return None

    async def noop_init_db():
        return None

    async def fake_faithfulness(*args, **kwargs):
        class R:
            faithfulness_score = 0.5
        return R()

    async def fake_relevance(*args, **kwargs):
        class R:
            answer_relevance_score = 0.5
        return R()

    monkeypatch.setattr(eval_runner, "_DATASET", Path(dataset_path))
    monkeypatch.setattr(eval_runner, "_RESULTS_DIR", results_dir)
    monkeypatch.setattr(eval_runner, "_PER_QUESTION_TIMEOUT_S", 1)
    monkeypatch.setattr(eval_runner, "init_db", noop_init_db)
    monkeypatch.setattr(eval_runner, "ResearchOrchestrator", SlowAgent)
    monkeypatch.setattr(eval_runner, "judge_faithfulness", fake_faithfulness)
    monkeypatch.setattr(eval_runner, "judge_relevance", fake_relevance)

    asyncio.run(eval_runner.run_eval())

    files = sorted(results_dir.glob("eval_*.jsonl"))
    assert files, "Expected eval result file"
    lines = files[-1].read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert "timeout" in record["agent_answer"].lower()
