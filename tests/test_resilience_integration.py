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


def test_synthesize_raises_without_openrouter_key(monkeypatch):
    async def fake_gemini(prompt):
        raise Exception("503 Service Unavailable")
        yield  # pragma: no cover

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(provider_router, "_synthesize_gemini", fake_gemini)

    async def run_once():
        async for _ in provider_router.synthesize(
            query="q",
            context_xml="<context></context>",
            doc_map={},
        ):
            pass

    try:
        asyncio.run(run_once())
        assert False, "Expected synthesis to raise without OpenRouter key"
    except Exception as e:
        assert "503" in str(e)


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
