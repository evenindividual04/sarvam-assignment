from __future__ import annotations

from utils.provider_adapters import normalize_search_items
from utils.prompt_registry import PROMPT_REGISTRY, prompt_id
from utils.failure_policy import FailurePolicy


def _domain(url: str) -> str:
    return url.split('/')[2]


def test_adapter_normalizes_valid_payload():
    payload = {"results": [{"url": "https://a.com/x", "title": "A", "snippet": "S"}]}
    out = normalize_search_items("parallel", payload, "2026-01-01T00:00:00+00:00", _domain)
    assert out.outcome.ok
    assert len(out.results) == 1
    assert out.results[0].domain == "a.com"


def test_adapter_flags_schema_mismatch():
    payload = {"results": {"url": "https://a.com/x"}}
    out = normalize_search_items("parallel", payload, "2026-01-01T00:00:00+00:00", _domain)
    assert not out.outcome.ok
    assert out.outcome.adapter_error_code == "parallel_schema_mismatch"


def test_prompt_registry_ids_nonempty():
    assert prompt_id("planner")
    assert prompt_id("synthesizer")
    assert "id" in PROMPT_REGISTRY["conflict_detection"]


def test_failure_policy_defaults_parsed():
    policy = FailurePolicy()
    assert policy.plan_timeout_s > 0
    assert policy.search_timeout_s > 0
    assert policy.max_retries_per_provider >= 1
