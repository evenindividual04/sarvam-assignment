from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.citation_guard import CitationGuard, convert_citations
from agent.context_engine import format_context_xml, rank_and_select
from agent.models import ContextSnippet


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str


def _make_snippet(domain: str, text: str, score: float, tokens: int = 120) -> ContextSnippet:
    s = ContextSnippet(
        doc_id="doc_1",
        url=f"https://{domain}/article",
        title=f"{domain} title",
        domain=domain,
        text=text,
        snippet=text[:80],
        token_count=tokens,
        retrieved_at="2026-01-01T00:00:00+00:00",
    )
    s.final_score = score
    return s


def check_empty_context_citations() -> CheckResult:
    guard = CitationGuard()
    score = guard.verify("No citations here.", {}, set())
    ok = score == 1.0
    return CheckResult("empty_context_no_citations", ok, f"score={score}")


def check_hallucinated_citation_detection() -> CheckResult:
    guard = CitationGuard()
    answer = "Claim [doc_1] and bad [doc_2]"
    doc_map = {
        "doc_1": ("A", "https://a.com", "a.com"),
        "doc_2": ("B", "https://b.com", "b.com"),
    }
    score = guard.verify(answer, doc_map, {"https://a.com"})
    ok = abs(score - 0.5) < 1e-9
    return CheckResult("hallucinated_citation_detection", ok, f"score={score}")


def check_xml_escaping() -> CheckResult:
    chunk = _make_snippet("x.com", 'alpha < beta & "quotes"', 1.0)
    xml, _ = format_context_xml([chunk])
    ok = all(x in xml for x in ["&lt;", "&amp;", "&quot;"])
    return CheckResult("xml_escaping", ok, "escaped markers present")


def check_diversity_selection() -> CheckResult:
    chunks = [
        _make_snippet("same.com", "a " * 800, 1.0, tokens=1200),
        _make_snippet("d1.com", "b", 0.9, tokens=300),
        _make_snippet("d2.com", "c", 0.8, tokens=300),
    ]
    selected = rank_and_select("test", chunks, max_tokens=700)
    domains = [c.domain for c in selected]
    ok = "d1.com" in domains and "d2.com" in domains
    return CheckResult("oversized_chunk_skip", ok, f"selected={domains}")


def check_citation_conversion() -> CheckResult:
    doc_map = {"doc_1": ("Title", "https://x.com", "x.com")}
    out = convert_citations("Fact [doc_1]", doc_map)
    ok = "[Title — x.com](https://x.com)" in out
    return CheckResult("citation_conversion", ok, out)


def run_all() -> int:
    checks = [
        check_empty_context_citations(),
        check_hallucinated_citation_detection(),
        check_xml_escaping(),
        check_diversity_selection(),
        check_citation_conversion(),
    ]

    failures = [c for c in checks if not c.ok]
    for c in checks:
        status = "PASS" if c.ok else "FAIL"
        print(f"{status} {c.name}: {c.details}")

    if failures:
        print(f"RED_TEAM_FAIL count={len(failures)}")
        return 1

    print("RED_TEAM_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
