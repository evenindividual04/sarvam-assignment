"""
Citation guard: verify all [doc_N] in answer map to real fetched URLs,
and convert internal [doc_N] markers to [Title — domain](URL) format.
"""
from __future__ import annotations

import re
from typing import Optional


# Pattern matching [doc_N] where N is one or more digits
_DOC_PATTERN = re.compile(r'\[doc_\d+\]')
_DOC_ID_PATTERN = re.compile(r'doc_\d+')
_GROUPED_DOC_PATTERN = re.compile(r'\[(doc_\d+(?:\s*,\s*doc_\d+)*)\]')


def _extract_doc_ids(text: str) -> list[str]:
    """Extract doc ids even from grouped refs such as [doc_1, doc_2]."""
    ids = []
    for match in _GROUPED_DOC_PATTERN.finditer(text):
        inner = match.group(1)
        for doc_id in inner.split(","):
            ids.append(doc_id.strip())
    return ids


def convert_citations(answer: str, doc_map: dict[str, tuple[str, str, str]]) -> str:
    """
    Replace [doc_N] with [Title — domain](URL).
    doc_map: {"doc_1": ("Title", "https://url.com", "domain.com")}
    Unknown doc IDs are left unchanged.
    """
    def to_link(doc_id: str) -> str:
        if doc_id in doc_map:
            title, url, domain = doc_map[doc_id]
            return f"[{title} — {domain}]({url})"
        return f"[{doc_id}]"

    def replace(match: re.Match) -> str:
        raw = match.group(0)           # e.g. [doc_1]
        doc_id = raw[1:-1]             # strip brackets → doc_1
        if doc_id in doc_map:
            return to_link(doc_id)
        return raw

    def replace_grouped(match: re.Match) -> str:
        content = match.group(1)  # e.g. "doc_1, doc_2"
        ids = [c.strip() for c in content.split(",")]
        return ", ".join(to_link(doc_id) for doc_id in ids)

    answer = _GROUPED_DOC_PATTERN.sub(replace_grouped, answer)
    return _DOC_PATTERN.sub(replace, answer)


class CitationGuard:
    """Verifies that every cited [doc_N] has a URL that was actually fetched."""

    def verify(
        self,
        answer: str,
        doc_map: dict[str, tuple[str, str, str]],
        fetched_urls: set[str],
    ) -> float:
        """
        Returns citation integrity score [0.0, 1.0].
        1.0 = all citations valid, 0.0 = all hallucinated.
        An answer with no citations scores 1.0 (no violations).
        """
        cited_ids = _extract_doc_ids(answer)

        if not cited_ids:
            return 1.0

        valid = 0
        for doc_id in cited_ids:
            if doc_id in doc_map:
                _, url, _ = doc_map[doc_id]
                if url in fetched_urls:
                    valid += 1

        return valid / len(cited_ids)
