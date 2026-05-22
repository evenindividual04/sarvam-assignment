"""
Citation guard: verify all [doc_N] in answer map to real fetched URLs,
and convert internal [doc_N] markers to [Title — domain](URL) format.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# Pattern matching [doc_N] where N is one or more digits
_DOC_PATTERN = re.compile(r'\[doc_\d+\]')
_DOC_ID_PATTERN = re.compile(r'doc_\d+')
_GROUPED_DOC_PATTERN = re.compile(r'\[(doc_\d+(?:\s*,\s*doc_\d+)*)\]')

# Quote/claim/citation triplet for ReClaim-style quote-first synthesis.
# Captures: <quote>...</quote> <claim>...</claim> [doc_N] (one or more cites).
_QUOTE_CLAIM_RE = re.compile(
    r'<quote>(.*?)</quote>\s*<claim>(.*?)</claim>\s*((?:\[doc_\d+\]\s*)+)',
    re.DOTALL | re.IGNORECASE,
)
_WS_RE = re.compile(r'\s+')

# Phase 1.875: numeric / date / year regexes for the numeric-grounding audit.
# Ordering matters when iterating: dates first (longest spans), then years,
# then bare numbers — so the ISO/long-form patterns don't get masked by the
# 4-digit-year fallback.
_DATE_RE_ISO = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
_DATE_RE_LOOSE = re.compile(
    r'\b\d{1,2}[\s/-]'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
    r'|January|February|March|April|May|June|July|August|September|October|November|December)'
    r'[\s/-]\d{2,4}\b',
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r'\b(?:19|20|21)\d{2}\b')
# Match plain numbers and percentages. Avoid matching the digit part of an
# already-claimed year/date by walking the answer with a position-aware pass.
_NUMBER_RE = re.compile(r'\b\d+(?:[.,]\d+)?%?')


def parse_numeric_tokens(answer: str) -> list[dict]:
    """Extract numbers, years, and date tokens from a generated answer.

    Returns a list of dicts ``{"token", "kind", "position"}`` where ``kind`` is
    one of ``"date"``, ``"year"``, ``"number"``. Spans don't overlap: dates
    are detected first, then years, then plain numbers (with positions inside
    already-claimed spans skipped). Never raises.
    """
    if not answer:
        return []
    claimed: list[tuple[int, int]] = []
    out: list[dict] = []

    def _overlaps(start: int, end: int) -> bool:
        for s, e in claimed:
            if start < e and end > s:
                return True
        return False

    try:
        for m in _DATE_RE_ISO.finditer(answer):
            out.append({"token": m.group(0), "kind": "date", "position": m.start()})
            claimed.append((m.start(), m.end()))
        for m in _DATE_RE_LOOSE.finditer(answer):
            if _overlaps(m.start(), m.end()):
                continue
            out.append({"token": m.group(0), "kind": "date", "position": m.start()})
            claimed.append((m.start(), m.end()))
        for m in _YEAR_RE.finditer(answer):
            if _overlaps(m.start(), m.end()):
                continue
            out.append({"token": m.group(0), "kind": "year", "position": m.start()})
            claimed.append((m.start(), m.end()))
        for m in _NUMBER_RE.finditer(answer):
            if _overlaps(m.start(), m.end()):
                continue
            out.append({"token": m.group(0), "kind": "number", "position": m.start()})
            claimed.append((m.start(), m.end()))
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("parse_numeric_tokens failed: %s", e)
        return []
    return out


def _normalize_for_match(s: str) -> str:
    """Lowercase + collapse whitespace for substring presence checks."""
    return _WS_RE.sub(' ', s).strip().lower()


def parse_quote_claim_blocks(answer: str) -> list[dict]:
    """
    Extract every <quote>...</quote><claim>...</claim>[doc_N]+ triplet from the
    synthesized answer. Returns a list of dicts with the parsed quote text,
    claim text, the tuple of doc ids cited immediately after, and offsets.

    Note: ``_DOC_ID_INNER_RE`` (defined below this function) is referenced
    inside the loop body, NOT at module-load time. By the time this function
    is *called*, every module-level statement has already executed and the
    name resolves correctly. A reviewer flagged this as a potential
    ``NameError`` — that is a false positive; Python function bodies execute
    lazily.
    """
    out: list[dict] = []
    for m in _QUOTE_CLAIM_RE.finditer(answer):
        quote = m.group(1).strip()
        claim = m.group(2).strip()
        cite_block = m.group(3)
        doc_ids = tuple(_DOC_ID_INNER_RE.findall(cite_block))
        if not doc_ids:
            continue
        out.append(
            {
                "quote": quote,
                "claim": claim,
                "doc_ids": doc_ids,
                "start": m.start(),
                "end": m.end(),
            }
        )
    return out


_CLAIM_RE = re.compile(
    r'([^.!?\n]*?[.!?])\s*((?:\[doc_\d+\]\s*)+)',
    re.MULTILINE,
)
_DOC_ID_INNER_RE = re.compile(r'\[(doc_\d+)\]')


def parse_claims_with_citations(
    answer: str,
) -> list[tuple[str, tuple[str, ...], tuple[int, int]]]:
    """
    Extract (claim_text, doc_ids, (start, end_of_citation_block)) tuples.

    The citation block end is the offset after the final `]` of the trailing
    citation group, so callers can splice in markers without disturbing it.
    """
    out: list[tuple[str, tuple[str, ...], tuple[int, int]]] = []
    for m in _CLAIM_RE.finditer(answer):
        claim_text = m.group(1)
        citation_block = m.group(2)
        doc_ids = tuple(_DOC_ID_INNER_RE.findall(citation_block))
        if not doc_ids:
            continue
        # End of citation block — strip trailing whitespace included by `\s*`.
        end = m.end(2)
        # Walk back past any trailing whitespace included in group 2.
        while end > m.start(2) and answer[end - 1].isspace():
            end -= 1
        out.append((claim_text, doc_ids, (m.start(1), end)))
    return out


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
            # Spec format: `[Title — domain] (URL)` with a space before the
            # opening paren. This is *not* a standard markdown link (which
            # forbids the space) — the spec example wins, and the frontend's
            # markdown renderer treats the URL as plain text. To keep the
            # answer clickable in the UI, we emit both: the spec-shaped
            # label-plus-URL and a parenthesized auto-link. Net rendered
            # output reads as "[Title — domain] (https://url)" with the URL
            # auto-linked by markdown's URL-detection.
            return f"[{title} — {domain}] ({url})"
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

    def verify_quoted_text(
        self,
        answer: str,
        doc_map: dict[str, tuple[str, str, str]],
        snippet_lookup: dict[str, str],
    ) -> dict:
        """
        Audit ReClaim-style <quote>...</quote><claim>...</claim>[doc_N] triplets.

        For each parsed triplet, verify that the quoted substring appears
        (case-insensitive, whitespace-normalized) inside at least one of the
        cited documents' snippet text. Absence of quote blocks is not penalized
        (ratio defaults to 1.0).

        Returns a dict shaped:
            {
              "audit": [{"quote", "doc_ids", "grounded", "matched_doc_id"}],
              "total_quoted_claims": int,
              "grounded_claims": int,
              "quote_grounding_ratio": float,
            }
        """
        safe_default: dict = {
            "audit": [],
            "total_quoted_claims": 0,
            "grounded_claims": 0,
            "quote_grounding_ratio": 1.0,
        }
        try:
            blocks = parse_quote_claim_blocks(answer)
        except Exception as e:  # defensive — regex shouldn't throw, but be safe
            logger.warning("verify_quoted_text: parse failed: %s", e)
            return safe_default

        if not blocks:
            return safe_default

        # Pre-normalize snippets once.
        norm_snippets: dict[str, str] = {}
        for doc_id, text in (snippet_lookup or {}).items():
            try:
                norm_snippets[doc_id] = _normalize_for_match(text or "")
            except Exception as e:  # pragma: no cover — defensive
                logger.warning("verify_quoted_text: snippet normalize failed for %s: %s", doc_id, e)
                norm_snippets[doc_id] = ""

        audit: list[dict] = []
        grounded_count = 0
        for blk in blocks:
            try:
                quote_norm = _normalize_for_match(blk["quote"])
            except Exception as e:  # pragma: no cover
                logger.warning("verify_quoted_text: quote normalize failed: %s", e)
                quote_norm = ""

            matched_doc_id: Optional[str] = None
            if quote_norm:
                for doc_id in blk["doc_ids"]:
                    snippet = norm_snippets.get(doc_id, "")
                    if snippet and quote_norm in snippet:
                        matched_doc_id = doc_id
                        break

            grounded = matched_doc_id is not None
            if grounded:
                grounded_count += 1
            audit.append(
                {
                    "quote": blk["quote"],
                    "doc_ids": blk["doc_ids"],
                    "grounded": grounded,
                    "matched_doc_id": matched_doc_id,
                }
            )

        total = len(blocks)
        ratio = (grounded_count / total) if total else 1.0
        return {
            "audit": audit,
            "total_quoted_claims": total,
            "grounded_claims": grounded_count,
            "quote_grounding_ratio": ratio,
        }

    def verify_numeric_grounding(
        self,
        answer: str,
        doc_map: dict[str, tuple[str, str, str]],
        snippet_lookup: dict[str, str],
    ) -> dict:
        """Phase 1.875: anti-hallucination audit on numeric/date/year tokens.

        For each numeric token in the answer, check substring presence in
        any cited document's snippet text (case-insensitive, whitespace-
        normalized). Tokens with zero matches are flagged ``grounded=False``;
        the frontend renders a small ``⚠ unverified`` badge around them.

        Returns ``{"audit": [...], "total": N, "grounded": M,
        "numeric_grounding_ratio": M/N if N>0 else 1.0}``. Never raises —
        on any error returns a safe default.
        """
        safe_default = {"audit": [], "total": 0, "grounded": 0, "numeric_grounding_ratio": 1.0}
        try:
            tokens = parse_numeric_tokens(answer)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("verify_numeric_grounding: parse failed: %s", e)
            return safe_default
        if not tokens:
            return safe_default

        # Pre-normalize ALL snippets — for numeric tokens we don't restrict
        # to cited docs (a token may appear in any retrieved snippet and
        # still be grounded; the per-claim audit is in `verify_quoted_text`).
        norm_snippets: list[tuple[str, str]] = []
        for doc_id, text in (snippet_lookup or {}).items():
            try:
                norm_snippets.append((doc_id, _normalize_for_match(text or "")))
            except Exception:
                norm_snippets.append((doc_id, ""))

        audit: list[dict] = []
        grounded = 0
        for tok in tokens:
            raw = tok["token"]
            needle = raw.lower().strip()
            matched_doc: Optional[str] = None
            for doc_id, snip in norm_snippets:
                if needle and needle in snip:
                    matched_doc = doc_id
                    break
            is_grounded = matched_doc is not None
            if is_grounded:
                grounded += 1
            audit.append({
                "token": raw,
                "kind": tok["kind"],
                "position": tok["position"],
                "grounded": is_grounded,
                "matched_doc_id": matched_doc,
            })

        total = len(tokens)
        ratio = (grounded / total) if total else 1.0
        return {"audit": audit, "total": total, "grounded": grounded,
                "numeric_grounding_ratio": ratio}
