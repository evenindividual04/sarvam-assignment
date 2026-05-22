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

# B4: detect a Markdown disagreement matrix in the answer. We require a header
# row containing all three column labels (Claim/Source A/Source B) followed by
# the GFM separator row (---). Whitespace inside cells is tolerated.
_DISAGREEMENT_TABLE_RE = re.compile(
    r'\|\s*Claim\s*\|\s*Source\s*A\s*\|\s*Source\s*B\s*\|\s*\n\s*\|\s*-{2,}',
    re.IGNORECASE,
)


def has_disagreement_matrix(answer: str) -> bool:
    """B4: True if ``answer`` contains a Markdown table with the mandated
    ``| Claim | Source A | Source B |`` header. Used to audit whether the
    synthesizer honored the disagreement-matrix instruction when a conflict
    was detected."""
    if not answer:
        return False
    return _DISAGREEMENT_TABLE_RE.search(answer) is not None


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


def extract_doc_ids(text: str) -> list[str]:
    """Extract doc ids even from grouped refs such as [doc_1, doc_2]."""
    ids = []
    for match in _GROUPED_DOC_PATTERN.finditer(text):
        inner = match.group(1)
        for doc_id in inner.split(","):
            ids.append(doc_id.strip())
    return ids


_QUOTE_TAG_RE = re.compile(r'<quote>(.*?)</quote>', re.DOTALL | re.IGNORECASE)
_CLAIM_TAG_RE = re.compile(r'<claim>(.*?)</claim>', re.DOTALL | re.IGNORECASE)


def strip_quote_claim_tags(answer: str) -> str:
    """Strip the ReClaim-style <quote>/<claim> wrapper tags from the answer
    text used for display. The synthesizer is instructed to wrap each
    verbatim quote in <quote>...</quote> and each paraphrase in
    <claim>...</claim>, both followed by [doc_N] citations. Those tags are
    invaluable for the audit pipeline (build_cite_quote_map,
    verify_quoted_text, judge faithfulness scoring) but they leak through
    react-markdown as raw text, so chat answers showed literal
    "<quote>...</quote>" strings.

    Transformation:
      <quote>X</quote>      → *"X"*      (italic curly-quoted span)
      <claim>Y</claim>      → Y          (plain inline text)

    Idempotent — re-running on an already-stripped answer is a no-op.
    The audit code that needs the original tags reads `full_answer` (the
    pre-conversion buffer), so this strip only affects the display copy.
    """
    if not answer:
        return answer
    answer = _QUOTE_TAG_RE.sub(lambda m: f'*"{m.group(1).strip()}"*', answer)
    answer = _CLAIM_TAG_RE.sub(lambda m: m.group(1).strip(), answer)
    return answer


def convert_citations(answer: str, doc_map: dict[str, tuple[str, str, str]]) -> str:
    """
    Replace [doc_N] with [Title — domain](URL).
    doc_map: {"doc_1": ("Title", "https://url.com", "domain.com")}
    Unknown doc IDs are left unchanged.
    Also strips ReClaim-style <quote>/<claim> wrapper tags so they don't
    leak through to the chat UI as raw HTML.
    """
    answer = strip_quote_claim_tags(answer)
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
    answer = _DOC_PATTERN.sub(replace, answer)
    return _dedupe_sources_block(answer)


# Heading markers that delimit the final "Sources" list — same-URL duplicates
# inside this block should collapse so the reader doesn't see the same link
# repeated 4× when the synthesizer cited two chunks from the same page.
_SOURCES_HEADING_RE = re.compile(
    r"(?im)^\s*(?:#+\s*)?(?:sources|sources?:|स्रोत|स्रोत:|स्रोत-)\s*$"
)
_URL_IN_LINE_RE = re.compile(r"https?://[^\s)\]]+")


def _dedupe_sources_block(answer: str) -> str:
    """Collapse duplicate-URL bullet lines inside a trailing Sources section.

    Catches the common synthesizer pattern where every [doc_N] chunk is
    listed as its own bullet even when multiple chunks come from the same
    page. We only dedupe inside the *last* Sources heading downward — not
    the body — so legitimate in-prose repetitions stay untouched.

    First-occurrence wins. Lines without a URL pass through unchanged.
    No-op if there's no Sources heading.
    """
    if not answer:
        return answer
    # Find the LAST sources heading; only dedupe below it.
    last = None
    for m in _SOURCES_HEADING_RE.finditer(answer):
        last = m
    if last is None:
        return answer
    head, tail = answer[: last.end()], answer[last.end():]
    lines = tail.split("\n")
    seen: set[str] = set()
    out_lines: list[str] = []
    for line in lines:
        urls_in_line = _URL_IN_LINE_RE.findall(line)
        if urls_in_line:
            # Normalise each URL the same way the in-pool check does
            # (strip query, drop scheme case, trim trailing slash) so
            # http://x.com/page and https://x.com/page/ collapse together.
            norm_urls = {_normalize_for_dedupe(u) for u in urls_in_line}
            if norm_urls and norm_urls.issubset(seen):
                continue
            seen.update(norm_urls)
        out_lines.append(line)
    return head + "\n".join(out_lines)


def _normalize_for_dedupe(url: str) -> str:
    """Light URL normalisation for dedup — preserves path but ignores
    scheme case, www prefix, trailing slash, and URL fragment."""
    u = url.strip()
    u = u.split("#", 1)[0]
    if u.endswith("/"):
        u = u[:-1]
    u_low = u.lower()
    if u_low.startswith("https://www."):
        u = "https://" + u[12:]
    elif u_low.startswith("http://www."):
        u = "http://" + u[11:]
    return u.lower()


# B5: quote-anchored citation popovers.
# Sentence boundary used when expanding an n-gram match to a full sentence.
_SENT_END_RE = re.compile(r'[.!?]\s')
_TOKEN_RE = re.compile(r"\w+")
# Words ignored when picking the longest claim n-gram. Mirrors the small stop
# list in claim_verifier so behavior is consistent across the two audits.
_QUOTE_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "with", "by", "from", "as", "also", "not", "no",
    "yes", "has", "have", "had", "will", "would", "can", "could", "should",
})


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, sentence_text)`` tuples covering ``text``."""
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for m in re.finditer(r'[.!?](?:\s|$)', text):
        end = m.end()
        s = text[cursor:end].strip()
        if s:
            spans.append((cursor, end, s))
        cursor = end
    if cursor < len(text):
        tail = text[cursor:].strip()
        if tail:
            spans.append((cursor, len(text), tail))
    return spans


def _truncate_at_sentence_boundary(text: str, max_chars: int) -> str:
    """Truncate ``text`` to at most ``max_chars`` characters, preferring to cut
    at the nearest sentence boundary (``.?!\\n``) within the slice; failing
    that, cut at the last whitespace. Appends an ellipsis when truncation
    actually shortens the text. Pure-Python, never raises.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text.strip()
    snippet = text[:max_chars]
    # Prefer the last sentence-terminator inside the window.
    best_cut = -1
    for ch in ".?!\n":
        idx = snippet.rfind(ch)
        if idx > best_cut:
            best_cut = idx
    if best_cut >= max_chars // 2:
        return snippet[: best_cut + 1].strip()
    # Fallback: last whitespace.
    space_idx = snippet.rfind(" ")
    if space_idx >= max_chars // 2:
        return snippet[:space_idx].rstrip() + "…"
    return snippet.rstrip() + "…"


# Minimum token-overlap score required to trust the sentence-scoring result.
# Below this threshold we fall back to a clean head-of-chunk excerpt so the
# UI popover always renders SOMETHING grounded in the source.
_ANCHOR_SCORE_THRESHOLD = 1


def _extract_anchor_quote(
    claim_text: str, chunk_text: str, max_chars: int = 240
) -> Optional[str]:
    """Find the verbatim sentence in ``chunk_text`` most lexically anchored to
    ``claim_text``.

    Strategy: split the chunk into sentences, score each by the count of
    shared content tokens with the claim (stop-words and short tokens
    excluded), and return the best-scoring sentence. When the best score is
    below ``_ANCHOR_SCORE_THRESHOLD`` (i.e. no meaningful overlap, e.g. a
    very short paraphrased claim), fall back to a clean head-of-chunk
    excerpt truncated at the nearest sentence boundary — this guarantees
    the citation popover always renders something useful from the source.
    Pure-Python — no LLM call. Returns ``None`` only on empty inputs.
    """
    if not claim_text or not chunk_text:
        return None
    claim_tokens = {
        t for t in _TOKEN_RE.findall(claim_text.lower())
        if t not in _QUOTE_STOPWORDS and len(t) > 2
    }
    sentences = _split_sentences(chunk_text)
    if not sentences:
        head = _truncate_at_sentence_boundary(chunk_text, max_chars)
        return head or None

    best_score = 0
    best_sentence = sentences[0][2]
    if claim_tokens:
        for _start, _end, sent in sentences:
            sent_tokens = {
                t for t in _TOKEN_RE.findall(sent.lower())
                if t not in _QUOTE_STOPWORDS and len(t) > 2
            }
            score = len(claim_tokens & sent_tokens)
            if score > best_score:
                best_score = score
                best_sentence = sent

    # Deterministic fallback: if the best sentence has no meaningful overlap
    # with the claim, prefer a sentence-boundary-clean prefix of the chunk
    # over the (arbitrary) first sentence picked by the scoring loop. The
    # head of a retrieved chunk is empirically its most representative text.
    if best_score < _ANCHOR_SCORE_THRESHOLD:
        head = _truncate_at_sentence_boundary(chunk_text, max_chars)
        return head or None

    snippet = best_sentence.strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "…"
    return snippet or None


def build_cite_quote_map(
    answer: str,
    doc_map: dict[str, tuple[str, str, str]],
    snippet_lookup: dict[str, str],
    max_chars: int = 240,
) -> dict[str, str]:
    """B5: for each ``[doc_N]`` cited in ``answer``, return the verbatim quote
    from the cited chunk that best anchors the surrounding claim sentence.

    Keys are ``doc_id`` strings (e.g. ``"doc_1"``) so the result is JSON-safe
    for persistence in ``run_metadata`` and transport to the frontend. Never
    raises — defensive callers can rely on an empty dict on failure.
    """
    out: dict[str, str] = {}
    if not answer or not doc_map or not snippet_lookup:
        return out
    try:
        parsed = parse_claims_with_citations(answer)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("build_cite_quote_map: parse failed: %s", e)
        return out
    for claim_text, doc_ids, _ in parsed:
        for doc_id in doc_ids:
            if doc_id in out:
                continue
            chunk = snippet_lookup.get(doc_id)
            if not chunk:
                continue
            quote = _extract_anchor_quote(claim_text, chunk, max_chars=max_chars)
            if quote:
                out[doc_id] = quote
    return out


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
        cited_ids = extract_doc_ids(answer)

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
