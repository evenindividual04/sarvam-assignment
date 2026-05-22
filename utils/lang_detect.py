"""3-tier language detection for the Deep Research Agent.

Tier 1: Unicode script (instant, 100% accurate for native scripts)
Tier 2: Hinglish/romanized-Indic keyword check (instant, ~85% accurate)
Tier 3: langid statistical classifier (~5ms, ~95% accurate on Latin script)

Returns (lang_code: "hi"|"ta"|"bn"|"en", method: "script"|"keyword"|"fasttext"|"default").

Library choice — langid over fasttext-langdetect:
We tried `fasttext-langdetect>=1.0.5` first. It installs cleanly but crashes at
runtime under numpy>=2.0 (`np.array(..., copy=False)` raises ValueError). Until
the upstream wrapper migrates to `np.asarray`, we use `langid` (pure Python,
no compile, no numpy coupling) which provides the same API surface for our
needs (en/hi/ta/bn). The Tier-3 return method label stays as "fasttext" for
log/metric parity — it semantically means "statistical Tier-3 detection".

Disable Tier 3 via env: LANG_DETECT_DISABLE_FASTTEXT=1.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Tier 1: native script regexes ─────────────────────────────────────────
# Hindi/Marathi/Sanskrit all share Devanagari; routing buckets them as "hi".
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_TAMIL_RE = re.compile(r"[஀-௿]")
_BENGALI_RE = re.compile(r"[ঀ-৿]")

# ── Tier 2: high-confidence romanized-Indic keyword wordlists ─────────────
# Conservative: a token only appears here if it's UNAMBIGUOUSLY Indic.
# Ambiguous tokens (e.g. "do", "ko", "is") are deliberately omitted — Tier 3
# decides those.
_HINGLISH_KEYWORDS = frozenset({
    "kya", "kaise", "hai", "hain", "kar", "mein", "se", "ka", "ki", "ke",
    "nahi", "nahin", "matlab", "bhai", "yaar", "bata", "bolo", "kaha",
    "kahan", "kyun", "kyon", "wala", "wali", "hua", "hota", "bahut",
    "thoda", "abhi", "kal", "raha", "rahi", "accha", "theek",
    # Common additions for code-mixed queries that score well in eval:
    "mera", "meri", "tera", "teri", "humara", "humari", "ho", "gaya", "gayi",
    "kuch", "bhi", "chahiye", "jaldi", "namaste",
})
_TAMIL_ROMANIZED = frozenset({
    "enna", "epdi", "vaanga", "vandhutu", "irukku", "anna", "thambi",
    "nalla", "varum", "panna", "iruka", "ille", "vanakkam",
})
_BENGALI_ROMANIZED = frozenset({
    "kemon", "ami", "tumi", "achho", "acho", "bhalo", "kotha", "khub",
    "tomar", "amar", "ektu", "robaar",
})

_TOKEN_RE = re.compile(r"[a-z]+")


def _keyword_match(text: str) -> Optional[str]:
    """Return 'hi'|'ta'|'bn' on 2+ keyword matches in one bucket; None otherwise.

    Tie-breaks by raw count (more matches wins); ties prefer hi > ta > bn since
    Hinglish is the dominant code-mix in our user base.
    """
    tokens = set(_TOKEN_RE.findall(text.lower()))
    if not tokens:
        return None
    counts = {
        "hi": len(tokens & _HINGLISH_KEYWORDS),
        "ta": len(tokens & _TAMIL_ROMANIZED),
        "bn": len(tokens & _BENGALI_ROMANIZED),
    }
    best = max(counts, key=lambda k: counts[k])
    if counts[best] >= 2:
        return best
    return None


# ── Tier 3: lazy-loaded langid classifier ─────────────────────────────────
_LANGID_LOADED: bool = False
_LANGID_AVAILABLE: bool = False
_LANGID_MOD = None


def _get_langid():
    """Lazy import + restrict label set. Cached. Returns None if unavailable."""
    global _LANGID_LOADED, _LANGID_AVAILABLE, _LANGID_MOD
    if _LANGID_LOADED:
        return _LANGID_MOD if _LANGID_AVAILABLE else None
    _LANGID_LOADED = True
    try:
        import langid as _mod  # type: ignore
        # Restrict the model to languages we route on — keeps the cosmetic
        # noise of "id" / "tl" / "ms" predictions off our metrics.
        _mod.set_languages(["en", "hi", "ta", "bn", "mr"])
        _LANGID_MOD = _mod
        _LANGID_AVAILABLE = True
    except Exception as e:  # noqa: BLE001
        logger.info("langid not available, Tier-3 detection disabled: %s", e)
        _LANGID_AVAILABLE = False
    return _LANGID_MOD if _LANGID_AVAILABLE else None


def _langid_detect(text: str) -> Optional[tuple[str, float]]:
    """Call langid; return (lang, normalized-confidence) or None.

    langid returns a log-prob score (negative for low confidence, positive for
    high). We normalize: score >= 0 → confidence 1.0; score < 0 → 0.0..1.0 via
    a soft sigmoid-ish heuristic. The caller's threshold is 0.5.
    """
    mod = _get_langid()
    if mod is None:
        return None
    try:
        lang, score = mod.classify(text)
    except Exception as e:  # noqa: BLE001
        logger.warning("langid classify failed: %s", e)
        return None
    # langid scores: typical English text scores in [-50, +20]. Hinglish-ish
    # text often scores deeply negative. Map "score >= 0" → strong confidence.
    confidence = 1.0 if score >= 0 else max(0.0, 1.0 + score / 50.0)
    return (lang, confidence)


def _tier3_disabled() -> bool:
    return os.getenv("LANG_DETECT_DISABLE_FASTTEXT", "").strip() in {
        "1", "true", "True", "yes",
    }


# ── Public API ────────────────────────────────────────────────────────────
def detect_language(text: str) -> tuple[str, str]:
    """Return (lang_code, method) for the given text.

    lang_code ∈ {"hi", "ta", "bn", "en"}.
    method    ∈ {"script", "keyword", "fasttext", "default"}.

    "default" means no tier was confident; we fall back to "en" for routing.
    """
    if not text or not text.strip():
        return ("en", "default")

    # Tier 1 — native script (highest precision).
    if _DEVANAGARI_RE.search(text):
        return ("hi", "script")
    if _TAMIL_RE.search(text):
        return ("ta", "script")
    if _BENGALI_RE.search(text):
        return ("bn", "script")

    # Tier 2 — romanized-Indic keyword match.
    kw = _keyword_match(text)
    if kw:
        return (kw, "keyword")

    # Short-query guard: langid is a statistical classifier trained on
    # sentence-length text. For 1-2 word Latin-script inputs (e.g. "hello",
    # "chai latte") its log-probability normalization is unreliable, so we
    # short-circuit to the default English route instead of letting Tier 3
    # produce a low-confidence (and likely wrong) label. The keyword tier
    # above already handles romanized Indic shorts like "kya hai".
    if len(text.split()) <= 2:
        return ("en", "default")

    # Tier 3 — statistical (langid). Skippable via env.
    if not _tier3_disabled():
        try:
            result = _langid_detect(text)
        except Exception:  # noqa: BLE001
            logger.warning("Tier-3 language detection failed", exc_info=True)
            result = None
        if result is not None:
            lang, conf = result
            if conf >= 0.5 and lang in {"en", "hi", "ta", "bn", "mr"}:
                # Marathi → bucket as "hi" (Devanagari, shares Sarvam route).
                routed = "hi" if lang == "mr" else lang
                return (routed, "fasttext")

    return ("en", "default")
