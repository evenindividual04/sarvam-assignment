"""3-tier language detection tests — covers utils/lang_detect.py.

Tier 1: Unicode script (Devanagari/Tamil/Bengali) — instant, 100% accurate
Tier 2: Hinglish/romanized-Indic keyword match — instant, conservative
Tier 3: langid statistical classifier — ~5ms, ~95% accurate on Latin script

Tests assert both lang code AND method, so we catch silent tier regressions
(e.g. a Hinglish query falling through to Tier 3 instead of being caught at
Tier 2 would be a wordlist coverage bug).
"""
from __future__ import annotations

import importlib
import os

import pytest


def _fresh_module():
    """Reset cached langid module state between tests for env-toggle tests."""
    import utils.lang_detect as mod
    importlib.reload(mod)
    return mod


# ── Tier 1: native script ─────────────────────────────────────────────────


def test_detect_devanagari_query():
    from utils.lang_detect import detect_language
    assert detect_language("नमस्ते कैसे हो") == ("hi", "script")


def test_detect_tamil_query():
    from utils.lang_detect import detect_language
    assert detect_language("தமிழ்நாட்டின் தலைநகர் எது?") == ("ta", "script")


def test_detect_bengali_query():
    from utils.lang_detect import detect_language
    assert detect_language("ভারতের রাজধানী কী?") == ("bn", "script")


# ── Tier 2: romanized-Indic keyword match ─────────────────────────────────


def test_detect_hinglish_kya_haal_hai():
    from utils.lang_detect import detect_language
    assert detect_language("kya haal hai bhai") == ("hi", "keyword")


def test_detect_romanized_tamil_epdi_iruka():
    from utils.lang_detect import detect_language
    # Two matches: "epdi" and "iruka" both in the Tamil-romanized wordlist.
    assert detect_language("epdi iruka anna") == ("ta", "keyword")


def test_detect_romanized_bengali_kemon_acho():
    from utils.lang_detect import detect_language
    # Two matches: "kemon" and "acho" / "tumi".
    assert detect_language("kemon acho tumi") == ("bn", "keyword")


def test_detect_code_mixed_mera_laptop_slow():
    """Code-mixed: "mera", "ho", "gaya", "hai" are all unambiguous Hinglish."""
    from utils.lang_detect import detect_language
    assert detect_language("mera laptop slow ho gaya hai") == ("hi", "keyword")


# ── Tier 3 / fallback behaviour ───────────────────────────────────────────


def test_detect_english_query():
    """Pure English — should land in Tier 3 ("fasttext") or fall through to
    "default" depending on langid confidence. Both are acceptable."""
    from utils.lang_detect import detect_language
    lang, method = detect_language("how to bake sourdough bread at home")
    assert lang == "en"
    assert method in {"fasttext", "default"}


def test_detect_empty_string():
    from utils.lang_detect import detect_language
    assert detect_language("") == ("en", "default")


def test_detect_whitespace_only_string():
    from utils.lang_detect import detect_language
    assert detect_language("   \t \n ") == ("en", "default")


def test_detect_single_token_english():
    """Single-token queries shouldn't strongly trigger any tier — they may
    pass Tier 3 with high confidence ('hello' scores +9 in langid) or fall to
    default. Either way they must remain English."""
    from utils.lang_detect import detect_language
    lang, _ = detect_language("hello")
    assert lang == "en"


def test_short_latin_query_skips_langid():
    """Queries with <=2 Latin-script tokens (and no Tier-2 keyword hit) must
    short-circuit to ('en', 'default') instead of asking langid, whose log-
    probability is unreliable on such short input."""
    from utils.lang_detect import detect_language
    # "hello" — single token, no Indic script, no keyword match.
    assert detect_language("hello") == ("en", "default")
    # "rbi update" — two tokens, English.
    assert detect_language("rbi update") == ("en", "default")


def test_detect_what_is_diwali_stays_english():
    """English question about an Indic topic must NOT be flagged as Hindi.
    None of the tokens are in the Hinglish wordlist; Tier 3 either picks "en"
    or returns low confidence → default → "en"."""
    from utils.lang_detect import detect_language
    lang, _ = detect_language("what is diwali")
    assert lang == "en"


def test_detect_what_does_namaste_mean_ascii_stays_english():
    """A single Hinglish-flavor token in an otherwise English sentence must
    NOT flip detection — Tier 2 requires 2+ matches by design. This protects
    against false positives like "explain karma" or "what is yoga"."""
    from utils.lang_detect import detect_language
    lang, _ = detect_language("what does namaste mean")
    # Single "namaste" hit isn't enough to trigger Tier 2.
    assert lang == "en"


def test_detect_what_does_namaste_mean_with_devanagari():
    """If a single Devanagari character appears, Tier 1 wins instantly.
    Documented behaviour: one native-script codepoint is sufficient because
    English sentences essentially never contain Devanagari accidentally."""
    from utils.lang_detect import detect_language
    assert detect_language("what does नमस्ते mean") == ("hi", "script")


# ── Single-keyword does not trigger Tier 2 ────────────────────────────────


def test_single_keyword_match_not_enough_for_tier2():
    """One Hinglish token alone must not flip detection — conservative by
    design to avoid false positives on English sentences that use a loan word."""
    from utils.lang_detect import detect_language
    lang, method = detect_language("the food was bahut good")
    # "bahut" alone shouldn't trigger; result depends on Tier 3.
    assert method != "keyword"


# ── Env override: disable Tier 3 ──────────────────────────────────────────


def test_lang_detect_disable_fasttext_env(monkeypatch):
    """LANG_DETECT_DISABLE_FASTTEXT=1 must skip Tier 3 entirely → English
    queries fall to ("en", "default")."""
    monkeypatch.setenv("LANG_DETECT_DISABLE_FASTTEXT", "1")
    mod = _fresh_module()
    assert mod.detect_language("how to bake sourdough bread") == ("en", "default")
    # Tier 1 and Tier 2 still work with Tier 3 disabled.
    assert mod.detect_language("नमस्ते") == ("hi", "script")
    assert mod.detect_language("kya haal hai bhai") == ("hi", "keyword")


# ── Integration: agent.search._detect_language delegates correctly ────────


def test_agent_search_detect_language_returns_hinglish():
    """The existing _detect_language wrapper must now classify Hinglish as 'hi'."""
    from agent.search import _detect_language
    assert _detect_language("kya haal hai bhai") == "hi"
    assert _detect_language("epdi iruka anna") == "ta"
    assert _detect_language("kemon acho tumi") == "bn"


def test_provider_router_detect_query_script_hinglish_is_indic():
    from utils.provider_router import _detect_query_script
    assert _detect_query_script("kya haal hai bhai") == "indic"
    assert _detect_query_script("What is the RBI repo rate?") == "english"
