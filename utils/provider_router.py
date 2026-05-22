"""
Multi-provider LLM router.
  plan()            → Groq  Llama 3.3 70B  max_tokens=200
  call_groq()       → Groq  Llama 3.3 70B  (conflict detection, rolling summary)
  synthesize()      → Gemini 2.5 Flash      max_tokens=1500 (streams)
  judge()           → GitHub Models GPT-4o-mini max_tokens=400

Synthesis falls back to OpenRouter DeepSeek R1 on ResourceExhausted.
"""
from __future__ import annotations

import logging
import os
import re as _re
from typing import AsyncIterator, Optional

from tenacity import retry, stop_after_attempt, wait_exponential
from utils.retry_helpers import wait_retry_after_or_exponential

import json as _json

from agent.models import PlannerOutput, QueryIntent, TypedQuery
from utils.circuit_breaker import CircuitOpenError, breaker
from utils.key_rotation import KeyRotator
from utils.prompt_registry import PROMPT_REGISTRY

# Module-level rotator shared by every Groq caller (plan, call_groq, judge).
# Single instance → shared throttle state across the process.
_GROQ_ROTATOR = KeyRotator(
    "groq", legacy_var="GROQ_API_KEY", multi_var="GROQ_API_KEYS"
)

# Same pattern for Gemini — shared by synth + structured-output planner +
# health probe so they all draw from the same throttle-aware key pool.
_GEMINI_ROTATOR = KeyRotator(
    "gemini", legacy_var="GEMINI_API_KEY", multi_var="GEMINI_API_KEYS"
)

# Cerebras multi-key rotation — shared by planner + judge + short calls +
# streaming synth + health probe. Same throttle-aware pool design as Groq/Gemini.
_CEREBRAS_ROTATOR = KeyRotator(
    "cerebras", legacy_var="CEREBRAS_API_KEY", multi_var="CEREBRAS_API_KEYS"
)


def _gemini_429_retry_after_s(exc: Exception) -> float | None:
    """Extract Retry-After from a Gemini ClientError-style exception. Lenient
    across SDK versions: checks ``response.headers`` and a few common attrs."""
    ra = _extract_retry_after_s(exc)
    if ra is not None:
        return ra
    # google-genai sometimes exposes the status payload as ``e.details`` or
    # ``e.args[0]`` dict. Best-effort only — never raise from this helper.
    try:
        details = getattr(exc, "details", None) or getattr(exc, "args", [None])[0]
        if isinstance(details, dict):
            for k in ("retry_after", "retryAfter", "Retry-After"):
                v = details.get(k) if hasattr(details, "get") else None
                if v is not None:
                    return float(v)
    except Exception:
        return None
    return None


def _is_gemini_429(exc: Exception) -> bool:
    """Detect 429 across google-genai SDK versions. Checks ``.code``,
    ``.status_code``, ``response.status_code``, and message text."""
    for attr in ("code", "status_code"):
        v = getattr(exc, attr, None)
        if v == 429:
            return True
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    msg = str(exc).lower()
    if "429" in msg or "resource_exhausted" in msg or "rate limit" in msg:
        return True
    return False


def _extract_retry_after_s(exc: Exception) -> float | None:
    """Pull a Retry-After (seconds) value from an SDK exception when present."""
    try:
        resp = getattr(exc, "response", None)
        if resp is not None:
            headers = getattr(resp, "headers", None)
            if headers:
                ra = headers.get("Retry-After") or headers.get("retry-after")
                if ra:
                    return float(ra)
    except Exception:
        return None
    return None

# Audit M3: strip ASCII control characters (incl. newlines, tabs) from
# planner-derived text that gets spliced into LLM prompts. Defends against
# prompt-format injection from a malicious or hallucinated planner output.
_CTRL_CHAR_RE = _re.compile(r"[\x00-\x1f\x7f]")
# Importing failure_policy registers all breakers at module import time.
from utils import failure_policy as _failure_policy  # noqa: F401

logger = logging.getLogger(__name__)

_GROQ_MODEL = "llama-3.3-70b-versatile"
_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_GITHUB_MODEL = "gpt-4o-mini"
_OPENROUTER_MODEL = "deepseek/deepseek-r1"
_SARVAM_BASE_URL = "https://api.sarvam.ai/v1"

# ── Indic-script detection (mirrors agent.search._detect_language) ─────────
# Duplicated locally to keep this module's import graph clean (it must not
# import agent.* — would create a cycle since search/orchestrator import us).
_DEVANAGARI_RE = _re.compile(r"[ऀ-ॿ]")
_TAMIL_RE = _re.compile(r"[஀-௿]")
_BENGALI_RE = _re.compile(r"[ঀ-৿]")


def _detect_query_script(text: str) -> str:
    """Return ``"indic"`` for any hi/ta/bn detection (script, keyword, or
    statistical), ``"english"`` otherwise.

    Delegates to ``utils.lang_detect`` so Hinglish ("kya haal hai") and
    romanized Tamil/Bengali ("epdi iruka", "kemon acho") now correctly route
    Sarvam-first instead of being mis-classified as English. The local
    Devanagari/Tamil/Bengali regexes are retained above as a fast-path for
    callers that may still import them, but are no longer the source of
    truth here."""
    if not text:
        return "english"
    from utils.lang_detect import detect_language as _ld
    lang, _method = _ld(text)
    return "indic" if lang in {"hi", "ta", "bn"} else "english"


# ── Run-metadata propagation ──────────────────────────────────────────────
# The orchestrator owns ``run_metadata``. The router decides chain order /
# planner provider / verifier provider but doesn't see the dict. We stamp
# decisions into ContextVars so concurrent orchestrator-driven requests
# (e.g. ``asyncio.gather`` of two /research calls, or the eval runner
# interleaving questions) each observe their OWN attribution. ContextVar
# values set inside the same task propagate back to the caller after
# ``await`` returns; only setters in *child* tasks would not.
import contextvars as _contextvars

_LAST_SYNTH_CHAIN_VAR: _contextvars.ContextVar[list[str] | None] = _contextvars.ContextVar(
    "provider_router.last_synth_chain", default=None
)
_LAST_PLANNER_PROVIDER_VAR: _contextvars.ContextVar[str | None] = _contextvars.ContextVar(
    "provider_router.last_planner_provider", default=None
)


def get_last_synth_chain() -> list[str] | None:
    """Return the chain order that the most recent ``synthesize()`` decided on."""
    return _LAST_SYNTH_CHAIN_VAR.get()


def get_last_planner_provider() -> str | None:
    """Return ``"cerebras" | "groq" | "gemini" | "fallback"`` for the most recent ``plan()`` call."""
    return _LAST_PLANNER_PROVIDER_VAR.get()


# ── Synthesis system prompt (verbatim from spec Section 2.9) ──────────────
SYNTHESIS_SYSTEM_PROMPT = PROMPT_REGISTRY["synthesizer"]["system"]

# ── Planning prompt template (from spec Section 2.9) ─────────────────────
PLANNING_PROMPT_TEMPLATE = PROMPT_REGISTRY["planner"]["template"]


_VALID_SOURCE_TYPES = {"news", "academic", "official", "wiki", "forum"}


def _fallback_planner(query: str) -> PlannerOutput:
    # Fallback represents low signal from the planner: triggers V3.2 second-hop eligibility.
    # Phase 1.875: enriched fields explicitly defaulted (Pydantic defaults
    # would suffice but stamping them keeps the fallback shape unambiguous).
    return PlannerOutput(
        strategy="Direct retrieval fallback",
        queries=[TypedQuery(text=query, intent=QueryIntent.PRIMARY)],
        confidence="low",
        time_sensitivity="static",
        expected_source_types=[],
        difficulty="medium",
        ambiguity_flag=False,
        success_criteria=[],
    )


def parse_planner_output(raw: str, query: str) -> PlannerOutput:
    """Parse typed planner output with deterministic fallback."""
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return _fallback_planner(query)
    try:
        data = _json.loads(raw[start:end])
    except Exception:
        return _fallback_planner(query)

    raw_queries = data.get("queries") or []
    typed: list[TypedQuery] = []
    for item in raw_queries:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        intent_str = (item.get("intent") or "").strip().lower()
        if not text or not intent_str:
            continue
        try:
            intent = QueryIntent(intent_str)
        except ValueError:
            return _fallback_planner(query)
        rationale = item.get("rationale")
        if rationale is not None and not isinstance(rationale, str):
            rationale = None
        typed.append(TypedQuery(text=text, intent=intent, rationale=rationale))

    if not typed:
        return _fallback_planner(query)

    strategy = (data.get("strategy") or "").strip() or "Direct retrieval fallback"
    confidence_raw = (data.get("confidence") or "").strip().lower()
    confidence = confidence_raw if confidence_raw in {"low", "medium", "high"} else "medium"

    # Phase 1.875: enriched fields. Every parser branch must be
    # never-raises — missing or malformed values fall back to defaults.
    ts_raw = (data.get("time_sensitivity") or "").strip().lower()
    time_sensitivity = ts_raw if ts_raw in {"live", "recent", "static"} else "static"

    diff_raw = (data.get("difficulty") or "").strip().lower()
    difficulty = diff_raw if diff_raw in {"easy", "medium", "hard"} else "medium"

    src_raw = data.get("expected_source_types") or []
    expected_source_types: list[str] = []
    if isinstance(src_raw, list):
        for s in src_raw:
            if isinstance(s, str) and s.strip().lower() in _VALID_SOURCE_TYPES:
                expected_source_types.append(s.strip().lower())

    amb_raw = data.get("ambiguity_flag")
    ambiguity_flag = bool(amb_raw) if isinstance(amb_raw, bool) else False

    # Audit M3: success_criteria gets injected verbatim into the synthesizer
    # prompt. Strip control characters (incl. newlines) to prevent prompt-
    # format injection where a malicious item could fake new prompt sections.
    # Cap to 200 chars per item and 5 items total (matches the prompt's
    # implicit budget).
    crit_raw = data.get("success_criteria") or []
    success_criteria: list[str] = []
    if isinstance(crit_raw, list):
        _ctrl_re = _CTRL_CHAR_RE
        for c in crit_raw:
            if isinstance(c, str) and c.strip():
                cleaned = _ctrl_re.sub("", c).strip()[:200]
                if cleaned:
                    success_criteria.append(cleaned)
        success_criteria = success_criteria[:5]

    try:
        return PlannerOutput(
            strategy=strategy,
            queries=typed[:4],
            confidence=confidence,  # type: ignore[arg-type]
            time_sensitivity=time_sensitivity,  # type: ignore[arg-type]
            expected_source_types=expected_source_types,  # type: ignore[arg-type]
            difficulty=difficulty,  # type: ignore[arg-type]
            ambiguity_flag=ambiguity_flag,
            success_criteria=success_criteria,
        )
    except Exception:
        # Never raise from the parser — defaults always win.
        return PlannerOutput(
            strategy=strategy,
            queries=typed[:4],
            confidence=confidence,  # type: ignore[arg-type]
        )


_CEREBRAS_AVAILABLE_MODELS_HINT = (
    "llama3.1-8b, qwen-3-235b-a22b-instruct-2507, zai-glm-4.7, gpt-oss-120b"
)


def _warn_cerebras_404_if_relevant(exc: Exception, model: str) -> None:
    """If exc looks like a Cerebras 404 (model not found), log a clear hint.

    Cerebras gates models per-account, so the default we ship may not be on
    every key. Surface the working models from CLAUDE.md inline rather than
    making operators dig through cloud.cerebras.ai/models.
    """
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    msg = str(exc).lower()
    if status == 404 or "does not exist" in msg or "do not have access" in msg:
        logger.warning(
            "Cerebras model '%s' not available on this account. "
            "Try CEREBRAS_MODEL=llama3.1-8b or check cloud.cerebras.ai/models. "
            "Known-good options: %s",
            model, _CEREBRAS_AVAILABLE_MODELS_HINT,
        )


async def _plan_with_cerebras(prompt: str) -> str:
    """Cerebras-backed planner call. Returns the raw text response.

    8K-context cap is enforced upstream by the caller checking prompt length.
    5-second timeout — the planner is on the critical path, so we'd rather
    fall back to Groq than wait.

    Default model is ``llama3.1-8b`` — small, fast, and broadly available on
    free Cerebras accounts. Set ``CEREBRAS_MODEL`` to override (e.g.
    ``qwen-3-235b-a22b-instruct-2507``, ``zai-glm-4.7``, ``gpt-oss-120b``).
    """
    from openai import AsyncOpenAI
    api_key = _CEREBRAS_ROTATOR.next_key()
    if not api_key:
        raise RuntimeError("No CEREBRAS_API_KEY / CEREBRAS_API_KEYS configured")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"),
        timeout=5.0,
    )
    model = os.environ.get("CEREBRAS_MODEL", "llama3.1-8b")
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
    except Exception as e:
        _warn_cerebras_404_if_relevant(e, model)
        status = getattr(e, "status_code", None) or getattr(
            getattr(e, "response", None), "status_code", None
        )
        if status == 429:
            _CEREBRAS_ROTATOR.mark_throttled(
                api_key, retry_after_s=_extract_retry_after_s(e)
            )
        raise
    _CEREBRAS_ROTATOR.mark_success(api_key)
    return (resp.choices[0].message.content or "").strip()


async def _plan_with_gemini_structured(prompt: str) -> str:
    """Gemini structured-output planner. Uses response_mime_type=application/json
    with the ``PlannerOutput`` Pydantic schema. Opt-in via ``PLANNER_PROVIDER=gemini``."""
    from google import genai
    from google.genai import types

    api_key = _GEMINI_ROTATOR.next_key()
    if not api_key:
        raise RuntimeError("No GEMINI_API_KEY / GEMINI_API_KEYS configured")
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=PlannerOutput,
        max_output_tokens=600,
        temperature=0.3,
    )
    try:
        resp = await client.aio.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
            config=config,
        )
    except Exception as e:
        if _is_gemini_429(e):
            _GEMINI_ROTATOR.mark_throttled(api_key, retry_after_s=_gemini_429_retry_after_s(e))
            # Tell the health probe layer Gemini is sick so the next
            # dashboard hit re-probes ahead of the TTL.
            try:
                from utils.provider_health import invalidate
                invalidate("gemini")
            except Exception:
                pass
        raise
    _GEMINI_ROTATOR.mark_success(api_key)
    return (resp.text or "").strip()


@breaker("groq")
@retry(wait=wait_retry_after_or_exponential(min=2, max=30), stop=stop_after_attempt(3), reraise=True)
async def _plan_with_groq(prompt: str) -> str:
    """Groq-backed planner call. Returns the raw text response."""
    import groq as groq_sdk
    key = _GROQ_ROTATOR.next_key()
    if not key:
        raise RuntimeError("No GROQ_API_KEY / GROQ_API_KEYS configured")
    client = groq_sdk.AsyncGroq(api_key=key)
    try:
        resp = await client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
    except groq_sdk.RateLimitError as e:
        _GROQ_ROTATOR.mark_throttled(key, retry_after_s=_extract_retry_after_s(e))
        try:
            from utils.provider_health import invalidate
            invalidate("groq")
        except Exception:
            pass
        raise
    _GROQ_ROTATOR.mark_success(key)
    return (resp.choices[0].message.content or "").strip()


async def plan(query: str, prior_summary: str = "No prior context.") -> PlannerOutput:
    """Generate strategy + search queries.

    Provider selection (PLANNER_PROVIDER env, default ``auto``):
      - ``auto``: Cerebras when key present + prompt fits in ~6K tokens, else Groq.
      - ``cerebras``: force Cerebras (errors fall back to Groq).
      - ``groq``: force Groq.
      - ``gemini``: Gemini structured-output mode (opt-in, errors fall back to Groq).

    The selected provider is stamped into ``_LAST_PLANNER_PROVIDER`` so the
    orchestrator can record ``run_metadata["planner_provider"]``.
    Never raises — degrades to the fallback parser on any error.
    """
    prompt = PLANNING_PROMPT_TEMPLATE.format(query=query, prior_summary=prior_summary)
    pref = os.environ.get("PLANNER_PROVIDER", "auto").lower().strip()

    raw: str | None = None
    chosen: str = "groq"

    # ── Try Cerebras (auto or forced) ─────────────────────────────────────
    if pref in ("auto", "cerebras") and os.environ.get("CEREBRAS_API_KEY"):
        from utils.token_counter import count_tokens
        # 8K cap minus 2K headroom for system + output = 6K usable prompt.
        if count_tokens(prompt) <= 6000:
            try:
                raw = await _plan_with_cerebras(prompt)
                chosen = "cerebras"
            except Exception as e:
                logger.warning("[planner] Cerebras failed (%s), falling back to Groq", e)
                raw = None

    # ── Try Gemini structured-output mode (opt-in only) ───────────────────
    if raw is None and pref == "gemini" and _GEMINI_ROTATOR.has_keys():
        try:
            raw = await _plan_with_gemini_structured(prompt)
            chosen = "gemini"
        except Exception as e:
            logger.warning("[planner] Gemini structured failed (%s), falling back to Groq", e)
            raw = None

    # ── Fall back to Groq (default + forced + final fallback) ─────────────
    if raw is None:
        try:
            raw = await _plan_with_groq(prompt)
            # ``chosen`` stays "groq" unless Cerebras/Gemini succeeded above.
            # If we got here after a Cerebras/Gemini attempt failed, mark as
            # ``fallback`` to surface the degradation in run_metadata.
            chosen = "fallback" if (
                pref in ("cerebras", "gemini")
                or (pref == "auto" and os.environ.get("CEREBRAS_API_KEY"))
            ) and chosen == "groq" else "groq"
        except Exception as e:
            logger.error("[planner] Groq also failed (%s); using deterministic fallback", e)
            _LAST_PLANNER_PROVIDER_VAR.set("fallback")
            return _fallback_planner(query)

    _LAST_PLANNER_PROVIDER_VAR.set(chosen)
    parsed = parse_planner_output(raw, query)
    if parsed.strategy == "Direct retrieval fallback":
        logger.warning("Plan parse failed, using fallback. Raw output: %s", raw)
    return parsed


@breaker("groq")
@retry(wait=wait_retry_after_or_exponential(min=2, max=30), stop=stop_after_attempt(3), reraise=True)
async def call_groq(prompt: str, max_tokens: int = 200) -> str:
    """Generic Groq call for conflict detection, rolling summary, etc."""
    import groq as groq_sdk
    key = _GROQ_ROTATOR.next_key()
    if not key:
        raise RuntimeError("No GROQ_API_KEY / GROQ_API_KEYS configured")
    client = groq_sdk.AsyncGroq(api_key=key)
    try:
        resp = await client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
    except groq_sdk.RateLimitError as e:
        _GROQ_ROTATOR.mark_throttled(key, retry_after_s=_extract_retry_after_s(e))
        raise
    _GROQ_ROTATOR.mark_success(key)
    from utils import provider_usage
    usage = getattr(resp, "usage", None)
    await provider_usage.record(
        "groq",
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )
    return resp.choices[0].message.content.strip()


async def call_cerebras(prompt: str, max_tokens: int = 200) -> str:
    """Generic Cerebras call for short-prompt tasks (conflict probe, follow-up gen).

    8K context cap upstream; callers should verify prompt length. Raises on
    missing key or API failure — caller falls back to Groq.
    """
    from openai import AsyncOpenAI
    api_key = _CEREBRAS_ROTATOR.next_key()
    if not api_key:
        raise RuntimeError("No CEREBRAS_API_KEY / CEREBRAS_API_KEYS configured")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"),
        timeout=10.0,
    )
    model = os.environ.get("CEREBRAS_MODEL", "llama3.1-8b")
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
    except Exception as e:
        _warn_cerebras_404_if_relevant(e, model)
        status = getattr(e, "status_code", None) or getattr(
            getattr(e, "response", None), "status_code", None
        )
        if status == 429:
            _CEREBRAS_ROTATOR.mark_throttled(
                api_key, retry_after_s=_extract_retry_after_s(e)
            )
        raise
    _CEREBRAS_ROTATOR.mark_success(api_key)
    return (resp.choices[0].message.content or "").strip()


def _sanitize_conflict_string(s: str, max_chars: int = 300) -> str:
    """FIX 2: sanitize LLM-sourced conflict text before interpolation into the
    synthesizer's mandatory-instruction block. Strips control chars, angle
    brackets, lines that look like prompt-injection attempts, and truncates.
    """
    if not s:
        return ""
    # Drop control chars and angle brackets
    cleaned = "".join(
        ch for ch in s if (ch == "\n" or ch == "\t" or (ord(ch) >= 32 and ch not in "<>"))
    )
    injection_markers = (
        "system:",
        "assistant:",
        "ignore previous",
        "disregard",
        "you are now",
    )
    safe_lines: list[str] = []
    for line in cleaned.splitlines():
        lowered = line.lower()
        if any(marker in lowered for marker in injection_markers):
            continue
        safe_lines.append(line)
    flattened = " ".join(safe_lines).strip()
    if len(flattened) > max_chars:
        flattened = flattened[:max_chars].rstrip() + "…"
    return flattened


def _build_disagreement_block(conflict_result: object | None) -> str:
    """V2.2 + B4 + DRAGged: enumerate non-temporal contradictions for the
    synthesizer and demand a structured Markdown disagreement matrix.
    Heading + column labels are chosen based on the DRAGged-into-Conflict
    `kind` taxonomy (Cattan et al. 2025, arXiv:2506.08500):

      - self        → "Internal contradiction within a source" + columns
                      "First mention / Second mention" — the *same* source
                      contradicts itself.
      - pair        → "Sources disagree on this" + "Source A / Source B" —
                      the canonical two-source disagreement.
      - conditional → "Sources agree under qualifier" + "Source A /
                      Source B / Qualifier" — sources only disagree when
                      a qualifier (year, region, sub-domain) is missing.

    When contradictions are mixed-kind we group by kind and emit one
    sub-table per group, so a `pair` and a `self` finding don't get
    incorrectly merged under one generic heading.
    """
    if conflict_result is None or not conflict_result.has_conflict:
        return ""
    real = [c for c in conflict_result.contradictions if not c.is_temporal_evolution]
    if not real:
        return ""

    lines = [
        "<cross_source_disagreement>",
        "The retrieval found conflicting claims you MUST present neutrally:",
    ]
    # Sanitize attacker-influenced LLM output before interpolation.
    sanitized: list[tuple[object, str, str, str]] = []
    for c in real:
        sanitized.append(
            (
                c,
                _sanitize_conflict_string(c.claim),
                _sanitize_conflict_string(c.position_a),
                _sanitize_conflict_string(c.position_b),
            )
        )
    for c, s_claim, s_pos_a, s_pos_b in sanitized:
        ids_a = ", ".join(c.doc_ids_a)
        ids_b = ", ".join(c.doc_ids_b)
        lines.append(
            f'- On "{s_claim}": [{ids_a}] state "{s_pos_a}", '
            f'while [{ids_b}] states "{s_pos_b}".'
        )
    lines.append(
        '  Do NOT pick a winner. Use phrasing appropriate to the contradiction kind '
        '(see the table headings below).'
    )
    lines.append("")
    lines.append("MANDATORY OUTPUT FORMAT — render the disagreement as a Markdown")
    lines.append("table inside the answer (verbatim shape, one row per conflict):")
    lines.append("")

    # Group sanitized contradictions by kind. Defaults to "pair" for legacy
    # rows whose kind wasn't populated by the probe.
    by_kind: dict[str, list[tuple[object, str, str, str]]] = {}
    for c, sc, sa, sb in sanitized:
        k = getattr(c, "kind", None) or "pair"
        by_kind.setdefault(k, []).append((c, sc, sa, sb))

    _HEADINGS = {
        "self": "**Internal contradiction within a source:**",
        "pair": "**Sources disagree on this:**",
        "conditional": "**Sources agree under qualifier:**",
    }
    _COL_HEADERS = {
        "self": "| Claim | First mention | Second mention |\n|---|---|---|",
        "pair": "| Claim | Source A | Source B |\n|---|---|---|",
        "conditional": "| Claim | Source A | Source B | Qualifier |\n|---|---|---|---|",
    }
    # Stable kind order matching severity (self first, then pair, then conditional).
    for k in ("self", "pair", "conditional"):
        rows = by_kind.get(k)
        if not rows:
            continue
        lines.append(_HEADINGS.get(k, _HEADINGS["pair"]))
        lines.append("")
        lines.append(_COL_HEADERS.get(k, _COL_HEADERS["pair"]))
        for c, s_claim, s_pos_a, s_pos_b in rows:
            first_a = c.doc_ids_a[0] if c.doc_ids_a else ""
            first_b = c.doc_ids_b[0] if c.doc_ids_b else ""
            if k == "conditional":
                qualifier = _sanitize_conflict_string(
                    getattr(c, "qualifier", "") or "—"
                )
                lines.append(
                    f"| {s_claim}: A={s_pos_a} / B={s_pos_b} "
                    f"| [{first_a}] | [{first_b}] | {qualifier} |"
                )
            elif k == "self":
                # First-mention / Second-mention — same source on both sides,
                # so emit the single doc_id reference in each cell.
                source_marker = f"[{first_a or first_b}]"
                lines.append(
                    f"| {s_claim}: first={s_pos_a} / then={s_pos_b} "
                    f"| {source_marker} | {source_marker} |"
                )
            else:
                lines.append(
                    f"| {s_claim}: A={s_pos_a} / B={s_pos_b} "
                    f"| [{first_a}] | [{first_b}] |"
                )
        lines.append("")
    lines.append("Cite each source using a bare [doc_N] marker inside the table")
    lines.append("cells — the post-processor expands them to [Title — domain](URL).")
    lines.append("The table is REQUIRED whenever this block is present; rendering")
    lines.append("only prose without the table is a failure.")
    lines.append("</cross_source_disagreement>")
    return "\n".join(lines)


async def synthesize(
    query: str,
    context_xml: str,
    doc_map: dict[str, tuple[str, str, str]],
    history_text: str = "",
    conflict_note: Optional[str] = None,
    conflict_result=None,
) -> AsyncIterator[tuple[str, int, int]]:
    """
    Streams synthesis from Gemini 2.5 Flash.
    Falls back to OpenRouter DeepSeek R1 on ResourceExhausted.
    Yields (text_chunk, prompt_tokens, completion_tokens) — token counts sent on final chunk.
    """
    # Build doc listing for the prompt
    doc_listing = "\n".join(
        f"  [{doc_id}]: {title} — {domain} ({url})"
        for doc_id, (title, url, domain) in doc_map.items()
    )
    conflict_instruction = ""
    if conflict_note:
        conflict_instruction = f"\n\nCONFLICT DETECTED: {conflict_note}\nYou MUST present both sides explicitly."

    disagreement_block = _build_disagreement_block(conflict_result)
    if disagreement_block:
        disagreement_block = "\n\n" + disagreement_block

    user_prompt = f"""Available documents:
{doc_listing}

{context_xml}
{conflict_instruction}{disagreement_block}

Research question: {query}

{("Prior conversation context:\n" + history_text) if history_text else ""}

Answer the research question using only the documents above. Cite every factual claim with [doc_N]."""

    # V3.9: ordered fallback chain. The first entry is determined by
    # SYNTH_PROVIDER; the rest are tried in order on 429 / breaker-open /
    # context-overflow / connection-refused. Each step is independent:
    # failure of one provider doesn't prevent the next from being tried.
    primary = os.environ.get("SYNTH_PROVIDER", "gemini").lower()
    chain_order = {
        "gemini":     ["gemini", "sarvam", "openrouter", "cerebras", "ollama"],
        "sarvam":     ["sarvam", "gemini", "openrouter", "cerebras", "ollama"],
        "openrouter": ["openrouter", "gemini", "sarvam", "cerebras", "ollama"],
        "cerebras":   ["cerebras", "gemini", "sarvam", "openrouter", "ollama"],
        "ollama":     ["ollama", "gemini", "sarvam", "openrouter", "cerebras"],
    }
    chain = chain_order.get(primary, ["gemini", "sarvam", "openrouter", "cerebras", "ollama"])

    # ── Indic auto-routing: Sarvam first when query script is Indic ───────
    # The assignment dataset is multilingual; Sarvam's models (sarvam-m,
    # sarvam-30b) are tuned for Indic languages. Whenever a Devanagari / Tamil
    # / Bengali query arrives AND SARVAM_API_KEY is configured AND the auto-
    # routing flag is on (default), promote Sarvam to the head of the chain.
    indic_auto = os.environ.get("SARVAM_INDIC_AUTO", "1").strip() not in {"0", "false", "False", ""}
    if (
        indic_auto
        and os.environ.get("SARVAM_API_KEY")
        and _detect_query_script(query) == "indic"
    ):
        chain = ["sarvam", "gemini", "openrouter", "cerebras", "ollama"]
        logger.info("Sarvam-primary routing for Indic query")

    # Stamp the chain so the orchestrator can record it in run_metadata.
    _LAST_SYNTH_CHAIN_VAR.set(list(chain))
    synth_fns = {
        "gemini":     _synthesize_gemini,
        "sarvam":     _synthesize_sarvam,
        "openrouter": _synthesize_openrouter,
        "cerebras":   _synthesize_cerebras,
        "ollama":     _synthesize_ollama,
    }

    last_err: Exception | None = None
    for step, name in enumerate(chain):
        fn = synth_fns[name]
        # Pre-flight: skip providers with no key configured (except ollama which
        # uses a placeholder key and gates on connection reachability instead).
        key_env = {
            "gemini": "GEMINI_API_KEY",
            "sarvam": "SARVAM_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "cerebras": "CEREBRAS_API_KEY",
            "ollama": None,
        }[name]
        # Gemini may use either GEMINI_API_KEY (legacy single) or GEMINI_API_KEYS
        # (multi-key). Defer to the rotator so multi-key-only setups still pass.
        if name == "gemini":
            if not _GEMINI_ROTATOR.has_keys():
                logger.debug("Skipping gemini: no keys configured")
                continue
        elif key_env is not None and not os.environ.get(key_env):
            logger.debug("Skipping %s: %s not set", name, key_env)
            continue
        try:
            yielded_any = False
            async for chunk in fn(user_prompt):
                yielded_any = True
                yield chunk
            if yielded_any:
                if step > 0:
                    logger.info("[synth] fallback succeeded: provider=%s (step %d)", name, step)
                return
        except CircuitOpenError as e:
            last_err = e
            logger.warning("[synth] %s breaker open, trying next: %s", name, e)
        except Exception as e:
            last_err = e
            logger.warning("[synth] %s failed (%s), trying next: %s", name, type(e).__name__, e)

    logger.error("[synth] all providers exhausted; last error: %s", last_err)
    yield (
        "[Synthesis temporarily unavailable: all configured providers are "
        "currently rate-limited or unreachable. Please retry in a few seconds.]",
        0,
        0,
    )


@breaker("gemini")
async def _synthesize_gemini(user_prompt: str) -> AsyncIterator[tuple[str, int, int]]:
    from google import genai
    from google.genai import types

    api_key = _GEMINI_ROTATOR.next_key()
    if not api_key:
        raise RuntimeError("No GEMINI_API_KEY / GEMINI_API_KEYS configured")
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYNTHESIS_SYSTEM_PROMPT,
        max_output_tokens=1500,
        temperature=0.2,
    )

    prompt_tokens = 0
    completion_tokens = 0

    candidate_models = [
        _GEMINI_MODEL,
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]
    seen: set[str] = set()
    last_err: Exception | None = None
    for model_name in candidate_models:
        if model_name in seen:
            continue
        seen.add(model_name)
        try:
            async for response in await client.aio.models.generate_content_stream(
                model=model_name,
                contents=user_prompt,
                config=config,
            ):
                if response.text:
                    yield (response.text, 0, 0)
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    prompt_tokens = response.usage_metadata.prompt_token_count or 0
                    completion_tokens = response.usage_metadata.candidates_token_count or 0
            # Final sentinel with token counts
            from utils import provider_usage
            await provider_usage.record(
                "gemini",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            _GEMINI_ROTATOR.mark_success(api_key)
            yield ("", prompt_tokens, completion_tokens)
            return
        except Exception as e:
            last_err = e
            if _is_gemini_429(e):
                _GEMINI_ROTATOR.mark_throttled(
                    api_key, retry_after_s=_gemini_429_retry_after_s(e)
                )
            if "not found" in str(e).lower() or "404" in str(e):
                logger.warning("Gemini model unavailable (%s): %s", model_name, e)
                continue
            raise

    if last_err is not None:
        raise last_err
    raise RuntimeError("No Gemini model candidates available")


@breaker("cerebras")
async def _synthesize_cerebras(user_prompt: str) -> AsyncIterator[tuple[str, int, int]]:
    """Cerebras Cloud — extremely fast Llama inference (~2000 tokens/s).

    Important caveat (per CLAUDE.md): Cerebras has an 8K context cap. Our
    synthesis budget can be up to 12K (system + history + web context).
    For prompts that exceed 8K we silently fall through to the next provider
    in the chain by raising — the caller's auto-promote logic will pick up
    the next viable synthesizer.

    Useful when: (a) Gemini and OpenRouter are both 429/breaker-open, (b) the
    prompt actually fits in 8K (short queries with minimal history). Otherwise
    a no-op fallback that yields to the next provider.
    """
    from utils.token_counter import count_tokens
    if count_tokens(user_prompt) + 1500 > 7800:  # leave headroom for system + output
        raise RuntimeError("Cerebras 8K context cap exceeded; skipping to next provider")
    from openai import AsyncOpenAI
    api_key = _CEREBRAS_ROTATOR.next_key()
    if not api_key:
        raise RuntimeError("No CEREBRAS_API_KEY / CEREBRAS_API_KEYS configured")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"),
        timeout=60.0,
    )
    model = os.environ.get("CEREBRAS_MODEL", "llama3.1-8b")
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        resp = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=1500, stream=True,
        )
    except Exception as e:
        _warn_cerebras_404_if_relevant(e, model)
        status = getattr(e, "status_code", None) or getattr(
            getattr(e, "response", None), "status_code", None
        )
        if status == 429:
            _CEREBRAS_ROTATOR.mark_throttled(
                api_key, retry_after_s=_extract_retry_after_s(e)
            )
        raise
    _CEREBRAS_ROTATOR.mark_success(api_key)
    prompt_tokens = 0
    completion_tokens = 0
    async for chunk in resp:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield (delta, 0, 0)
        if hasattr(chunk, "usage") and chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens or 0
            completion_tokens = chunk.usage.completion_tokens or 0
    from utils import provider_usage
    await provider_usage.record(
        "cerebras", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )
    yield ("", prompt_tokens, completion_tokens)


@breaker("ollama")
async def _synthesize_ollama(user_prompt: str) -> AsyncIterator[tuple[str, int, int]]:
    """Ollama — local OpenAI-compatible inference server.

    Designed for the "all cloud providers are exhausted" fallback. Default
    model is `llama3.1:8b` which is small enough to run on a typical laptop
    but produces respectable synthesis. The base URL defaults to
    http://localhost:11434/v1, which works whenever Ollama is running
    locally; on HF Spaces this won't connect (no Ollama process), which is
    fine — the next provider in the chain will be tried instead.

    No quota tracking. By definition unlimited (your machine's RAM is the limit).
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key="ollama",  # Ollama ignores the key; required by the SDK
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        timeout=120.0,
    )
    model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    resp = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=1500, stream=True,
    )
    prompt_tokens = 0
    completion_tokens = 0
    async for chunk in resp:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield (delta, 0, 0)
        if hasattr(chunk, "usage") and chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens or 0
            completion_tokens = chunk.usage.completion_tokens or 0
    from utils import provider_usage
    await provider_usage.record(
        "ollama", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )
    yield ("", prompt_tokens, completion_tokens)


@breaker("openrouter")
async def _synthesize_openrouter(user_prompt: str) -> AsyncIterator[tuple[str, int, int]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
        timeout=90.0,
    )
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    resp = await client.chat.completions.create(
        model=_OPENROUTER_MODEL,
        messages=messages,
        max_tokens=1500,
        stream=True,
    )
    async for chunk in resp:
        text = chunk.choices[0].delta.content or ""
        if text:
            yield (text, 0, 0)
    yield ("", 0, 0)


async def _stream_sarvam_model(
    user_prompt: str,
    model: str,
) -> AsyncIterator[tuple[str, int, int]]:
    """Open a single Sarvam streaming completion against the given model."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ["SARVAM_API_KEY"],
        base_url=_SARVAM_BASE_URL,
        timeout=90.0,
    )
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1500,
        temperature=0.2,
        stream=True,
    )
    async for chunk in resp:
        # Sarvam follows OpenAI's chunk schema; .delta.content is None on the
        # final chunk where finish_reason lands.
        delta = ""
        try:
            delta = chunk.choices[0].delta.content or ""
        except (AttributeError, IndexError):
            delta = ""
        if delta:
            yield (delta, 0, 0)
    # Sarvam does not currently emit per-stream token counts; orchestrator
    # tolerates 0 prompt/completion totals.
    yield ("", 0, 0)


@breaker("sarvam")
async def _synthesize_sarvam(user_prompt: str) -> AsyncIterator[tuple[str, int, int]]:
    """Sarvam Model API synthesizer (OpenAI-compatible).

    Tries SARVAM_MODEL (default 'sarvam-m'); on any exception retries once with
    'sarvam-30b' before raising. One Tenacity-exhausted call still counts as a
    single breaker failure via the outer @breaker decoration.
    """
    primary = os.environ.get("SARVAM_MODEL", "sarvam-m")
    yielded_any = False
    try:
        async for chunk in _stream_sarvam_model(user_prompt, primary):
            yielded_any = True
            yield chunk
        return
    except Exception as exc:
        # Only fall back if we never produced output — otherwise the consumer
        # has a partial answer and a second stream would duplicate content.
        if primary == "sarvam-30b" or yielded_any:
            raise
        logger.warning(
            "Sarvam model %s failed (%s); retrying once with sarvam-30b", primary, exc
        )
    async for chunk in _stream_sarvam_model(user_prompt, "sarvam-30b"):
        yield chunk


_JUDGE_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "github": {
        "base_url": "https://models.inference.ai.azure.com",
        "api_key_env": "GITHUB_TOKEN",
        "default_model": "gpt-4o-mini",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        # Qwen family — non-overlapping with Gemini synth and GPT-4o-mini
        # secondary, gives the most independent cross-family signal. Free-tier
        # TPD on Cerebras is ~10× Groq's, eliminating the 22-turn ceiling.
        "default_model": "qwen-3-235b-a22b-instruct-2507",
    },
}


@breaker("judge")
@retry(wait=wait_retry_after_or_exponential(min=2, max=30), stop=stop_after_attempt(3), reraise=True)
async def judge(prompt: str) -> str:
    """LLM judge for eval scoring.

    Provider is selected by JUDGE_PROVIDER env var (default 'groq'). The judge
    *must* be a different model family from the synthesis generator to avoid
    self-preference bias. Defaults: synth=Gemini, judge=Groq Llama — cross-family.
    If you set SYNTH_PROVIDER=sarvam (also Llama-based), switch JUDGE_PROVIDER
    to 'github' (GPT-4o-mini) to preserve the invariant.
    """
    from openai import AsyncOpenAI

    provider = os.environ.get("JUDGE_PROVIDER", "groq").lower()
    cfg = _JUDGE_PROVIDERS.get(provider) or _JUDGE_PROVIDERS["groq"]
    model = os.environ.get("JUDGE_MODEL", cfg["default_model"])

    # Groq/Cerebras judges pull from the multi-key rotator so the same key pool
    # covers planner + conflict + judge calls and shares throttle state.
    is_groq = provider == "groq"
    is_cerebras = provider == "cerebras"
    api_key: str | None = None
    if is_groq:
        api_key = _GROQ_ROTATOR.next_key()
        if not api_key:
            raise RuntimeError("No GROQ_API_KEY / GROQ_API_KEYS configured")
    elif is_cerebras:
        api_key = _CEREBRAS_ROTATOR.next_key()
        if not api_key:
            raise RuntimeError(
                "No CEREBRAS_API_KEY / CEREBRAS_API_KEYS configured"
            )
    else:
        api_key = os.environ[cfg["api_key_env"]]

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=cfg["base_url"],
        timeout=60.0,
    )
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.0,
        )
    except Exception as e:
        # OpenAI SDK surfaces 429 as openai.RateLimitError; we treat anything
        # with .status_code == 429 as a throttle to stay SDK-version-tolerant.
        if api_key:
            status = getattr(e, "status_code", None) or getattr(
                getattr(e, "response", None), "status_code", None
            )
            if status == 429:
                if is_groq:
                    _GROQ_ROTATOR.mark_throttled(
                        api_key, retry_after_s=_extract_retry_after_s(e)
                    )
                elif is_cerebras:
                    _CEREBRAS_ROTATOR.mark_throttled(
                        api_key, retry_after_s=_extract_retry_after_s(e)
                    )
        raise
    if api_key:
        if is_groq:
            _GROQ_ROTATOR.mark_success(api_key)
        elif is_cerebras:
            _CEREBRAS_ROTATOR.mark_success(api_key)
    return resp.choices[0].message.content.strip()


async def _claim_verify_with_deepseek(claim: str, snippet: str) -> str:
    """Verify a single claim against an evidence snippet via DeepSeek R1 (OpenRouter).

    Returns the raw text content. The caller parses JSON downstream.

    CRITICAL: DeepSeek R1 returns chain-of-thought in a separate ``reasoning_content``
    field on the message. We deliberately use ONLY ``message.content`` (the
    final answer), never ``reasoning_content`` — to keep CoT out of the SSE
    pipeline and out of the parsed JSON output.
    """
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=20.0,
    )
    prompt = (
        f"CLAIM: {claim}\n"
        f"EVIDENCE SNIPPET: {snippet[:1500]}\n"
        "Does the snippet explicitly support this exact claim "
        "(entities, numbers, dates must match)?\n"
        'JSON only: {"supported": bool, "reasoning": str}'
    )
    resp = await client.chat.completions.create(
        model=_OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.0,
    )
    # NOTE: do NOT read ``reasoning_content`` — that's DeepSeek R1's chain of
    # thought. The visible final answer lives in ``message.content``.
    msg = resp.choices[0].message
    return (getattr(msg, "content", "") or "").strip()


async def rolling_summary(turns_text: str) -> str:
    """Compress old turns into a rolling summary via Groq.

    Phase 1.75: tightened prompt requires structured preservation of
    (a) named entities, (b) dates, (c) decisions, (d) source contradictions.
    """
    prompt = f"""Summarize the following research conversation.

Preserve: (a) named entities, (b) dates, (c) decisions made,
(d) any contradictions noted between sources.

Output as 2-3 short paragraphs (<= 50 words each) separated by '---'.
Do not add any information not present in the conversation.

{turns_text}

Summary:"""
    return await call_groq(prompt, max_tokens=300)


async def compress_history(
    existing_summary: str,
    remaining_turns: list,
    max_tokens: int,
) -> str:
    """Second-pass summary-of-summaries.

    Phase 1.75: called when even the rolling-summary + most-recent-turn
    history still overflows ``ContextBudget.history_budget``. Compresses
    ``existing_summary`` plus ``remaining_turns`` into at most ``max_tokens``
    cl100k tokens while preserving structured signal.

    Falls back to hard truncation if the Groq call fails. Never raises.
    """
    from utils.token_counter import count_tokens, truncate_to_tokens

    turns_text = "\n\n".join(
        f"Q: {getattr(t, 'query', '')}\nA: {getattr(t, 'response', '') or ''}"
        for t in (remaining_turns or [])
    )
    combined = f"[Earlier summary]\n{existing_summary}\n\n[Recent turns]\n{turns_text}".strip()

    prompt = (
        f"Compress the following research session history into <= {max_tokens} tokens "
        "while preserving: (a) named entities, (b) dates, (c) decisions made, "
        "(d) any contradictions noted between sources.\n"
        "Output as 2-3 short paragraphs separated by '---'.\n\n"
        f"{combined}\n\nCompressed history:"
    )

    try:
        compressed = await call_groq(prompt, max_tokens=max(64, max_tokens))
        if compressed and count_tokens(compressed) <= max_tokens:
            return compressed
        # Model returned but overshot the budget: precision-truncate.
        return truncate_to_tokens(compressed or combined, max_tokens)
    except Exception as e:
        logger.warning(
            "compress_history Groq call failed, hard-truncating: %s",
            e,
            extra={"component": "provider_router"},
        )
        return truncate_to_tokens(combined, max_tokens)
