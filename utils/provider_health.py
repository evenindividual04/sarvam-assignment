"""Cached provider health probes.

Endpoint `/health/providers` returns a snapshot of which external providers
are reachable and authenticated. The result is cached in-memory for
HEALTH_CACHE_TTL_S seconds (default 60) so a burst of page visits triggers
exactly one probe set.

Each probe uses the *cheapest possible* call that exercises auth + a real
endpoint: 1-token LLM completions, 1-result searches. Total probe cost per
refresh: ~8 requests, all parallel, ~1–3 seconds wall clock.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

HEALTH_CACHE_TTL_S = int(os.environ.get("HEALTH_CACHE_TTL_S", "60"))
_PROBE_TIMEOUT_S = 8.0


@dataclass(frozen=True)
class ProviderProbe:
    name: str
    role: str  # "search" | "synth" | "planner" | "judge" | "synth-indic"
    # "ok" | "degraded" | "down" | "missing_key" | "not_configured"
    # Backwards-compat: existing clients that check status != "ok" still work.
    # `not_configured` means the provider is intentionally absent (e.g. local
    # Ollama on a remote deployment), so it should not count as a failure.
    status: str
    latency_ms: Optional[int] = None
    detail: str = ""


@dataclass(frozen=True)
class HealthSnapshot:
    checked_at: float
    overall: str  # "ok" | "degraded" | "down"
    providers: list[ProviderProbe] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "checked_at": self.checked_at,
            "overall": self.overall,
            "providers": [asdict(p) for p in self.providers],
            "cache_ttl_s": HEALTH_CACHE_TTL_S,
        }


_cache: Optional[HealthSnapshot] = None
_lock = asyncio.Lock()


def _is_deployed_env() -> bool:
    """Heuristic: are we running in a remote/deployed environment where a
    'localhost' service can't exist? Used to suppress noisy 'down' reports
    for optional local-only providers (e.g. Ollama)."""
    for v in ("HF_SPACE_ID", "SPACE_ID", "RENDER", "FLY_APP_NAME",
              "RAILWAY_ENVIRONMENT", "VERCEL"):
        if os.environ.get(v):
            return True
    return False


def _ollama_not_configured() -> Optional[str]:
    """Return a 'not_configured' detail message for Ollama if it shouldn't be
    probed, else None. Local Ollama is optional — if the user hasn't set
    OLLAMA_BASE_URL and we're deployed, OR if base URL is local-only and
    we're deployed, skip the probe."""
    base = os.environ.get("OLLAMA_BASE_URL")
    if base is None and _is_deployed_env():
        return "Local Ollama not running (optional fallback)"
    if base is not None:
        # Explicitly configured — let the probe run (user opted in).
        return None
    # No env var, not deployed → local dev. Run the probe; if it fails,
    # surface as not_configured (not "down") because Ollama is optional.
    return None


def _not_configured(name: str, role: str, detail: str) -> ProviderProbe:
    return ProviderProbe(name=name, role=role, status="not_configured",
                         detail=detail)


async def _probe(
    name: str, role: str, coro_factory
) -> ProviderProbe:
    # Provider-specific not_configured short-circuits.
    if name == "ollama":
        msg = _ollama_not_configured()
        if msg:
            return _not_configured(name, role, msg)
    if not _has_required_key(name):
        # Promote missing-key providers (other than Ollama which is keyless)
        # to "not_configured" so the UI can de-emphasize them rather than
        # treating absence as failure.
        return _not_configured(name, role, "API key not configured (optional)")
    start = time.perf_counter()
    try:
        await asyncio.wait_for(coro_factory(), timeout=_PROBE_TIMEOUT_S)
        latency = int((time.perf_counter() - start) * 1000)
        return ProviderProbe(name=name, role=role, status="ok", latency_ms=latency)
    except asyncio.TimeoutError:
        return ProviderProbe(name=name, role=role, status="degraded",
                             detail=f"timeout after {_PROBE_TIMEOUT_S}s")
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code in (401, 403):
            env_var = _REQUIRED_KEYS.get(name) or f"{name.upper()}_API_KEY"
            detail = (f"HTTP {code} — invalid or expired key. "
                      f"Check your {env_var} value.")
        else:
            detail = f"HTTP {code}"
        return ProviderProbe(name=name, role=role, status="down", detail=detail)
    except Exception as e:
        # Ollama unreachable on a local box with no server running → optional,
        # not a real failure. Same for any provider with no key (defensive).
        if name == "ollama":
            return _not_configured(
                name, role, "Local Ollama not running (optional fallback)"
            )
        return ProviderProbe(name=name, role=role, status="down",
                             detail=f"{type(e).__name__}: {str(e)[:80]}")


_REQUIRED_KEYS = {
    "parallel": "PARALLEL_API_KEY",
    "tavily": "TAVILY_API_KEY",
    "serper": "SERPER_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "github_models": "GITHUB_TOKEN",
    "openrouter": "OPENROUTER_API_KEY",
    "sarvam": "SARVAM_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    # Ollama uses a local server, not an API key. We treat "always
    # has key" so the probe runs, and the probe itself reports
    # "down" when the server isn't reachable — that's the right signal.
    "ollama": None,
}


def _has_required_key(name: str) -> bool:
    if name not in _REQUIRED_KEYS:
        return False
    env_var = _REQUIRED_KEYS[name]
    if env_var is None:
        # Keyless provider (e.g. Ollama). Probe always runs; failure mode is
        # surfaced by the probe itself ("down" = server unreachable).
        return True
    # Providers with multi-key fallback envs: accept either form as "configured".
    if name == "groq" and os.environ.get("GROQ_API_KEYS"):
        return True
    if name == "gemini" and os.environ.get("GEMINI_API_KEYS"):
        return True
    return bool(os.environ.get(env_var))


async def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=_PROBE_TIMEOUT_S,
        headers={"User-Agent": "sarvam-research-agent/health-check"},
    )


async def _probe_parallel(c: httpx.AsyncClient) -> None:
    r = await c.post(
        "https://api.parallel.ai/v1beta/search",
        json={"objective": "ping", "search_queries": ["ping"], "max_results": 1},
        headers={"x-api-key": os.environ["PARALLEL_API_KEY"]},
    )
    r.raise_for_status()


async def _probe_tavily(c: httpx.AsyncClient) -> None:
    r = await c.post(
        "https://api.tavily.com/search",
        json={"api_key": os.environ["TAVILY_API_KEY"], "query": "ping",
              "max_results": 1},
    )
    r.raise_for_status()


async def _probe_serper(c: httpx.AsyncClient) -> None:
    r = await c.post(
        "https://google.serper.dev/search",
        json={"q": "ping", "num": 1},
        headers={"X-API-KEY": os.environ["SERPER_API_KEY"]},
    )
    r.raise_for_status()


async def _probe_chat(c: httpx.AsyncClient, base_url: str, key: str,
                     model: str) -> None:
    r = await c.post(
        f"{base_url}/chat/completions",
        json={"model": model,
              "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 1, "temperature": 0.0},
        headers={"Authorization": f"Bearer {key}"},
    )
    r.raise_for_status()


async def _probe_groq(c: httpx.AsyncClient) -> None:
    await _probe_chat(c, "https://api.groq.com/openai/v1",
                      os.environ["GROQ_API_KEY"], "llama-3.3-70b-versatile")


async def _probe_github(c: httpx.AsyncClient) -> None:
    await _probe_chat(c, "https://models.inference.ai.azure.com",
                      os.environ["GITHUB_TOKEN"], "gpt-4o-mini")


async def _probe_openrouter(c: httpx.AsyncClient) -> None:
    """Probe OpenRouter via /models (metadata, no generation).

    The actual synthesis fallback uses DeepSeek R1 — a reasoning model with
    5-30s typical latency that legitimately exceeds the 8s probe budget. The
    /models endpoint proves auth + connectivity with no token cost and ~200ms
    response, giving an honest signal of OpenRouter health without conflating
    it with R1's intrinsic slowness."""
    r = await c.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    )
    r.raise_for_status()


async def _probe_sarvam(c: httpx.AsyncClient) -> None:
    await _probe_chat(c, "https://api.sarvam.ai/v1",
                      os.environ["SARVAM_API_KEY"],
                      os.environ.get("SARVAM_MODEL", "sarvam-m"))


async def _probe_gemini(c: httpx.AsyncClient) -> None:
    # Pull from the rotator so the probe drains a key from the same pool as
    # synthesis — gives quota-balanced telemetry across multi-key setups.
    from utils.provider_router import _GEMINI_ROTATOR
    key = _GEMINI_ROTATOR.next_key() or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("No GEMINI_API_KEY / GEMINI_API_KEYS configured")
    r = await c.post(
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash:generateContent?key={key}",
        json={"contents": [{"parts": [{"text": "hi"}]}],
              "generationConfig": {"maxOutputTokens": 1}},
    )
    r.raise_for_status()


async def _probe_cerebras(c: httpx.AsyncClient) -> None:
    """Cerebras Cloud — OpenAI-compatible chat completions endpoint."""
    base = os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
    model = os.environ.get("CEREBRAS_MODEL", "llama3.1-8b")
    await _probe_chat(c, base, os.environ["CEREBRAS_API_KEY"], model)


async def _probe_ollama(c: httpx.AsyncClient) -> None:
    """Ollama local server — checks the OpenAI-compatible /v1/models endpoint
    (which doesn't require a hot model). 'missing_key' translates here to
    'server not reachable', which is the right user-visible signal."""
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    r = await c.get(f"{base.rstrip('/')}/models", timeout=3.0)
    r.raise_for_status()


_PROBE_SPECS = [
    ("parallel",      "search",      _probe_parallel),
    ("tavily",        "search",      _probe_tavily),
    ("serper",        "search",      _probe_serper),
    ("gemini",        "synth",       _probe_gemini),
    ("groq",          "planner",     _probe_groq),
    ("github_models", "judge",       _probe_github),
    ("openrouter",    "synth-fallback", _probe_openrouter),
    ("sarvam",        "synth-indic", _probe_sarvam),
    ("cerebras",      "synth-fast",  _probe_cerebras),
    ("ollama",        "synth-local", _probe_ollama),
]


# Critical providers — if any of these are "down", overall = "down".
# Anything else degraded → overall "degraded".
_CRITICAL = {"parallel", "gemini", "groq"}


async def _run_all_probes() -> HealthSnapshot:
    async with await _client() as c:
        tasks = [_probe(name, role, lambda fn=fn, c=c: fn(c))
                 for name, role, fn in _PROBE_SPECS]
        probes = await asyncio.gather(*tasks)

    statuses = {p.name: p.status for p in probes}
    # `not_configured` is missing-by-design (e.g. optional Ollama on a remote
    # deployment) — it must NOT count toward degraded/down.
    if any(statuses.get(n) == "down" for n in _CRITICAL):
        overall = "down"
    elif any(p.status in ("down", "degraded") for p in probes):
        overall = "degraded"
    else:
        overall = "ok"

    return HealthSnapshot(
        checked_at=time.time(),
        overall=overall,
        providers=list(probes),
    )


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    snapshot: Optional[HealthSnapshot] = None


async def eval_preflight() -> PreflightResult:
    """Pre-flight check for `eval_runner.py`.

    Eval needs four working capability groups; any missing group is a blocker:
      1. At least one search provider (parallel | tavily | serper)
      2. At least one synthesis provider (gemini | sarvam | openrouter)
      3. Groq — required for planner, conflict probe, rolling summary
      4. Configured judge provider (JUDGE_PROVIDER, default groq)

    Returns a typed result; eval_runner formats and exits on `.ok == False`.
    """
    snap = await get_health(force=True)
    status = {p.name: p.status for p in snap.providers}

    blockers: list[str] = []
    warnings: list[str] = []

    def _summarize(names: list[str]) -> str:
        return ", ".join(f"{n}={status.get(n, '?')}" for n in names)

    search_providers = ["parallel", "tavily", "serper"]
    if not any(status.get(p) == "ok" for p in search_providers):
        blockers.append(
            "no working search provider — " + _summarize(search_providers)
        )

    synth_providers = ["gemini", "sarvam", "openrouter"]
    if not any(status.get(p) == "ok" for p in synth_providers):
        blockers.append(
            "no working synthesis provider — " + _summarize(synth_providers)
        )

    if status.get("groq") != "ok":
        blockers.append(
            f"groq is required for planner/conflict-probe/rolling-summary "
            f"but reports {status.get('groq', '?')}"
        )

    judge_provider = os.environ.get("JUDGE_PROVIDER", "groq").lower()
    judge_name = "github_models" if judge_provider == "github" else judge_provider
    if status.get(judge_name) != "ok":
        blockers.append(
            f"configured judge provider '{judge_provider}' "
            f"(probe={judge_name}) reports {status.get(judge_name, '?')}"
        )

    # Soft warnings — not blockers, just nice to surface.
    for p in snap.providers:
        if p.status == "degraded":
            warnings.append(f"{p.name} degraded: {p.detail}")
        elif p.status == "down" and p.name not in {
            *search_providers, *synth_providers, "groq", judge_name
        }:
            warnings.append(f"{p.name} down: {p.detail}")

    return PreflightResult(
        ok=not blockers,
        blockers=blockers,
        warnings=warnings,
        snapshot=snap,
    )


async def get_health(force: bool = False) -> HealthSnapshot:
    """Return cached snapshot. Refresh if older than TTL or force=True."""
    global _cache
    now = time.time()
    if (not force
            and _cache is not None
            and (now - _cache.checked_at) < HEALTH_CACHE_TTL_S):
        return _cache
    async with _lock:
        # Double-check after acquiring the lock — another caller may have
        # refreshed while we were waiting.
        if (not force
                and _cache is not None
                and (time.time() - _cache.checked_at) < HEALTH_CACHE_TTL_S):
            return _cache
        try:
            _cache = await _run_all_probes()
        except Exception as exc:
            logger.warning("provider health probe failed: %s", exc)
            _cache = HealthSnapshot(
                checked_at=time.time(),
                overall="down",
                providers=[],
            )
        return _cache
