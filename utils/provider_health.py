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

HEALTH_CACHE_TTL_S = int(os.environ.get("HEALTH_CACHE_TTL_S", "90"))
_PROBE_TIMEOUT_S = 8.0
# Audit smart-polling: per-provider exponential backoff after consecutive
# failures, capped at 15 minutes. Resets on first success. Separate from the
# `Retry-After`-driven throttle window which short-circuits earlier.
_BACKOFF_MAX_S = 900
_BACKOFF_FAIL_THRESHOLD = 3


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
    # When True, the cached snapshot is stale because the most recent probe
    # attempt itself failed (network, exception). We keep the previous data
    # so the UI doesn't go blank, but signal that callers should treat this
    # as untrusted.
    stale: bool = False

    def to_dict(self) -> dict:
        return {
            "checked_at": self.checked_at,
            "overall": self.overall,
            "providers": [asdict(p) for p in self.providers],
            "cache_ttl_s": HEALTH_CACHE_TTL_S,
            "stale": self.stale,
        }


_cache: Optional[HealthSnapshot] = None
_lock = asyncio.Lock()

# Per-provider next-probe-allowed timestamp. Set when a probe returns 429
# with Retry-After, or after N consecutive failures (exponential backoff).
# Probes are short-circuited to a cached "throttled"/"down" probe entry
# until the timestamp passes. Invalidation clears the entry.
_next_probe_at: dict[str, float] = {}
_consecutive_failures: dict[str, int] = {}
# Providers explicitly invalidated by callers (e.g. orchestrator on 429).
# Cleared on the next probe of that provider.
_force_reprobe: set[str] = set()


def invalidate(provider: Optional[str] = None) -> None:
    """Mark the cached snapshot stale so the next ``get_health()`` re-probes.

    When ``provider`` is set, only that provider's throttle/backoff state is
    cleared and the next ``get_health()`` will run a fresh probe round. When
    ``provider`` is None, the entire cache is invalidated.

    Wired from real-failure choke points (orchestrator 429 handling) so a sick
    provider gets re-probed before the 90s TTL expires, while idle dashboards
    continue to coast on the cached snapshot.
    """
    global _cache
    if provider is None:
        _cache = None
        _next_probe_at.clear()
        _consecutive_failures.clear()
        _force_reprobe.clear()
        return
    _next_probe_at.pop(provider, None)
    _consecutive_failures.pop(provider, None)
    _force_reprobe.add(provider)
    # Mark the cached snapshot as stale so callers see the hint even before
    # the next refresh tick completes.
    if _cache is not None:
        _cache = HealthSnapshot(
            checked_at=_cache.checked_at,
            overall=_cache.overall,
            providers=_cache.providers,
            stale=True,
        )


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


def _record_failure(name: str) -> float:
    """Bump consecutive-failure count and return the resulting backoff seconds
    once the threshold is reached (0 below threshold). Capped at _BACKOFF_MAX_S."""
    n = _consecutive_failures.get(name, 0) + 1
    _consecutive_failures[name] = n
    if n < _BACKOFF_FAIL_THRESHOLD:
        return 0.0
    # 3 failures → 2× TTL, 4 → 5× TTL, 5+ → 10× TTL, capped.
    multipliers = {3: 2, 4: 5}
    mult = multipliers.get(n, 10)
    return float(min(_BACKOFF_MAX_S, HEALTH_CACHE_TTL_S * mult))


def _record_success(name: str) -> None:
    _consecutive_failures.pop(name, None)


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

    # Honor per-provider throttle windows: if Retry-After or exponential
    # backoff has set a future next-probe-at, return a cached "throttled"
    # entry instead of hitting the wire. invalidate(provider) clears this.
    now = time.time()
    if name in _force_reprobe:
        _force_reprobe.discard(name)
        _next_probe_at.pop(name, None)
    next_at = _next_probe_at.get(name)
    if next_at and now < next_at:
        wait = int(next_at - now)
        return ProviderProbe(
            name=name, role=role, status="throttled",
            detail=f"backing off; retrying in ~{wait}s",
        )

    start = time.perf_counter()
    try:
        await asyncio.wait_for(coro_factory(), timeout=_PROBE_TIMEOUT_S)
        latency = int((time.perf_counter() - start) * 1000)
        _record_success(name)
        return ProviderProbe(name=name, role=role, status="ok", latency_ms=latency)
    except asyncio.TimeoutError:
        backoff = _record_failure(name)
        if backoff:
            _next_probe_at[name] = time.time() + backoff
        return ProviderProbe(name=name, role=role, status="degraded",
                             detail=f"timeout after {_PROBE_TIMEOUT_S}s")
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code == 429:
            # Respect Retry-After when the provider tells us; else default to
            # the configured TTL so we're not hammering during a known throttle.
            retry_after_s = _parse_retry_after(e.response.headers.get("Retry-After"))
            if retry_after_s is None:
                retry_after_s = float(HEALTH_CACHE_TTL_S)
            _next_probe_at[name] = time.time() + retry_after_s
            _record_success(name)  # 429 is a quota signal, not a hard failure
            return ProviderProbe(
                name=name, role=role, status="throttled",
                detail=f"HTTP 429 — retry after ~{int(retry_after_s)}s",
            )
        if code in (401, 403):
            env_var = _REQUIRED_KEYS.get(name) or f"{name.upper()}_API_KEY"
            detail = (f"HTTP {code} — invalid or expired key. "
                      f"Check your {env_var} value.")
        else:
            detail = f"HTTP {code}"
        backoff = _record_failure(name)
        if backoff:
            _next_probe_at[name] = time.time() + backoff
        return ProviderProbe(name=name, role=role, status="down", detail=detail)
    except Exception as e:
        # Ollama unreachable on a local box with no server running → optional,
        # not a real failure. Same for any provider with no key (defensive).
        if name == "ollama":
            return _not_configured(
                name, role, "Local Ollama not running (optional fallback)"
            )
        backoff = _record_failure(name)
        if backoff:
            _next_probe_at[name] = time.time() + backoff
        return ProviderProbe(name=name, role=role, status="down",
                             detail=f"{type(e).__name__}: {str(e)[:80]}")


def _parse_retry_after(header: Optional[str]) -> Optional[float]:
    """Parse a `Retry-After` header value. Returns seconds, or None if absent
    or unparseable. Supports the integer-seconds form; HTTP-date form is
    treated as missing because the providers we hit emit seconds."""
    if not header:
        return None
    try:
        return float(header.strip())
    except (TypeError, ValueError):
        return None


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
    if name == "cerebras" and os.environ.get("CEREBRAS_API_KEYS"):
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


def _first_key(single_var: str, multi_var: str) -> str:
    # Multi-key mode (GROQ_API_KEYS / GEMINI_API_KEYS) is the new default;
    # the legacy single-key var stays a valid fallback. Prefer the first
    # multi-key entry so the probe matches what real traffic uses.
    multi = os.environ.get(multi_var, "").strip()
    if multi:
        first = next((p.strip() for p in multi.split(",") if p.strip()), "")
        if first:
            return first
    return os.environ[single_var]


async def _probe_groq(c: httpx.AsyncClient) -> None:
    # Cheap probe: list models. Validates auth, no token cost. Avoids
    # consuming the daily request quota that 30s dashboard polling would
    # otherwise eat into.
    r = await c.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {_first_key('GROQ_API_KEY', 'GROQ_API_KEYS')}"},
    )
    r.raise_for_status()


async def _probe_github(c: httpx.AsyncClient) -> None:
    # GitHub Models doesn't expose a public /models list, but a 1-token chat
    # probe is the cheapest valid auth check. Cached at 90s, this is ~960
    # calls/day worst case, well under any per-day cap.
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
    # Sarvam Model API has no free `/models` listing — the cheapest valid
    # auth check is a 1-token chat completion. Aggressive TTL caching keeps
    # this from burning meaningful quota (90s → 960 calls/day worst case).
    await _probe_chat(c, "https://api.sarvam.ai/v1",
                      os.environ["SARVAM_API_KEY"],
                      os.environ.get("SARVAM_MODEL", "sarvam-m"))


async def _probe_gemini(c: httpx.AsyncClient) -> None:
    # Cheap probe: GET /models. Lists available models, no quota / token cost.
    # CRITICAL for Gemini specifically — the free tier is 1500 requests/day,
    # which a 30s polling loop on a few open tabs can exhaust in hours. The
    # previous probe issued a real generateContent call.
    # We pull from the rotator so probes drain the same key pool as synthesis
    # gets quota-balanced telemetry across multi-key setups.
    from utils.provider_router import _GEMINI_ROTATOR
    key = _GEMINI_ROTATOR.next_key() or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("No GEMINI_API_KEY / GEMINI_API_KEYS configured")
    r = await c.get(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    )
    r.raise_for_status()


async def _probe_cerebras(c: httpx.AsyncClient) -> None:
    """Cerebras Cloud — cheap GET /models endpoint (no token cost)."""
    base = os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
    key = _first_key("CEREBRAS_API_KEY", "CEREBRAS_API_KEYS")
    r = await c.get(
        f"{base.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    r.raise_for_status()


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
            # If a probe round itself fails (rare — _run_all_probes catches
            # per-provider errors), keep the previous snapshot but flag it
            # stale so the UI can warn. Falling back to a "down/empty"
            # snapshot would erase healthy state from a transient hiccup.
            logger.warning("provider health probe failed: %s", exc)
            if _cache is not None:
                _cache = HealthSnapshot(
                    checked_at=_cache.checked_at,
                    overall=_cache.overall,
                    providers=_cache.providers,
                    stale=True,
                )
            else:
                _cache = HealthSnapshot(
                    checked_at=time.time(),
                    overall="down",
                    providers=[],
                    stale=True,
                )
        return _cache
