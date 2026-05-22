import asyncio
import json
import logging
import os
import re
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import aiosqlite

from agent import eval_queries
from agent import memory as _memory
from agent.memory import init_db, DB_PATH
from agent.orchestrator import ResearchOrchestrator, RuntimeConfig
from utils.cancellation import get_registry
from utils.env_check import validate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    validate()
except Exception as e:
    logger.error(f"Startup validation failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    # V3.10: seed eval_runs from bundled JSONL on first boot, so a freshly
    # deployed HF Space doesn't show "No eval runs recorded — run python
    # eval/eval_runner.py" to a reviewer who can't run python from the URL.
    # Idempotent: no-op once any row exists in eval_runs.
    from utils.eval_seed import seed_if_empty
    seeded = await seed_if_empty()
    if seeded:
        logger.info("Seeded %d eval rows from bundled JSONL", seeded)

    # V3.8: resolve retrieval mode visibly on startup. `hybrid` mode raises on
    # capability failure (fail-loud CI contract); `auto` and `lexical` log
    # structured state. After this block the effective default is observable
    # in logs and `/settings/defaults`.
    from utils.retrieval_mode import resolve, requested_mode
    requested = requested_mode()
    try:
        eff = resolve(requested, _memory._VEC_AVAILABLE)
    except RuntimeError as e:
        logger.error("Refusing to start: %s", e)
        raise
    if eff.is_fallback:
        logger.warning(
            "[retrieval] mode=lexical_fallback requested=%s reason=%s — "
            "hybrid retrieval is disabled for this process. Switch to "
            "RETRIEVAL_MODE=lexical to silence, or to a Python compiled with "
            "--enable-loadable-sqlite-extensions to enable hybrid.",
            requested.value, eff.reason,
        )
    else:
        logger.info(
            "[retrieval] mode=%s requested=%s",
            eff.effective.value, requested.value,
        )

    # Pre-warm the embedding ONNX session when hybrid is the effective default
    # so the *first* user query doesn't pay the cold-start. Dockerfile already
    # baked the model weights into the image layer; this just initializes the
    # in-process ONNX session (~1s) against the cached weights.
    if eff.is_hybrid:
        from agent.embedder import warm as _warm_embeddings
        asyncio.create_task(_warm_embeddings())

    yield


app = FastAPI(title="Deep Research Agent", lifespan=lifespan)

_allowed = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
_allow_origins = [o.strip() for o in _allowed.split(",") if o.strip()]
# S3 fix: with allow_credentials=True a wildcard origin is both incorrect (the
# CORS spec forbids it) and a security risk. Fail loudly at import time so a
# misconfigured deployment never reaches users.
if "*" in _allow_origins:
    raise RuntimeError(
        "CORS misconfiguration: '*' incompatible with allow_credentials=True. "
        "Set ALLOWED_ORIGINS to a specific comma-separated origin list."
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_MAX_OVERRIDES_KEYS = 32
_MAX_OVERRIDE_VALUE_CHARS = 500


class ChatRequest(BaseModel):
    # Audit M2: bound query length to defend against abusive payloads. 2000
    # chars is far longer than any realistic research question; anything
    # beyond is almost certainly junk or an attempt to inflate tokens.
    query: str = Field(max_length=2000)
    session_id: str = Field(max_length=200)
    # Stateless runtime overrides (Option D). Keys are env-var-style names
    # (e.g. "HYBRID_RETRIEVAL", "FAILURE_POLICY_MAX_HOPS", "CONTEXT_SELECTION_STRATEGY").
    # Values may be strings, ints, or booleans — the orchestrator coerces.
    # Frontend stores these in localStorage and includes them with each turn,
    # so eval_runner.py (which doesn't send overrides) stays reproducible
    # from env vars alone.
    overrides: dict | None = None
    # Phase 2: human-in-the-loop plan-approval gate. Off by default so eval
    # and unmodified clients behave identically. When True, the orchestrator
    # pauses after PLANNING and emits a `plan_approval` SSE event; the client
    # must call POST /research/approve/{turn_id} (or /cancel) to resume.
    approval_required: bool = False

    @field_validator("overrides")
    @classmethod
    def _validate_overrides(cls, v: dict | None) -> dict | None:
        """Audit M2: cap overrides dict size + each value length. Prevents a
        client from shipping unbounded `overrides` payloads to inflate the
        request envelope or stuff our log lines with junk."""
        if v is None:
            return None
        if len(v) > _MAX_OVERRIDES_KEYS:
            raise ValueError(
                f"overrides has too many keys: {len(v)} (max={_MAX_OVERRIDES_KEYS})"
            )
        for k, val in v.items():
            if not isinstance(k, str) or len(k) > 100:
                raise ValueError(f"overrides key invalid or too long: {k!r}")
            if isinstance(val, str) and len(val) > _MAX_OVERRIDE_VALUE_CHARS:
                raise ValueError(
                    f"overrides[{k!r}] string value exceeds "
                    f"{_MAX_OVERRIDE_VALUE_CHARS} chars"
                )
        return v


class ApprovePayload(BaseModel):
    """Phase 2: payload accepted by POST /research/approve/{turn_id}.

    `sub_queries` is the only editable surface — length-capped at 6 and each
    item capped at 200 chars. None means "accept the plan as-is".

    S1 fix: `session_id` proves the caller owns this turn. Required — handler
    returns 404 if missing or mismatched.
    """
    sub_queries: list[str] | None = None
    session_id: str | None = None


_APPROVAL_MAX_SUB_QUERIES = 6
_APPROVAL_MAX_QUERY_CHARS = 200
# Strip <…> tags (HTML/script) from sub_queries — schema-validated only,
# never trust the client edit surface to be safe to render or to feed to the
# planner output. Applied after whitespace strip.
_APPROVAL_HTML_PATTERN = re.compile(r"<[^>]*>")


# Phase 1.25: CoT pre-emit filter — defense-in-depth against accidentally
# streaming Gemini `thought_summary`/`thought_tokens` content if a future
# refactor wires it in. The filter operates on the serialized payload so it
# catches both `data.text="<thinking>..."` and `data={"thought_summary": …}`.
_COT_PATTERNS = re.compile(
    r"(<thinking>|</thinking>|<thought>|</thought>|"
    r"thought_summary|thought_tokens|"
    r"reasoning_content|reasoning_tokens|"
    r"redacted_thinking)",
    re.IGNORECASE,
)


def _contains_cot(payload: dict) -> bool:
    try:
        return bool(_COT_PATTERNS.search(json.dumps(payload, default=str)))
    except Exception:
        return False


def _format_event(payload: dict, *, event_id: int | None = None,
                  event_type: str | None = None) -> str:
    """Encode an SSE frame. When `event_type` is set, emits the SSE `event:`
    discriminator; `EventSource.addEventListener(event_type, …)` will then
    dispatch to a typed handler. `event_id` enables `Last-Event-ID` replay."""
    parts: list[str] = []
    if event_id is not None:
        parts.append(f"id: {event_id}")
    if event_type:
        parts.append(f"event: {event_type}")
    parts.append(f"data: {json.dumps(payload)}")
    return "\n".join(parts) + "\n\n"


# Phase 1.25: per-turn ring buffer for SSE replay on reconnect. Bounded to
# avoid unbounded memory if a client never connects back. 50 frames covers a
# typical run's burst budget (planning + ~10 search results + ~10 fetches +
# selections + answer chunks are sampled).
_SSE_REPLAY_MAX = 50
# Audit L1: cap the outer dict at 100 active turns; LRU eviction. Without
# this, a server that's never reconnected to leaks per-turn buffers forever.
_SSE_REPLAY_TURNS_MAX = 100
_sse_replay: "OrderedDict[str, deque[tuple[int, str | None, dict]]]" = OrderedDict()


def _replay_remember(turn_id: str, eid: int, etype: str | None, payload: dict) -> None:
    buf = _sse_replay.get(turn_id)
    if buf is None:
        buf = deque(maxlen=_SSE_REPLAY_MAX)
        _sse_replay[turn_id] = buf
        # Evict oldest if we're over the cap. Safe because a recently dropped
        # buffer is one whose stream finished long ago — a Last-Event-ID
        # reconnect to it would already be useless.
        while len(_sse_replay) > _SSE_REPLAY_TURNS_MAX:
            _sse_replay.popitem(last=False)
    else:
        _sse_replay.move_to_end(turn_id)
    buf.append((eid, etype, payload))


def _replay_since(turn_id: str, last_event_id: int) -> list[tuple[int, str | None, dict]]:
    buf = _sse_replay.get(turn_id) or deque()
    return [(eid, et, pl) for (eid, et, pl) in buf if eid > last_event_id]


def _replay_drop(turn_id: str) -> None:
    _sse_replay.pop(turn_id, None)


async def _disconnect_watcher(
    request: Request, token, turn_id: str, interval_s: float = 1.0
) -> None:
    """Background task: polls `request.is_disconnected()` independently of the
    SSE generator so cancellation fires *during* long stalls (e.g. multi-second
    synth chunks) instead of only at the next event boundary."""
    try:
        while True:
            await asyncio.sleep(interval_s)
            # Phase 2: suspend disconnect polling while the orchestrator is
            # paused on the plan-approval gate. A momentary SSE proxy hiccup
            # during a 5-minute approval window must not auto-cancel the turn
            # — the heartbeat keeps the actual stream alive, and the explicit
            # /cancel endpoint remains the way to abort.
            if getattr(token, "paused", False):
                continue
            if await request.is_disconnected():
                logger.info(
                    "Client disconnected; cancelling stream",
                    extra={"component": "sse", "turn_id": turn_id},
                )
                token.cancel()
                return
    except asyncio.CancelledError:
        return


_HEARTBEAT_INTERVAL_S = float(os.environ.get("SSE_HEARTBEAT_S", "15"))


async def _research_stream(req: ChatRequest, request: Request, *,
                           turn_id: str | None = None,
                           replay_from: int = -1):
    """SSE generator: registers a cancellation token, spawns a parallel
    disconnect watcher, streams ExecutionEvents from the orchestrator.

    Phase 1.25: monotonic `id:` per event, `retry:` once on open, periodic
    heartbeat comments, per-turn ring buffer for `Last-Event-ID` replay,
    CoT-content pre-emit filter, SSE `event:` discriminator carrying the typed
    event name when the orchestrator sets `event.event_type`.
    """
    turn_id = turn_id or str(uuid.uuid4())
    registry = get_registry()
    # S1 fix: bind the session_id to the token so /cancel and /approve can
    # verify ownership. turn_id leaks via the first SSE frame; session_id is
    # the secret that proves the caller owns this turn.
    token = await registry.register(turn_id, session_id=req.session_id)
    orchestrator = ResearchOrchestrator()
    first_event = True
    watcher = asyncio.create_task(_disconnect_watcher(request, token, turn_id))
    config = RuntimeConfig.from_overrides(
        req.overrides, approval_required=req.approval_required
    )
    event_id_counter = 0

    # Heartbeat queue: a side coroutine pushes ": ping" comments into this
    # queue every _HEARTBEAT_INTERVAL_S so proxies (nginx/cf/vercel) don't
    # close an idle stream during slow synth. We use a queue rather than
    # racing the orchestrator because asyncio.wait()'s ergonomics get gnarly
    # when the inner producer is an async generator.
    out_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _heartbeat() -> None:
        try:
            while True:
                await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
                await out_queue.put(": ping\n\n")
        except asyncio.CancelledError:
            return

    async def _drive() -> None:
        nonlocal first_event, event_id_counter
        try:
            async for event in orchestrator.run(
                req.query, req.session_id, cancel_token=token, turn_id=turn_id,
                config=config,
            ):
                data = event.data
                if first_event and event.step == "planning":
                    if isinstance(data, dict):
                        data = {**data, "turn_id": turn_id}
                    else:
                        data = {"turn_id": turn_id}
                payload = {"step": event.step, "label": event.label, "data": data}

                # Phase 1.25: CoT pre-emit filter — drop and log if matched.
                if _contains_cot(payload):
                    logger.warning(
                        "Dropping SSE frame containing CoT-like content",
                        extra={"component": "sse", "turn_id": turn_id, "step": event.step},
                    )
                    continue

                event_id_counter += 1
                etype = event.event_type
                _replay_remember(turn_id, event_id_counter, etype, payload)
                frame = _format_event(payload, event_id=event_id_counter, event_type=etype)
                await out_queue.put(frame)
                first_event = False
        except Exception:
            # S4 fix: never leak exception text to the SSE stream — it can
            # contain file paths, internal URLs, or stack-trace fragments.
            # Log the full exception server-side and send a generic message.
            logger.exception("SSE handler error", extra={"turn_id": turn_id})
            event_id_counter += 1
            err_payload = {
                "step": "error",
                "label": "error",
                "data": "Internal processing error.",
            }
            await out_queue.put(_format_event(err_payload, event_id=event_id_counter))
        finally:
            await out_queue.put(None)  # sentinel

    hb_task = asyncio.create_task(_heartbeat())
    drive_task = asyncio.create_task(_drive())

    try:
        # SSE bootstrap: retry hint + replay buffered events for Last-Event-ID.
        yield "retry: 3000\n\n"
        if replay_from >= 0:
            for (eid, etype, payload) in _replay_since(turn_id, replay_from):
                yield _format_event(payload, event_id=eid, event_type=etype)

        while True:
            frame = await out_queue.get()
            if frame is None:
                break
            yield frame
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except (asyncio.CancelledError, Exception):
            pass
        # Drain drive_task to clean exit; it should already be done.
        try:
            await drive_task
        except (asyncio.CancelledError, Exception):
            pass
        watcher.cancel()
        try:
            await watcher
        except (asyncio.CancelledError, Exception):
            pass
        await orchestrator.aclose()
        await registry.release(turn_id)
        # Keep the per-turn replay buffer around briefly so a near-instant
        # reconnect can resume. Schedule eviction in the background; for now,
        # drop immediately to keep test isolation predictable.
        _replay_drop(turn_id)


@app.post("/research")
async def research(req: ChatRequest, request: Request):
    # Phase 1.25: honor `Last-Event-ID` for resumed streams. Browsers attach
    # this header automatically on EventSource auto-reconnect. We can also
    # accept an explicit `?turn_id=…&last_event_id=…` for non-EventSource
    # clients (e.g. our fetch-based reader in use-sse-research.ts).
    leid_str = request.headers.get("Last-Event-ID") or request.query_params.get("last_event_id")
    try:
        leid = int(leid_str) if leid_str is not None else -1
    except (TypeError, ValueError):
        leid = -1
    resume_turn = request.query_params.get("turn_id")
    return StreamingResponse(
        _research_stream(req, request, turn_id=resume_turn, replay_from=leid),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/stream/labels")
async def stream_labels():
    """Phase 1.25: single source of truth for the user-facing phase labels.
    Frontend fetches this once on mount and uses the result to render its
    pipeline rail — eliminating the duplicated PIPELINE array drift risk."""
    from agent.orchestrator import STREAM_LABELS, PHASE_ORDER
    return {
        "labels": STREAM_LABELS,
        "order": PHASE_ORDER,
    }


# DEPRECATED: remove after 2026-06-30 — use /research instead.
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """Deprecated alias for /research — kept until static/app.js is removed."""
    logger.warning(
        "Deprecated /chat/stream endpoint hit; clients should migrate to /research",
        extra={"component": "api", "endpoint": "/chat/stream"},
    )
    resp = await research(req, request)
    # Per RFC 8594; some HTTP clients surface this header in dev tools.
    resp.headers["Deprecation"] = "true"
    resp.headers["Sunset"] = "Tue, 30 Jun 2026 23:59:59 GMT"
    return resp


class CancelPayload(BaseModel):
    """S1 fix: require session_id so the caller proves ownership of the turn.
    Optional for backwards compat at parse time; the handler enforces presence."""
    session_id: str | None = None


@app.post("/research/cancel/{turn_id}")
async def cancel_research(
    turn_id: str,
    request: Request,
    payload: CancelPayload | None = None,
):
    # Accept session_id from JSON body OR query param for flexibility.
    session_id: str | None = None
    if payload is not None:
        session_id = payload.session_id
    if not session_id:
        session_id = request.query_params.get("session_id")
    # S1 fix: a missing session_id is treated as "no proof of ownership" — we
    # return 404 (pretending the turn doesn't exist) rather than 400 to avoid
    # leaking information about which turn_ids are currently active.
    ok = await get_registry().cancel(turn_id, session_id=session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="turn_id not active")
    return {"cancelled": True}


def _sanitize_sub_query(raw: str) -> str | None:
    """Strip whitespace and HTML-ish tags from a candidate sub_query.
    Returns None when the query is empty after sanitization or exceeds the
    per-item char cap (caller maps None → 400)."""
    if not isinstance(raw, str):
        return None
    cleaned = _APPROVAL_HTML_PATTERN.sub("", raw).strip()
    if not cleaned:
        return None
    if len(cleaned) > _APPROVAL_MAX_QUERY_CHARS:
        return None
    return cleaned


@app.post("/research/approve/{turn_id}")
async def approve_research(turn_id: str, request: Request, payload: ApprovePayload):
    """Phase 2: resolve a paused plan-approval gate. Idempotent — a second
    call after approve/cancel/timeout returns 409.

    The body's `sub_queries` is the only editable surface (length-capped at
    6, each item ≤ 200 chars, HTML tags stripped). None or omitted means
    "accept the plan as-is".

    S1 fix: requires `session_id` (body or `?session_id=…` query param) so the
    caller proves ownership. Missing or mismatched → 404.
    """
    session_id = payload.session_id or request.query_params.get("session_id")
    tok = await get_registry().get(turn_id, session_id=session_id)
    if tok is None:
        raise HTTPException(status_code=404, detail="turn_id not active")

    # Idempotency / state gate. We only resolve a wait that is actually
    # pending. Any other status — including the initial "idle" before the
    # orchestrator has reached the gate — returns 409 so the client surfaces
    # a clean error rather than a silent no-op.
    status = getattr(tok, "approval_status", "idle")
    if status != "waiting":
        raise HTTPException(
            status_code=409,
            detail=f"approval gate not awaiting input (status={status})",
        )

    sanitized: list[str] | None = None
    if payload.sub_queries is not None:
        if len(payload.sub_queries) == 0:
            raise HTTPException(
                status_code=400, detail="sub_queries must be non-empty when provided"
            )
        if len(payload.sub_queries) > _APPROVAL_MAX_SUB_QUERIES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"too many sub_queries: {len(payload.sub_queries)} "
                    f"(max={_APPROVAL_MAX_SUB_QUERIES})"
                ),
            )
        sanitized = []
        for q in payload.sub_queries:
            cleaned = _sanitize_sub_query(q)
            if cleaned is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "sub_query is empty or exceeds "
                        f"{_APPROVAL_MAX_QUERY_CHARS} chars after sanitization"
                    ),
                )
            sanitized.append(cleaned)

    tok.approved_payload = {"sub_queries": sanitized} if sanitized is not None else None
    tok.approval_event.set()
    return {
        "approved": True,
        "edited": sanitized is not None,
        "sub_queries": sanitized,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "version": "v2"}


@app.get("/health/providers")
async def health_providers(force: bool = False):
    """Per-provider reachability snapshot + today's usage counters. Cached
    for HEALTH_CACHE_TTL_S (default 60s) so a burst of page visits triggers
    exactly one probe set. Pass ?force=1 to bypass the cache.

    V3.9: `usage_today` joins documented free-tier quotas with running totals
    from the `provider_usage` table so deployers can see "Gemini: 850k/1M
    tokens used today" *before* an eval run starts.
    """
    from utils.provider_health import get_health
    from utils import provider_usage
    from utils.provider_router import _GROQ_ROTATOR, _GEMINI_ROTATOR
    snap = await get_health(force=force)
    out = snap.to_dict()
    out["usage_today"] = await provider_usage.snapshot()
    out["groq_key_rotator"] = _GROQ_ROTATOR.snapshot()
    out["gemini_key_rotator"] = _GEMINI_ROTATOR.snapshot()
    return out


@app.get("/settings/defaults")
async def settings_defaults():
    """Effective config when no per-request override is sent.
    The settings page reads this as 'baseline' so the user can see what would
    happen without overrides — and which knobs are currently runtime-toggleable.
    """
    cfg = RuntimeConfig.from_overrides(None)
    from utils.retrieval_mode import resolve, RetrievalMode
    # Compute effective mode under each user-selectable choice so the UI can
    # show "what would actually happen if I picked this" without re-probing.
    vec = _memory._VEC_AVAILABLE
    effective_per_choice: dict[str, str] = {}
    for choice in ("auto", "hybrid", "lexical"):
        try:
            eff = resolve(RetrievalMode(choice), vec)
            effective_per_choice[choice] = eff.effective.value
        except RuntimeError:
            # `hybrid` raises when sqlite-vec is unavailable — mark as inert.
            effective_per_choice[choice] = "unavailable"
    return {
        "effective": cfg.as_dict(),
        "knobs": [
            {
                "key": "RETRIEVAL_MODE",
                "label": "Retrieval mode",
                "type": "enum",
                "choices": ["auto", "hybrid", "lexical"],
                "available": True,
                "effective_per_choice": effective_per_choice,
                "note": (
                    "auto = use hybrid (BM25 + sqlite-vec RRF) when the "
                    "extension is available, fall back to lexical otherwise. "
                    "hybrid = require it (fail-loud). lexical = BM25 + FlashRank only."
                ) + (
                    "" if vec
                    else "  Note: sqlite-vec is NOT loadable on this Python build, "
                    "so 'hybrid' would fail and 'auto' falls back to lexical."
                ),
            },
            {
                "key": "FAILURE_POLICY_MAX_HOPS",
                "label": "Max retrieval hops",
                "type": "integer",
                "min": 1, "max": 3,
            },
            {
                "key": "CONTEXT_SELECTION_STRATEGY",
                "label": "Selection strategy",
                "type": "enum",
                "choices": ["heuristic", "mmr"],
            },
            {
                "key": "MMR_LAMBDA",
                "label": "MMR relevance/diversity tradeoff (λ)",
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "note": (
                    "Only applies when CONTEXT_SELECTION_STRATEGY=mmr. "
                    "λ=1.0 = pure relevance, λ=0.0 = pure novelty. "
                    "Default 0.7 leans relevance with meaningful redundancy penalty."
                ),
            },
            {
                "key": "RETRIEVAL_DOMAIN_BLOCKLIST",
                "label": "Domain blocklist (extends default)",
                "type": "string",
                "note": (
                    "Comma-separated additional domains to block. "
                    "Default already blocks: "
                    "reddit.com, twitter.com, x.com, tiktok.com, "
                    "quora.com, instagram.com, facebook.com, pinterest.com. "
                    "Use 'none' to disable the blocklist entirely."
                ),
            },
            # --- Routing knobs (Phase 6 polish): expose provider preferences so
            # reviewers can A/B-test the routing decisions live without re-deploy.
            {
                "key": "PLANNER_PROVIDER",
                "label": "Planner provider",
                "type": "enum",
                "choices": ["auto", "cerebras", "groq", "gemini"],
                "note": "auto = capability-probed cascade. Explicit choice forces that provider for the planning call.",
            },
            {
                "key": "CLAIM_VERIFIER_PROVIDER",
                "label": "Claim verifier provider",
                "type": "enum",
                "choices": ["auto", "deepseek", "gpt4o"],
                "note": "Used to re-check whether each claim's cited snippet actually supports it.",
            },
            {
                "key": "CONFLICT_PROBE_PROVIDER",
                "label": "Conflict probe provider",
                "type": "enum",
                "choices": ["auto", "cerebras", "groq"],
                "note": "Short LLM call that surfaces disagreements between sources.",
            },
            {
                "key": "FOLLOW_UP_PROVIDER",
                "label": "Follow-up suggestion provider",
                "type": "enum",
                "choices": ["auto", "cerebras", "groq"],
                "note": "Generates the 2–3 follow-up questions surfaced under the answer.",
            },
            # --- Approval gate
            {
                "key": "APPROVAL_TIMEOUT_S",
                "label": "Approval gate timeout (seconds)",
                "type": "integer",
                "min": 30,
                "max": 1800,
                "note": "How long a paused plan-approval gate waits before auto-cancelling.",
            },
            # --- Supplementary provider toggles
            {
                "key": "SARVAM_INDIC_AUTO",
                "label": "Auto-route Indic queries to Sarvam",
                "type": "boolean",
                "note": "When on, queries detected as Indic-language route through Sarvam's translation + retrieval path.",
            },
            {
                "key": "LANG_DETECT_DISABLE_FASTTEXT",
                "label": "Disable fastText language detection",
                "type": "boolean",
                "note": "Falls back to the lighter Tier 1/2 detector. Useful on hosts where fastText isn't installable.",
            },
            {
                "key": "WIKIPEDIA_DISABLED",
                "label": "Disable Wikipedia supplementary provider",
                "type": "boolean",
            },
            {
                "key": "SCHOLAR_DISABLED",
                "label": "Disable Scholar supplementary provider",
                "type": "boolean",
            },
            {
                "key": "JINA_READER_DISABLED",
                "label": "Disable Jina Reader fallback fetcher",
                "type": "boolean",
            },
        ],
    }


@app.get("/sessions")
async def get_sessions():
    # Join the first turn's `query` per session so the sidebar can render a
    # human-friendly title instead of a hash slug. Uses a correlated subquery
    # against `turns.created_at ASC LIMIT 1` to avoid N+1.
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            """
            SELECT s.session_id,
                   s.updated_at,
                   s.turn_count,
                   (SELECT t.query FROM turns t
                     WHERE t.session_id = s.session_id
                     ORDER BY t.created_at ASC LIMIT 1) AS first_query
              FROM sessions s
             ORDER BY s.updated_at DESC
             LIMIT 50
            """
        )
        out = []
        for r in rows:
            d = dict(r)
            fq = d.get("first_query")
            if isinstance(fq, str) and len(fq) > 200:
                d["first_query"] = fq[:200]
            out.append(d)
        return out


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            """
            SELECT t.turn_id, t.session_id, t.query, t.response, t.created_at,
                   t.state_trace, t.doc_map, t.search_queries, t.urls_opened,
                   t.context_xml_sent, t.citation_integrity_score, t.claim_precision_score,
                   t.prompt_tokens, t.completion_tokens, t.latency_ms,
                   t.planning_ms, t.search_ms, t.fetch_ms, t.select_ms, t.synthesize_ms,
                   t.run_metadata_json,
                   p.probe_ms AS probe_ms
              FROM turns t
              LEFT JOIN contradiction_probes p ON p.turn_id = t.turn_id
             WHERE t.session_id = ? ORDER BY t.created_at ASC
            """,
            (session_id,),
        )
        out = []
        for r in rows:
            d = dict(r)
            for k in (
                "doc_map",
                "state_trace",
                "search_queries",
                "urls_opened",
                "run_metadata_json",
            ):
                if d.get(k):
                    try:
                        d[k] = json.loads(d[k])
                    except (TypeError, ValueError):
                        pass
            out.append(d)
        return out


@app.get("/sessions/{session_id}/turns/{turn_id}")
async def get_turn_detail(session_id: str, turn_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        t = await db.execute_fetchall(
            "SELECT * FROM turns WHERE session_id = ? AND turn_id = ? LIMIT 1",
            (session_id, turn_id),
        )
        if not t:
            raise HTTPException(status_code=404, detail="turn not found")
        turn = dict(t[0])
        for k in ("doc_map", "state_trace", "search_queries", "urls_opened", "run_metadata_json"):
            if turn.get(k):
                try:
                    turn[k] = json.loads(turn[k])
                except (TypeError, ValueError):
                    pass

        ca = await db.execute_fetchall(
            "SELECT * FROM claim_audit WHERE turn_id = ? ORDER BY claim_idx ASC", (turn_id,)
        )
        claim_audit = [dict(r) for r in ca]
        for c in claim_audit:
            if c.get("cited_doc_ids"):
                try:
                    c["cited_doc_ids"] = json.loads(c["cited_doc_ids"])
                except (TypeError, ValueError):
                    pass

        p = await db.execute_fetchall(
            "SELECT * FROM contradiction_probes WHERE turn_id = ? LIMIT 1", (turn_id,)
        )
        probe = dict(p[0]) if p else None
        if probe and probe.get("contradictions_json"):
            try:
                probe["contradictions_json"] = json.loads(probe["contradictions_json"])
            except (TypeError, ValueError):
                pass

        # V3.11: surface per-snippet trust signals (V2.3 source trust prior) in
        # the trace inspector. Frontend renders these as tier-colored dots in
        # the Sources tab. Tier is recomputed server-side via source_trust.trust_for
        # because `turn_context` only persists the score, not the tier name.
        ctx_rows = await db.execute_fetchall(
            "SELECT doc_id, url, title, domain, snippet, bm25_score, "
            "recency_score, final_score, trust_score, provider_relevance, "
            "provider_relevance_source "
            "FROM turn_context WHERE turn_id = ? ORDER BY final_score DESC",
            (turn_id,),
        )
        context_snippets: list[dict] = []
        if ctx_rows:
            from utils.source_trust import trust_for
            for r in ctx_rows:
                row = dict(r)
                _, tier = trust_for(row.get("domain") or "")
                row["trust_tier"] = tier
                context_snippets.append(row)

    # Flatten into a single object matching frontend `TurnDetail` shape:
    # turn fields at root + `claim_audit`, `contradiction_probes` (plural),
    # and `context_snippets` alongside.
    flat = dict(turn)
    flat["claim_audit"] = claim_audit
    flat["contradiction_probes"] = _adapt_probe(probe)
    flat["context_snippets"] = context_snippets
    return flat


# ── Eval endpoints ─────────────────────────────────────────────────────────
#
# The internal `agent.eval_queries` module returns SQL-canonical shapes:
#   - metric columns suffixed `_score` (e.g. `faithfulness_score`)
#   - per-run summary nested as `{by_category, failure_distribution, run_summary, ...}`
#   - question detail nested as `{eval_row, turn, claim_audit, contradiction_probe}`
#
# The Next.js frontend (`frontend/lib/types.ts`) consumes flat shapes with bare
# metric names (`faithfulness`, `avg_relevance`, `failure_class_distribution`).
# These adapter helpers reshape at the HTTP boundary so internal consumers
# (eval_runner.py, ablation_report.py, …) keep using the canonical form.

_EVAL_METRIC_RENAME = {
    "faithfulness_score": "faithfulness",
    "answer_relevance_score": "answer_relevance",
    "context_precision_score": "context_precision",
    "citation_integrity_score": "citation_integrity",
    "claim_precision_score": "claim_precision",
    "conflict_adherence_score": "conflict_adherence",
    "session_coherence_score": "session_coherence",
    "factual_accuracy_score": "factual_accuracy",
}

# Frontend `EvalRun.avg_*` keys (note: `avg_relevance`, not `avg_answer_relevance`).
_EVAL_AVG_RENAME = {
    "avg_faithfulness_score": "avg_faithfulness",
    "avg_answer_relevance_score": "avg_relevance",
    "avg_context_precision_score": "avg_context_precision",
    "avg_citation_integrity_score": "avg_citation_integrity",
    "avg_claim_precision_score": "avg_claim_precision",
    "avg_conflict_adherence_score": "avg_conflict_adherence",
    "avg_session_coherence_score": "avg_session_coherence",
    "avg_factual_accuracy_score": "avg_factual_accuracy",
}


def _rename_metric_keys(d: dict) -> dict:
    """Strip `_score` suffix from metric columns and `_score` from avg_* columns."""
    out = dict(d)
    for old, new in _EVAL_METRIC_RENAME.items():
        if old in out and new not in out:
            out[new] = out.pop(old)
    for old, new in _EVAL_AVG_RENAME.items():
        if old in out and new not in out:
            out[new] = out.pop(old)
    return out


def _adapt_probe(p: dict | None) -> dict | None:
    """Map backend contradiction_probes row → frontend `ContradictionProbeRow`."""
    if p is None:
        return None
    out = dict(p)
    if "contradictions_json" in out:
        out["contradictions"] = out.pop("contradictions_json")
    if "probe_skipped_reason" in out:
        out["skip_reason"] = out.pop("probe_skipped_reason")
    if "has_conflict" in out:
        out["has_conflict"] = bool(out["has_conflict"])
    return out


def _adapt_eval_question(row: dict) -> dict:
    """Map an `eval_runs` row → frontend `EvalQuestion`."""
    out = _rename_metric_keys(row)
    fc = out.get("failure_class")
    out["pass"] = (fc == "PASS") or (fc is None)
    return out


def _rename_category_metric_keys(d: dict) -> dict:
    """For per-category rows: backend uses `avg_X_score`, frontend
    `EvalSummaryCategoryRow` uses the un-suffixed metric name from
    `_EVAL_METRIC_RENAME` (e.g. `faithfulness`, `answer_relevance`)."""
    out = dict(d)
    for score_col, bare in _EVAL_METRIC_RENAME.items():
        avg_col = f"avg_{score_col}"
        if avg_col in out and bare not in out:
            out[bare] = out.pop(avg_col)
    return out


def _adapt_eval_summary(payload: dict) -> dict:
    """Map `eval_queries.get_run_summary(...)` → frontend `EvalSummary`."""
    rs = payload.get("run_summary") or {}
    failure_dist = payload.get("failure_distribution") or []
    failure_class_distribution = {r["failure_class"]: r["n"] for r in failure_dist}

    by_category = [
        _rename_category_metric_keys(r) for r in (payload.get("by_category") or [])
    ]

    total = sum(failure_class_distribution.values())
    pass_count = failure_class_distribution.get("PASS", 0)
    pass_rate = rs.get("pass_rate")
    if pass_rate is None:
        pass_rate = pass_count / total if total > 0 else 0.0

    return {
        "run_at": payload.get("run_at"),
        "pass_rate": pass_rate,
        "avg_faithfulness": rs.get("mean_faithfulness"),
        "avg_relevance": rs.get("mean_relevance"),
        "avg_context_precision": rs.get("mean_context_precision"),
        "avg_citation_integrity": rs.get("mean_citation_integrity"),
        "avg_claim_precision": rs.get("mean_claim_precision"),
        "avg_conflict_adherence": rs.get("mean_conflict_adherence"),
        "by_category": by_category,
        "failure_class_distribution": failure_class_distribution,
        # Extra sections preserved for any future UI use; frontend types ignore them.
        "cross_language": payload.get("cross_language"),
        "calibration": payload.get("calibration"),
        "run_summary": rs,
        # Tier A (Phase 1+): per-turn quality means surfaced to the dashboard.
        "per_turn_quality": payload.get("per_turn_quality"),
    }


def _adapt_question_detail(payload: dict) -> dict:
    """Map `eval_queries.get_question_detail(...)` → frontend `EvalQuestionDetail`.
    Flattens `{eval_row, turn, claim_audit, contradiction_probe}` into one object
    with renamed metric keys and `contradiction_probes` (plural to match TS type).
    """
    eval_row = _rename_metric_keys(payload.get("eval_row") or {})
    turn = payload.get("turn") or {}
    flat = dict(eval_row)
    for k in (
        "context_xml_sent",
        "doc_map",
        "prompt_tokens",
        "completion_tokens",
        "state_trace",
        "claim_verification_json",
    ):
        if k in turn and turn[k] is not None:
            flat[k] = turn[k]
    flat["claim_audit"] = payload.get("claim_audit") or []
    flat["contradiction_probes"] = _adapt_probe(payload.get("contradiction_probe"))
    fc = flat.get("failure_class")
    flat["pass"] = (fc == "PASS") or (fc is None)
    return flat


@app.get("/eval/runs")
async def list_runs():
    rows = await eval_queries.list_eval_runs()
    return [_rename_metric_keys(r) for r in rows]


@app.get("/eval/runs/{run_at}/summary")
async def run_summary(run_at: str):
    payload = await eval_queries.get_run_summary(unquote(run_at))
    return _adapt_eval_summary(payload)


@app.get("/eval/runs/{run_at}/questions")
async def run_questions(run_at: str):
    rows = await eval_queries.get_run_questions(unquote(run_at))
    return [_adapt_eval_question(r) for r in rows]


@app.get("/eval/runs/{run_at}/questions/{question_id}")
async def question_detail(run_at: str, question_id: str):
    detail = await eval_queries.get_question_detail(unquote(run_at), question_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="question not found in run")
    return _adapt_question_detail(detail)


# ── Smoke eval ────────────────────────────────────────────────────────────
#
# V3.10: lets a reviewer trigger a real 8-question eval from the browser.
# Designed for the deployed HF Space where running `python eval/eval_runner.py`
# from the URL is obviously not an option.
#
# Constraints:
#   - 15-minute cooldown per IP (drive-by-click protection)
#   - Single concurrent runner (one eval at a time across all visitors)
#   - Quota gate: refuses if today's Gemini requests > 80% of free-tier limit
#   - 8 questions ≈ 2-3 minutes; bounded SSE connection lifetime
#
# The 8 are picked to cover all 6 categories + a multi-turn pair + an extra
# conflicting-source case (the differentiator). The full 53-question eval
# remains CLI-only for reproducibility — `python eval/eval_runner.py`.

_SMOKE_QUESTION_IDS = [
    "F-1",       # factual: RBI repo rate
    "MH-1",      # multi-hop
    "C-1",       # comparison
    "IE-1",      # insufficient evidence
    "CF-1",      # conflicting sources
    "CF-2",      # second conflicting case (differentiator showcase)
    "MT-1-T1",   # multi-turn — turn 1
    "MT-1-T2",   # multi-turn — turn 2 (session coherence)
]
_SMOKE_COOLDOWN_S = 900  # 15 minutes per IP
_SMOKE_QUOTA_GATE_PCT = 0.8  # refuse if Gemini > 80% of daily quota
_smoke_lock = asyncio.Lock()
# Audit M1: bound the dict at 1000 entries (LRU) so abusive request volume
# can't grow it without limit. client_ip → unix-ts of last successful start.
_SMOKE_LAST_RUN_MAX = 1000
_smoke_last_run: "OrderedDict[str, float]" = OrderedDict()

# Audit M1: number of leading X-Forwarded-For hops to skip as "trusted proxy".
# Default 0 means "don't trust the header" — only use request.client.host. When
# deployed behind a known reverse proxy (Cloudflare, HF Spaces edge), set this
# to the number of trusted hops so the real client IP is used for rate limits.
_TRUSTED_PROXY_COUNT = int(os.environ.get("TRUSTED_PROXY_COUNT", "0"))


def _smoke_client_ip(request: Request) -> str:
    """Resolve the client IP for rate-limiting. With TRUSTED_PROXY_COUNT > 0,
    walks X-Forwarded-For from the right (closest proxy first) and skips N
    leading entries, returning the leftmost remaining hop. Falls back to
    `request.client.host` when the header is absent or count is 0."""
    if _TRUSTED_PROXY_COUNT > 0:
        xff = request.headers.get("x-forwarded-for") or ""
        hops = [h.strip() for h in xff.split(",") if h.strip()]
        # XFF is left-to-right: original client first. Trusted proxies append
        # themselves at the right edge. Skip the last N (trusted) hops; the
        # rightmost remaining hop is the closest non-trusted client.
        remaining = hops[: max(0, len(hops) - _TRUSTED_PROXY_COUNT)]
        if remaining:
            return remaining[-1]
    return request.client.host if request.client else "unknown"


def _smoke_record_run(ip: str, now: float) -> None:
    """LRU-record this IP's run start, evicting oldest if over the cap."""
    if ip in _smoke_last_run:
        _smoke_last_run.move_to_end(ip)
    _smoke_last_run[ip] = now
    while len(_smoke_last_run) > _SMOKE_LAST_RUN_MAX:
        _smoke_last_run.popitem(last=False)


async def _smoke_quota_ok() -> tuple[bool, str | None]:
    """Returns (allowed, reason_if_blocked). Allowed=True when Gemini is
    below the threshold or we can't determine usage."""
    try:
        from utils import provider_usage
        rows = await provider_usage.snapshot()
        gemini = next((r for r in rows if r["provider"] == "gemini"), None)
        if not gemini:
            return True, None
        limit = gemini.get("request_limit") or 0
        used = gemini.get("requests", 0)
        if limit and used / limit > _SMOKE_QUOTA_GATE_PCT:
            return False, (
                f"Gemini daily usage at {used}/{limit} (>{int(_SMOKE_QUOTA_GATE_PCT*100)}%); "
                "refusing smoke run to protect remaining quota."
            )
    except Exception:
        pass
    return True, None


async def _smoke_eval_stream(request: Request):
    """SSE generator that drives the smoke eval. Emits `start`, periodic
    `progress`, and a final `done` (or `error`) event."""
    client_ip = _smoke_client_ip(request)

    # ── Cooldown ──
    now = asyncio.get_running_loop().time()
    last = _smoke_last_run.get(client_ip, 0.0)
    if now - last < _SMOKE_COOLDOWN_S:
        wait_s = int(_SMOKE_COOLDOWN_S - (now - last))
        yield _format_event({
            "step": "error", "label": "cooldown",
            "data": f"Try again in {wait_s}s. The smoke eval is rate-limited to one run per IP per 15 minutes.",
        })
        return

    # ── Single-runner ──
    if _smoke_lock.locked():
        yield _format_event({
            "step": "error", "label": "busy",
            "data": "Another smoke eval is in progress. Please wait for it to finish (~2-3 min).",
        })
        return

    # ── Quota gate ──
    ok, reason = await _smoke_quota_ok()
    if not ok:
        yield _format_event({"step": "error", "label": "quota", "data": reason})
        return

    async with _smoke_lock:
        _smoke_record_run(client_ip, now)
        run_at_holder: dict = {"value": None}

        async def _do_run():
            from eval.eval_runner import run_eval
            run_at_holder["value"] = await run_eval(question_ids=_SMOKE_QUESTION_IDS)

        task = asyncio.create_task(_do_run())

        yield _format_event({
            "step": "start",
            "label": "Starting smoke eval",
            "data": {
                "n_questions": len(_SMOKE_QUESTION_IDS),
                "question_ids": _SMOKE_QUESTION_IDS,
                "estimated_seconds": 180,
            },
        })

        # Poll progress every 3s. We can't know the exact per-question state
        # without an instrumentation channel into run_eval; we approximate by
        # counting rows whose run_at is the newest in the DB.
        start_wall = asyncio.get_running_loop().time()
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
            except asyncio.TimeoutError:
                pass
            elapsed = int(asyncio.get_running_loop().time() - start_wall)
            done_count = await _smoke_completed_count(start_wall_iso=None)
            yield _format_event({
                "step": "progress",
                "label": f"Running ({elapsed}s)",
                "data": {
                    "completed": done_count,
                    "total": len(_SMOKE_QUESTION_IDS),
                    "elapsed_s": elapsed,
                },
            })

        try:
            await task  # surface exceptions
        except Exception:
            # S4 fix: don't echo raw exception text to the client.
            logger.exception("smoke eval failed")
            yield _format_event({"step": "error", "label": "failed", "data": "Smoke eval failed."})
            return

        yield _format_event({
            "step": "done",
            "label": "Done",
            "data": {"run_at": run_at_holder["value"]},
        })


async def _smoke_completed_count(start_wall_iso: str | None) -> int:
    """Count rows in eval_runs whose run_at is the most-recent timestamp
    (i.e. belonging to the currently-running smoke eval). Returns 0 on any
    error so a flaky DB read can't kill the SSE stream."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            row = await db.execute_fetchall(
                "SELECT run_at, COUNT(*) AS n FROM eval_runs "
                "GROUP BY run_at ORDER BY run_at DESC LIMIT 1"
            )
            return row[0]["n"] if row else 0
    except Exception:
        return 0


@app.post("/eval/smoke")
async def eval_smoke(request: Request):
    """Trigger an 8-question smoke eval and stream progress via SSE.
    See `_smoke_eval_stream` for the contract (start/progress/done/error events)."""
    return StreamingResponse(
        _smoke_eval_stream(request),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# Mount static files at the root (legacy vanilla-JS frontend)
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
