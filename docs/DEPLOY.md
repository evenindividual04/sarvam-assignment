# Deploy in Your Own Cloud

## When to use this guide

For organizations that need to run the Deep Research Agent inside their own
infrastructure for data-residency, compliance, or air-gap requirements. This
is not a SaaS — it is a self-hostable Python service with a thin React
frontend. Common drivers:

- Regulated workloads (RBI / SEBI / HIPAA-equivalent) where conversation
  history and retrieved web content must not leave the VPC.
- Existing observability stack the team already trusts (Datadog, Splunk, ELK)
  that you want the agent's JSON logs to feed into.
- A captive LLM contract — e.g. you already pay for Sarvam Model API or run
  a Gemini private endpoint — and want this agent to use those credits
  rather than a third-party.

If none of those apply, the public demo (Vercel + HF Spaces) is fine.

## 30-minute deployment

### Prerequisites

- A Linux host with Docker + Docker Compose (4 vCPU, 8 GB RAM minimum).
- Outbound HTTPS to:
  - `api.parallel.ai` (search; primary)
  - `api.sarvam.ai` (synthesis, if `SYNTH_PROVIDER=sarvam`)
  - `generativelanguage.googleapis.com` (synthesis, if `SYNTH_PROVIDER=gemini`)
  - `api.groq.com` (planning + conflict detection)
  - `models.inference.ai.azure.com` (eval judge — GitHub Models)
- At least one LLM API key. For data residency we recommend Sarvam.
- One search provider key (Parallel recommended; Tavily / Serper supported as
  fallbacks).

### Step 1 — Clone and configure

```bash
git clone <your-fork-or-this-repo> && cd sarvam
cp .env.example .env
$EDITOR .env
```

Set, at minimum:

```
PARALLEL_API_KEY=...
GROQ_API_KEY=...
GITHUB_TOKEN=...
# Pick exactly one synthesizer for default operation:
GEMINI_API_KEY=...        # if SYNTH_PROVIDER=gemini (default)
SARVAM_API_KEY=sk_...     # if SYNTH_PROVIDER=sarvam
SYNTH_PROVIDER=gemini     # or sarvam, or openrouter
```

Persistent storage:

```
DB_PATH=/data/research.db  # mount /data onto a real volume
```

### Step 2 — Launch

```bash
docker-compose up -d
curl http://localhost:7860/health
# {"status": "ok", ...}
```

The backend listens on port 7860 (FastAPI + uvicorn).

### Step 3 — Point the frontend

The React frontend is a separate Next.js app under `frontend/`. To host it
inside your VPC:

```bash
cd frontend
npm install
NEXT_PUBLIC_BACKEND_URL=https://research.your-org.internal npm run build
# Serve .next/ via your existing static host (nginx, Caddy, S3 + CloudFront, etc.)
```

Lock CORS down to that exact origin on the backend by setting
`ALLOWED_ORIGINS=https://research.your-org.internal` in `.env`.

## Data residency

| Data | Where it lives | Retention |
|------|---------------|-----------|
| Conversation history (sessions, turns) | SQLite file at `DB_PATH` | Forever, until you delete |
| Fetched web pages | In-memory Trafilatura extracts; never persisted | Per turn only |
| Eval results | Same SQLite file: `eval_runs`, `claim_audit`, `contradiction_probes`, `cross_language_consistency`, `circuit_events` | Forever |
| LLM call payloads | Stdlib `logging` with JSON formatter; pipe to your SIEM | Per your retention policy |
| API keys | Environment variables; never written to disk by the agent | n/a |

To keep ALL processing inside your VPC:

1. Set `SYNTH_PROVIDER=sarvam` and contract with Sarvam for their
   enterprise / private deployment of the Model API (Sarvam's stack
   supports VPC isolation per their go-to-market docs).
2. Replace Parallel / Tavily / Serper with an internal search index if
   web egress is a concern. `agent/search.py` is provider-pluggable —
   add a new `_search_internal()` async function returning
   `list[SearchResult]` and wire it into `_search_single_query`.
3. Mount `DB_PATH` onto an encrypted-at-rest volume on your storage tier
   of choice.

## Provider swapping

The agent supports multiple providers per stage. Switch via env var:

| Stage | Default | Alternatives | Env var |
|-------|--------|--------------|---------|
| Planner | Groq Llama 3.3 70B | (Groq only at present) | `GROQ_API_KEY` |
| Synth | Gemini 2.5 Flash | Sarvam-M / Sarvam-30B, OpenRouter DeepSeek R1 | `SYNTH_PROVIDER`, `SARVAM_MODEL` |
| Search | Parallel | Tavily, Serper (set the corresponding `*_API_KEY`) | `PARALLEL_API_KEY`, `TAVILY_API_KEY`, `SERPER_API_KEY` |
| Judge (eval) | GitHub Models GPT-4o-mini | (any OpenAI-compatible — change `base_url` in `judge()`) | `GITHUB_TOKEN` |

Sarvam-specific:

- `SARVAM_API_KEY` — Sarvam Model API key (`sk_...`).
- `SARVAM_MODEL` — `sarvam-m` (default), `sarvam-30b`, or `sarvam-105b`.
- The Sarvam path automatically falls back from `sarvam-m` to `sarvam-30b`
  on first-call failure before raising. Subsequent failures trip the
  per-provider circuit breaker (`sarvam`, threshold 4 in 60s).

## Observability

The agent emits structured JSON log lines per turn with:

- `session_id`, `turn_id`, `module`, `latency_ms`
- per-stage `prompt_tokens` / `completion_tokens`
- `fallback_path_taken` (which provider answered after fallbacks)
- circuit breaker state transitions (`circuit_transition` events,
  also persisted to the `circuit_events` table)

Redirect stdout to your existing log aggregator. The
`/eval/runs/{run_at}/summary` endpoint exposes aggregate stats over the
SQLite `eval_runs` table — wire this into your analytics dashboard or
scheduled job for ongoing quality tracking.

## Scaling notes

For >100 concurrent users:

- **SQLite is the bottleneck.** Migrate to Postgres. `agent/memory.py`
  uses `aiosqlite`; swap to `asyncpg` and adjust the SQL flavour
  (placeholders, JSON columns, FTS replacement with `tsvector`).
- **Per-turn LLM cost dominates.** Enable `HYBRID_RETRIEVAL=1` for better
  context precision (fewer wasted synth tokens). Confirm your container
  base image supports SQLite extension loading for `sqlite-vec`.
- **Horizontal scale**: the orchestrator is stateless per-turn. Run N
  replicas behind a load balancer with sticky sessions — the
  cancellation registry is per-process, so cross-replica cancel won't
  work without a shared registry (Redis pub/sub is the obvious
  extension).

## Security checklist

- [ ] API keys in env vars or your secret manager — never committed.
- [ ] `DB_PATH` backed up and encrypted at rest.
- [ ] HTTPS terminated at a reverse proxy (Caddy, nginx, or your existing
      ingress controller).
- [ ] `ALLOWED_ORIGINS` scoped to your specific frontend domain.
- [ ] Outbound DNS / firewall allow-list pinned to the providers actually
      in use (don't leave Gemini reachable if `SYNTH_PROVIDER=sarvam`).
- [ ] If extending to multi-tenant: add `session_id` namespacing and an
      auth middleware (the existing FastAPI app has no auth — by design,
      for the single-tenant deployment model).

## Support escalation

For issues during deployment:

1. Hit `/health` — confirms the FastAPI process is up.
2. Inspect the `circuit_events` table — surfaces provider outages and
   breaker transitions across the last N turns.
3. Inspect `run_metadata_json` on a specific turn for
   `fallback_path_taken` to see which provider chain executed.
4. Open an issue on the GitHub repository, or reach the deployment
   engineer named in your kickoff packet.
