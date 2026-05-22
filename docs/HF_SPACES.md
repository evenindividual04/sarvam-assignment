# Deploy to Hugging Face Spaces (Public Demo)

This guide covers the **public demo deployment**: a Next.js frontend on Vercel
talking to a FastAPI backend on Hugging Face Spaces (Docker SDK). For
self-hosting in your own VPC (regulated workloads, captive LLM contracts,
data-residency), see [DEPLOY.md](./DEPLOY.md) instead.

## 1. Architecture overview

```
┌────────────────┐    HTTPS    ┌─────────────────────┐    HTTPS    ┌──────────────┐
│  Vercel        │ ──────────▶ │  HF Spaces (Docker) │ ──────────▶ │  Provider    │
│  (frontend)    │             │  FastAPI + SQLite   │             │  APIs        │
│  Next.js 15    │ ◀────────── │  agent/orchestrator │             │  (Parallel,  │
│                │   SSE       │                     │             │   Gemini,    │
└────────────────┘             └─────────────────────┘             │   Groq, ...) │
                                                                    └──────────────┘
```

- **Vercel** serves the React UI, calls the backend at `NEXT_PUBLIC_API_BASE_URL`.
- **HF Spaces** runs the agent (FastAPI in a container) and persists session
  + eval data to SQLite.
- **External APIs** are called from HF Spaces over outbound HTTPS.

## 2. Space setup

1. Create an HF account: <https://huggingface.co/join>.
2. New Space → **SDK: Docker** (not Gradio, not Static).
3. Visibility: Public (free) or Private (free for personal accounts).
4. Hardware: free CPU tier (2 vCPU, 16 GB RAM) is sufficient for the demo.
5. Connect your GitHub repo or push directly with `git push hf main`.

## 3. Dockerfile

The repo ships with a production-ready `Dockerfile` at the root. HF Spaces
detects it automatically — no `app_file` config needed.

```
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=7860
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

HF Spaces expects port **7860**.

## 4. Secrets configuration

Set every env var as an HF Spaces **Secret** (Settings → Variables and secrets
→ New secret). Never commit them to the repo.

### Required

| Secret name | Purpose |
|-------------|---------|
| `PARALLEL_API_KEY` | Primary search provider (16k free queries) |
| `GEMINI_API_KEY` | Synthesis (Gemini 2.5 Flash), single-key mode |
| `GEMINI_API_KEYS` | **OR** multi-key mode: `AIzaSy_xxx,AIzaSy_yyy` — round-robin across N keys gives N× the 1500/day quota; takes precedence over `GEMINI_API_KEY` if both set |
| `GROQ_API_KEY` | Planning + conflict probe (single-key mode) |
| `GROQ_API_KEYS` | **OR** multi-key mode: `gsk_aaa,gsk_bbb,gsk_ccc` (preferred for eval ablations; takes precedence over `GROQ_API_KEY` if both set) |
| `GITHUB_TOKEN` | GitHub Models eval judge (GPT-4o-mini, different family from generator) |

### Recommended (fallbacks + secondary providers)

| Secret name | Purpose |
|-------------|---------|
| `TAVILY_API_KEY` | Search fallback when Parallel quota is exhausted |
| `SERPER_API_KEY` | Last-resort search + Serper Scholar for academic intent |
| `OPENROUTER_API_KEY` | Synthesis fallback (DeepSeek R1) when Gemini rate-limits |
| `CEREBRAS_API_KEY` | Very fast Llama inference (planner, 8K context cap auto-detected) |
| `COHERE_API_KEY` | Rerank v3.5 between FlashRank and 5-factor scoring (English only) |
| `SARVAM_API_KEY` | Indic-first synthesizer; auto-routes for Devanagari/Tamil/Bengali |

### Tuning (string values)

| Secret name | Purpose |
|-------------|---------|
| `CONFLICT_PROBE_PROVIDER` | `groq` / `cerebras` / `auto` |
| `FOLLOW_UP_PROVIDER` | Follow-up generation provider override |
| `PLANNER_PROVIDER` | `auto` / `cerebras` / `groq` / `gemini` |
| `CLAIM_VERIFIER_PROVIDER` | `auto` / `deepseek` / `gpt4o` |
| `SARVAM_INDIC_AUTO` | `1` (default) auto-routes Indic queries to Sarvam |

### Deployment + ops

| Secret name | Purpose |
|-------------|---------|
| `ALLOWED_ORIGINS` | CORS allow-list, **set to your Vercel URL** (not `*`) |
| `DB_PATH` | SQLite path; default `research.db` (see §5 on persistence) |
| `KEY_ROTATION_REFRESH_S` | Lazy env reload interval (default 60s; `0` disables) |
| `KEY_ROTATION_PERSIST` | `1` to persist throttle state to SQLite across restarts |
| `QUOTA_WEBHOOK_URL` | Slack/Discord webhook for quota-exhaustion alerts |
| `QUOTA_ALERT_THRESHOLD` | Fraction (`0.8` = 80%); default 0.8 |
| `QUOTA_ALERT_COOLDOWN_S` | Re-alert cooldown per provider; default 3600 |

## 5. Persistent storage

**Free tier**: HF Spaces **has no persistent disk** — every restart wipes
SQLite. This is fine for a public demo (sessions are ephemeral by design) but
expect:

- Per-session history lost on restart.
- Eval `results/` directory lost; export before the Space sleeps.
- Provider-usage counters reset; quota-tracking restarts from zero.

**Workarounds**:

1. **HF Spaces Pro** ($9/mo) → mounts `/data` as persistent. Set
   `DB_PATH=/data/research.db`.
2. **External Postgres** (Neon, Supabase, Railway) — replace SQLite by
   pointing `DB_PATH` at an external file is not enough; you'd need to swap
   the `aiosqlite` driver for `asyncpg`. Out of scope here.
3. **Accept ephemeral** — for the assignment demo, this is the right call.

## 6. CORS

Set `ALLOWED_ORIGINS` to **your exact Vercel domain**:

```
ALLOWED_ORIGINS=https://deep-research-agent.vercel.app
```

Never use `*` in production — it disables the browser's same-origin
protections. Multiple origins are comma-separated.

## 7. Multi-key rotation in HF Spaces

To stitch quota from multiple Groq accounts:

1. Set `GROQ_API_KEYS=gsk_aaa,gsk_bbb,gsk_ccc` in HF secrets.
2. Set `KEY_ROTATION_REFRESH_S=60` (default — already on).
3. To add a fourth key without restart:
   - Update the secret to `gsk_aaa,gsk_bbb,gsk_ccc,gsk_ddd`.
   - Within 60 seconds the rotator picks it up — no rebuild, no downtime.
4. To remove a leaked key: same flow, just delete the entry.
5. Set `KEY_ROTATION_PERSIST=1` so 429 cooldowns survive container restarts
   (otherwise a restart at T+5s of a T+60s cooldown will re-hammer the
   throttled key).

## 8. Cold-start behavior

Free Spaces sleep after **48 hours of inactivity**. Cold-start takes
30–60 seconds (image pull + Python imports + DB init).

**Warmup pattern** (optional):

- Cron a GET to `https://<your-space>.hf.space/health` every 6 hours from
  GitHub Actions or an UptimeRobot free monitor.
- Frontend can show a "Waking up..." state on the first 503 response.

## 9. Quota monitoring

Set `QUOTA_WEBHOOK_URL` to a Slack or Discord incoming webhook. When any
provider crosses `QUOTA_ALERT_THRESHOLD` (default 80%) of its documented
daily quota, the agent POSTs:

```json
{
  "text": "groq at 87.5% of daily quota (875000/1000000 tokens)",
  "provider": "groq",
  "fraction": 0.875,
  "used": 875000,
  "limit": 1000000,
  "unit": "tokens"
}
```

Cooldown (`QUOTA_ALERT_COOLDOWN_S`, default 1 hour) prevents alert spam. Both
Slack and Discord accept the `text` field directly; richer formatting is left
to the receiving end.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Build fails with `pip install` timeout | HF builder rate-limited | Retry; consider pre-built image |
| `KeyError: 'GEMINI_API_KEY'` at startup | Secret typo or not saved | Check Settings → Variables; redeploy |
| Frontend gets CORS error | `ALLOWED_ORIGINS` mismatch | Set to exact Vercel URL incl. `https://` |
| 503 on every request | Space is sleeping | First request wakes it (30–60s) |
| Sessions disappear randomly | Free-tier ephemeral disk | See §5 |
| Outbound to `api.parallel.ai` fails | HF egress firewall (rare) | File a support ticket; fall back to Tavily |
| `Groq 429` storm after restart | Throttle state lost | Set `KEY_ROTATION_PERSIST=1` |

## 11. Cost estimate

For the assignment demo, **all components fit in free tiers**:

| Component | Free quota | Monthly cost |
|-----------|------------|--------------|
| Vercel Hobby | 100 GB bandwidth | $0 |
| HF Spaces (free CPU) | unlimited (48h sleep) | $0 |
| Parallel Base | 16,000 queries | $0 |
| Gemini Flash | 1,500 req/day | $0 |
| Groq Llama 3.3 70B | ~1M tokens/day/key (×N keys) | $0 |
| GitHub Models | 150 req/day (judge) | $0 |
| **Total** | — | **$0** |

For reliable always-on production (no cold starts, persistent storage,
24/7 SLA):

- HF Spaces Pro: $9/mo
- Or self-host on a small VPS (`$5–10/mo`) per [DEPLOY.md](./DEPLOY.md)

The free stack is the right choice for the assignment and any demo with
predictable bursty traffic. Move to paid only when you have a reason — e.g.
the eval harness needs to be reproducible 24/7, or session history matters
across days.
