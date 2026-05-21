---
title: Deep Research Agent
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Web-grounded research with citation audit and conflict probe
---

# Deep Research Agent

A multi-source web research agent built with plain Python asyncio. Every claim is traced back to a real URL retrieved during the session. Conflicts between sources are surfaced, not hidden. Each generated claim is verified against its cited snippet at generation time.

**Live demo:**
- **Frontend (Next.js, Vercel):** https://frontend-a519j5gkh-evenindividual04s-projects.vercel.app
- **Backend (FastAPI, Hugging Face Spaces):** https://evenindividual00-sarvam-deep-research.hf.space
- **Source:** https://github.com/evenindividual04/sarvam-assignment

> **Deploying in your own org?** See [docs/DEPLOY.md](docs/DEPLOY.md) for a 30-minute VPC deployment guide covering data residency, provider swapping (including Sarvam Model API), observability, and scaling notes.

---

## Target Users and Problem

Enterprise researchers, analysts, and knowledge workers who need answers that go beyond a single search result. Standard LLMs answer from stale, unverifiable training data. Standard search engines return links but don't synthesize. This agent sits in between — it conducts live web research, evaluates evidence, and generates grounded answers where every factual claim is traced to a URL retrieved in that session.

Secondary users: developers building AI pipelines who want a reference implementation of a production-grade research agent built without orchestration frameworks (no LangChain / LangGraph / CrewAI / LlamaIndex / Haystack).

## Definition of "Deep Research"

1. **Multi-source triangulation** — the planner emits 2-4 typed search queries per question (`primary`, `comparison`, `recency_check`, `contradiction_probe`, `definition`), fetching and evaluating content from multiple domains.
2. **Evidence-grounded generation** — the synthesizer is never allowed to answer from training data. Every claim must attribute to a specific retrieved chunk.
3. **Conflict-aware synthesis** — a dedicated cross-source contradiction probe runs as its own pipeline stage between retrieval and synthesis. When real disagreements are detected (distinguished from temporal evolution), the answer surfaces both positions and cites both sources.
4. **Claim-level verification** — every sentence-with-citation is checked against its cited snippet at generation time. Unsupported claims get an `[UNVERIFIED]` marker; the per-claim audit is queryable in the eval drill-down UI.
5. **Adaptive 2-hop retrieval** — the planner emits a confidence signal; on low confidence with sparse context, a single bounded second hop fires with refined `recency_check` / `contradiction_probe` queries. Hard cap `MAX_HOPS=2` prevents infinite loops.
6. **Session continuity** — prior conversation turns are persisted in SQLite; the most relevant prior turns are retrieved via FTS5 keyword search into context for follow-up questions.

## Architecture

```text
User Query
    |
    v
[PLANNER]      Groq Llama 3.3 70B
               → typed queries (primary | comparison | recency_check | contradiction_probe)
               → confidence: low | medium | high
    |
    v
[SEARCHER]     Parallel AI (primary) → Tavily → Serper (fallback chain)
               Per-intent routing; per-provider circuit breakers
    |
    v
[FETCHER]      httpx async, semaphore(3), Trafilatura readability extract
    |
    v
[CONTEXT]      BM25 → FlashRank cross-encoder rerank → 3-factor scoring
               (relevance + recency + diversity + source trust prior)
               Optional V3.1: hybrid RRF with bge-small-en-v1.5 via sqlite-vec
    |
    v
[CONFLICT_CHECK]  Dedicated stage: detects cross-source contradictions
                  Distinguishes contradiction from temporal evolution
    |
    v
[SYNTHESIZER]  Gemini 2.5 Flash (default) | Sarvam-M/30B | OpenRouter DeepSeek R1
               Streams [doc_N] markers; citation guard converts to [Title — domain](URL)
    |
    v
[CLAIM VERIFIER]  Per-sentence: deterministic token+entity overlap → LLM fallback
                  Unsupported claims appended with [UNVERIFIED]
    |
    v
[PERSISTENCE]  aiosqlite: sessions, turns, turn_context, claim_audit,
               contradiction_probes, circuit_events, eval_runs, FTS5 index
```

## Providers (multi-stage, swappable)

| Stage | Default | Alternatives |
|---|---|---|
| Planning + conflict detection | Groq Llama 3.3 70B | (Groq) |
| Search | Parallel AI | Tavily, Serper (auto fallback) |
| Synthesis | Gemini 2.5 Flash | `SYNTH_PROVIDER=sarvam` → Sarvam-M / Sarvam-30B (Indic-first); `SYNTH_PROVIDER=openrouter` → DeepSeek R1 |
| Eval judge | GitHub Models GPT-4o-mini | (any OpenAI-compatible — different model family from generator required) |

**Sarvam Model API** (Sarvam-M default, Sarvam-30B / 105B available, 64K-128K context, Apache-2.0 base models) is the recommended synthesizer for Indic-heavy workloads and data-residency-sensitive deployments. See [docs/DEPLOY.md](docs/DEPLOY.md).

## V2 / V3 Production Features

| | Feature | Why it matters |
|---|---|---|
| **V2.1** | Typed query decomposition | Planner emits intent-tagged queries; downstream stages dispatch per intent |
| **V2.2** | Contradiction probe as a named pipeline stage | Explicit `CONFLICT_CHECK` in `state_trace`; distinguishes real contradictions from temporal evolution |
| **V2.3** | Deterministic source trust prior | Tiered (`tier_1_primary` → `tier_5_low`); bounded additive contribution to the relevance score |
| **V2.4** | Claim-level verification | Two-tier: deterministic overlap first, LLM fallback only for the ambiguous mid-band. `[UNVERIFIED]` marker is auditable. |
| **V2.5** | Per-provider circuit breakers | Prevents Tenacity-storm cascades during eval re-runs; explicit fallback chains per provider |
| **V2.6** | Cancellation propagation | `AbortController`-driven; closes server-side disconnect watcher; partial-state persisted |
| **V3.1** | Hybrid RRF retrieval (opt-in via `HYBRID_RETRIEVAL=1`) | BM25 + bge-small-en-v1.5 vector ranking fused via Reciprocal Rank Fusion (k=60) |
| **V3.2** | Adaptive 2-hop retrieval gated by planner confidence | `MAX_HOPS=2` hard cap; second hop restricted to `RECENCY_CHECK` + `CONTRADICTION_PROBE` intents |
| **V3.4** | Multi-Indic eval subset | Hindi + Tamil + Bengali + Marathi questions; cross-language consistency check via entity Jaccard |

## Setup

### Quick start (local)

```bash
git clone https://github.com/evenindividual04/sarvam-assignment.git
cd sarvam-assignment
cp .env.example .env       # fill in PARALLEL_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, GITHUB_TOKEN
pip install -r requirements.txt
python -c "from agent.memory import init_db; import asyncio; asyncio.run(init_db())"
uvicorn main:app --port 7860
# In another shell, for the React UI:
cd frontend && npm install && npm run dev
# Open http://localhost:3000
```

### Docker (single command)

```bash
docker-compose up --build
# Backend on http://localhost:7860
# Frontend dev separately (cd frontend && npm run dev) or build static and host
```

### Required API keys (free tiers)

| Key | Provider | Where to get |
|---|---|---|
| `PARALLEL_API_KEY` | Parallel AI (search) | parallel.ai |
| `GEMINI_API_KEY` | Google Gemini (synth) | aistudio.google.com |
| `GROQ_API_KEY` | Groq (planner + conflict probe) | console.groq.com |
| `GITHUB_TOKEN` | GitHub Models (eval judge) | github.com/settings/tokens |
| `TAVILY_API_KEY` | (optional, fallback search) | tavily.com |
| `SERPER_API_KEY` | (optional, fallback search) | serper.dev |
| `SARVAM_API_KEY` | (optional, alternative synth) | sarvam.ai |
| `OPENROUTER_API_KEY` | (optional, alternative synth) | openrouter.ai |

## Evaluation Methodology

53 questions across 4 languages (English, Hindi, Tamil, Bengali, Marathi) and 6 categories (factual, multi_hop, comparison, insufficient_evidence, conflicting, multi_turn). Run the harness:

```bash
python eval/eval_runner.py                          # single run, BM25 retrieval
python eval/eval_runner.py --ablate                 # ablation: BM25 vs hybrid RRF
HYBRID_RETRIEVAL=1 python eval/eval_runner.py       # hybrid retrieval only
SYNTH_PROVIDER=sarvam python eval/eval_runner.py    # synth via Sarvam Model API
```

After a run, navigate to `/eval` in the React UI for the full dashboard with per-question drill-down (6 tabs: Answer · Context · Doc Map · Judge · Claims · Probe).

### Metrics and why these ones

| Metric | Type | Catches |
|---|---|---|
| **Faithfulness** | LLM (GPT-4o-mini) | Hallucinated claims that look citation-grounded but aren't in the retrieved context |
| **Citation Integrity** | Deterministic | Cited URLs that don't exist in the fetched pool — unambiguous failure, ungameable |
| **Claim Precision** | Deterministic + LLM fallback | Per-sentence verification at generation time; reduces HALLUCINATION_FACT / HALLUCINATION_ATTRIBUTION |
| **Answer Relevance** | LLM | Catches the orthogonal failure to Faithfulness (grounded but didn't answer the question) |
| **Context Precision** | LLM | Did the retrieval layer fetch the necessary information? Isolates retrieval failures from synthesis failures |
| **Conflict Adherence** | LLM | Applied only to conflicting-source queries: did the agent surface disagreement, or pick a side? |
| **Session Coherence** | LLM | Multi-turn continuity: forgets prior context vs maintains it |
| **Factual Accuracy** | Deterministic vs gold-truth | Substring + entity intersection against `gold_answer` for factual questions only |
| **Cross-Language Consistency** | Deterministic (entity Jaccard) | Same `concept_id` answered in EN and an Indic script should cite compatible facts |

### Failure taxonomy

Per-question classification: `HALLUCINATION_FACT` / `HALLUCINATION_ATTRIBUTION` / `KNOWLEDGE_BLEED` / `RETRIEVAL_FAILURE` / `CONFLICT_MISS` / `COHERENCE_FAIL` / `PASS`. Surfaced in the React dashboard as a distribution chart.

## Runtime Failure Budget

The orchestrator enforces per-stage budgets and degrades gracefully instead of crashing a turn.

| Policy key | Default | Behavior on breach |
|---|---:|---|
| `FAILURE_POLICY_PLAN_TIMEOUT_S` | `25` | Fallback to direct-query planning (single PRIMARY query) |
| `FAILURE_POLICY_SEARCH_TIMEOUT_S` | `45` | Continue with empty results; downstream stages adapt |
| `FAILURE_POLICY_FETCH_TIMEOUT_S` | `60` | Continue with partial extracted content |
| `FAILURE_POLICY_SELECT_TIMEOUT_S` | `20` | Fallback to heuristic selector |
| `FAILURE_POLICY_SYNTH_TIMEOUT_S` | `90` | Emit bounded fallback response with follow-up queries |
| `FAILURE_POLICY_MAX_TOTAL_TURN_TIME_S` | `240` | Tag `budget_breach` in run metadata; stop extra work |
| `FAILURE_POLICY_MAX_HOPS` | `2` | Hard cap on adaptive 2-hop retrieval |

Per-provider circuit breakers (V2.5) sit inside Tenacity retry boundaries — one Tenacity-exhausted call counts as one logical failure. Thresholds: search providers 4 failures / 60s, LLM providers 5 failures / 60s, judge 3 failures / 60s.

All stage timings (`planning_ms`, `search_ms`, `fetch_ms`, `select_ms`, `probe_ms`, `synthesize_ms`, `verification_ms`) and fallback/budget tags are persisted per turn in `run_metadata_json` and surfaced in the trace inspector.

## Risks and Limitations

- **Free-tier rate limits** — heavy concurrent eval runs can hit Gemini / Groq rate limits. The V2.5 circuit breaker prevents Tenacity-storm cascades but won't increase the limit itself.
- **Web SEO spam** — `favor_precision=True` in Trafilatura plus the V2.3 source trust prior reduce but don't eliminate low-quality sources. Combine with a domain allowlist for production.
- **Ground truth unknowability** — the system detects conflicts but cannot adjudicate them. The `judge_factual_accuracy` deterministic check covers questions where a gold answer exists; for genuinely contested facts, the system explicitly surfaces disagreement instead of choosing.
- **JavaScript-rendered pages** — single-page apps with client-side content aren't fetched. A Playwright-backed extractor is the fix; not yet implemented.
- **macOS system Python + `sqlite-vec`** — Python compiled without `--enable-loadable-sqlite-extensions` silently disables V3.1 hybrid retrieval (gracefully falls back to BM25). The Linux Docker image avoids this.

## Two Future Improvements (prioritized)

1. **Confidence-calibrated context budget** — currently a fixed 16K token allocation (`15% system / 25% history / 40% web / 20% output`). Should be adaptive based on planner confidence and detected query complexity. The data to drive this already exists (`planner.confidence`, V3.6 calibration correlation).
2. **MCP server interface** — exposing the agent's `/research` endpoint as an MCP tool so other agents (specifically Sarvam's ARYA team's multi-agent backends) can consume it directly. Scoped in plan; not yet shipped.

## Reproducibility

Generate a deterministic runtime report:

```bash
python scripts/repro_report.py --mode quick
```

Prints timestamp, git metadata, Python/platform versions, env knobs (non-secret), prompt versions (e.g. `synth_v5_indic`, `planner_v4`, `conflict_v3`), failure policy config, and smoke-run summary.

## Assumptions

1. "Deep research" = multi-source retrieval with claim verification and conflict detection, not single-source summarisation.
2. The libraries used (`rank_bm25`, `trafilatura`, `aiosqlite`, `fastembed`, `sqlite-vec`) are **not** orchestration frameworks. No LangChain, LangGraph, CrewAI, LlamaIndex, or Haystack.
3. Session IDs are client-generated UUIDs stored in browser localStorage. Sessions persist in SQLite until manually deleted.
4. Citation format `[Title — domain](URL)` applied in final output. Internal `[doc_N]` markers used during generation, converted post-synthesis. Unsupported claims marked with `[UNVERIFIED]` inline.
5. Eval uses LLM-as-judge with a different model family from the generator (GPT-4o-mini judges Gemini output) to prevent same-family bias.
6. Rolling summary triggers when turn count > 5, compressing turns older than the last 3, firing once per 3 new turns thereafter.
7. Iterative re-search is **planner-confidence-gated**, not uncertainty-marker-driven. `MAX_HOPS=2` hard cap; second hop's queries restricted to `RECENCY_CHECK` and `CONTRADICTION_PROBE` intents.

## Project Structure

```
.
├── agent/                          # Core pipeline
│   ├── orchestrator.py             #   state machine: PLANNING → ... → DONE/CANCELLED
│   ├── search.py                   #   multi-Indic language detection, provider chain
│   ├── extractor.py                #   httpx + Trafilatura
│   ├── context_engine.py           #   BM25 + FlashRank + 3-factor + (optional) hybrid RRF
│   ├── synthesizer.py              #   streaming wrapper
│   ├── citation_guard.py           #   [doc_N] → [Title — domain](URL)
│   ├── claim_verifier.py           #   per-sentence verification (V2.4)
│   ├── embedder.py                 #   bge-small-en-v1.5 (V3.1, optional)
│   ├── memory.py                   #   aiosqlite schema, migrations, FTS5
│   ├── eval_queries.py             #   per-language, per-category, drill-down helpers
│   └── models.py                   #   dataclasses + Pydantic
├── utils/                          # Cross-cutting
│   ├── provider_router.py          #   Gemini / Groq / Sarvam / OpenRouter / GitHub Models
│   ├── circuit_breaker.py          #   V2.5
│   ├── cancellation.py             #   V2.6 token registry
│   ├── source_trust.py             #   V2.3 tier table
│   ├── prompt_registry.py          #   versioned prompts (planner_v4, synth_v5_indic, …)
│   ├── failure_policy.py           #   per-stage timeouts + breaker thresholds
│   ├── cost_model.py               #   per-model token cost lookup
│   └── token_counter.py            #   tiktoken + ContextBudget
├── eval/
│   ├── dataset.json                #   53 questions, 4 languages, 6 categories
│   ├── eval_runner.py              #   single-mode + --ablate
│   ├── judge.py                    #   8 metrics: faithfulness, relevance, …, cross-lang
│   └── ablation_report.py          #   BM25 vs hybrid delta report
├── frontend/                       # Next.js 16 App Router, Tailwind, shadcn/ui
│   ├── app/                        #   chat / sessions / eval / drill-down
│   ├── components/                 #   chat / eval / trace / shell
│   └── lib/                        #   SSE hook, API client, types
├── main.py                         # FastAPI: /research SSE, /sessions, /eval/*, /health
├── docs/DEPLOY.md                  # 30-min VPC deployment guide for enterprise IT
├── legacy/                         # Decommissioned Streamlit V1 (preserved for reproducibility)
└── tests/                          # 187 passing
```

## Demo Video

[Add demo video link here when recorded]

## License

MIT. See LICENSE.
