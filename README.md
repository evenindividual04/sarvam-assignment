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

A multi-source web research agent built with plain Python asyncio, featuring conflict detection, session persistence, and LLM-as-judge evaluation.

## Target Users and Problem

Enterprise researchers, journalists, analysts, and knowledge workers who need answers that go beyond a single search result. The core problem: standard LLMs answer from training data that is stale, unverifiable, and cannot cite sources. Standard search engines return links but don't synthesize. This agent sits in between — it conducts live web research, selects the most credible evidence, and generates grounded answers where every factual claim is traced back to a real URL retrieved in that session.

Secondary users: developers building AI pipelines who want a reference implementation of a production-grade research agent built without orchestration frameworks.

## Definition of "Deep Research"

Shallow retrieval means fetching the top search result and summarising it. Deep research, as implemented here, means:

1. **Multi-source triangulation** — issuing 2-4 search queries per question targeting different angles, then fetching and independently evaluating content from multiple domains.
2. **Evidence-grounded generation** — the model is never allowed to generate from its training data. Every factual claim must be attributable to a specific retrieved document chunk.
3. **Conflict-aware synthesis** — when retrieved sources disagree, the system does not pick a side. It explicitly surfaces the disagreement, cites both conflicting sources, and expresses epistemic uncertainty to the user.
4. **Iterative refinement** — if initial retrieval is insufficient, the agent plans new queries and re-searches, up to a bounded limit, rather than generating a low-confidence answer.
5. **Session continuity** — prior conversation turns are persisted and the most relevant turns are retrieved into context for follow-up questions, enabling genuine multi-turn research dialogue.

## Data Flow

```text
User Query
    |
    v
[PLANNER] -> 2-4 search queries (Groq Llama 3.3 70B)
    |
    v
[SEARCHER] -> Parallel API (Primary) -> list[SearchResult]
              Tavily / Serper fallback
    |
    v
[FETCHER] -> httpx async (semaphore=3) -> Trafilatura extract
    |
    v
[CONTEXT ENGINE] -> BM25 ranking + recency decay + domain diversity cap
                 -> Conflict detection (Groq pre-synthesis call)
                 -> XML doc blocks with provenance metadata
    |
    v
[SYNTHESIZER] -> Gemini Flash streaming -> citation guard -> format conversion
    |
    v
[PERSISTENCE] -> aiosqlite: session, turns, context, summaries, FTS5 index
```

## WHY THESE 5 METRICS:

Faithfulness: Measures whether claims are grounded in retrieved context, not
training data. This is the core RAG failure mode. Citation volume alone is
misleading — an agent can cite correct URLs while hallucinating claim content.

Citation Integrity: The only deterministic metric. No LLM judge. Score < 1.0
is an unambiguous failure. Prevents gaming: you can't get credit by citing
non-existent sources.

Answer Relevance: Catches the orthogonal failure to Faithfulness. High
faithfulness + low relevance = found real facts that didn't answer the
question. These two form a necessary pair.

Conflict Adherence: Addresses the most dangerous silent failure: an agent that
finds contradictory sources and confidently picks one side. Only applied to
conflicting-source queries, making it a targeted, non-diluted test.

Session Coherence: Evaluates what the other four miss — multi-turn continuity.
A system that answers each turn correctly in isolation but forgets prior
context is not maintaining sessions. This catches that failure.

*Note: Context Precision was added to evaluate if the retrieval layer fetched the necessary information successfully.*

## Evaluation Results

Out of 17 rigorous test questions across factual, multi-hop, comparison, conflicting, and insufficient evidence categories:
- **Pass Rate:** ~85% (Passing standard metrics).
- **Failure Taxonomy:** The most common failure modes mitigated during development were KNOWLEDGE_BLEED (hallucinating facts not in context) and CONFLICT_MISS (failing to present both sides of conflicting data). Both have been significantly improved.

## Risks, Limitations, and 2 Future Improvements

**Risks and Limitations:**
- Rate limits: Free tier provider rate limits may bottleneck intense simultaneous multi-session workloads.
- Low-quality sources: `favor_precision=True` in Trafilatura reduces but doesn't eliminate SEO spam.
- Conflicting sources: system detects and surfaces conflicts but cannot determine ground truth.
- Context length: 6,000-token budget may be insufficient for complex multi-document synthesis.
- Dynamic content: JavaScript-rendered pages (SPAs) are not fetched.

**2 Future Improvements:**
1. Adaptive context budget: allocate dynamically based on query complexity rather than fixed 6K tokens.
2. Source credibility scoring: integrate domain reputation signal (academic/government detection) into context ranking.

## Deployment

The application uses Streamlit and can be easily deployed via **Streamlit Community Cloud** or Docker.

### Local Deployment (Docker)

```bash
docker-compose up --build
```
The app will run at `http://localhost:8501`.

## Runtime Failure Budget Policy

The orchestrator enforces per-stage budgets and degrades gracefully instead of crashing a turn.

| Policy key | Default | Behavior on breach |
|---|---:|---|
| `FAILURE_POLICY_PLAN_TIMEOUT_S` | `25` | Fallback to direct-query planning |
| `FAILURE_POLICY_SEARCH_TIMEOUT_S` | `45` | Continue with empty results and fallback search path |
| `FAILURE_POLICY_FETCH_TIMEOUT_S` | `60` | Continue with partial/empty extracted content |
| `FAILURE_POLICY_SELECT_TIMEOUT_S` | `20` | Fallback to heuristic selector |
| `FAILURE_POLICY_SYNTH_TIMEOUT_S` | `90` | Emit bounded fallback response with follow-up queries |
| `FAILURE_POLICY_MAX_TOTAL_TURN_TIME_S` | `240` | Tag `budget_breach` in run metadata and stop extra work |
| `FAILURE_POLICY_MAX_RETRIES_PER_PROVIDER` | `3` | Bound outbound retries per provider |

All stage timings (`planning_ms`, `search_ms`, `fetch_ms`, `select_ms`, `synthesize_ms`) and fallback/budget tags are persisted per turn in `run_metadata_json`.

## Reproducibility

Generate a deterministic runtime report:

```bash
python scripts/repro_report.py --mode quick
```

This prints timestamp, git metadata (or `unknown`), Python/platform versions, env knobs (non-secret), prompt versions, failure policy config, and smoke-run summary.

## Assumptions Made

1. **"Deep research"** = multi-source retrieval with conflict detection, not single-source summarisation.
2. Libraries used (`rank_bm25`, `trafilatura`, `aiosqlite`) are not frameworks. No LangChain, LangGraph, CrewAI, LlamaIndex, or Haystack used.
3. **Gemini 2.5 Flash** (google-genai, free tier) = synthesis LLM. **Groq Llama 3.3 70B** (free tier) = planning + conflict detection. **GitHub Models GPT-4o-mini** = eval judge (different model family from generator — required for eval integrity).
4. Session IDs are user-generated UUIDs. Sessions persist in SQLite until manually deleted.
5. Citation format `[Title — domain](URL)` applied in final output. Internal `[doc_N]` markers used during generation and converted post-synthesis before streaming to UI.
6. Eval uses LLM-as-judge (GitHub Models GPT-4o-mini). Faithfulness and citation integrity are primary metrics.
7. Rolling summary triggers when turn count > 5, compressing turns older than the last 3, firing once per 3 new turns thereafter.
8. Iterative re-search triggers up to `MAX_HOPS=5` when the synthesis output contains specific uncertainty markers, allowing the agent to self-correct retrieval failures without endless looping.

## Video Demo

[Add Video Demo Link Here]
