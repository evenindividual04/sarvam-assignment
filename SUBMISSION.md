---
title: "Deep Research Agent — Submission"
author: "Anmol Sen"
subtitle: "Sarvam AI FDSE Intern Assignment"
---

# Deep Research Agent

A web-grounded research agent that issues typed search queries across multiple providers, fetches and reranks sources, and synthesizes citation-traced answers — with every claim audited against the snippet it cites at generation time. Conflicts between sources are surfaced as structured disagreements rather than collapsed into a single take. Built in plain Python `asyncio` with no orchestration framework, in line with the assignment constraint.

---

## Required Links

| Artifact | URL |
|---|---|
| **Live frontend** (Vercel) | <https://sarvam-deep-research-agent.vercel.app> |
| **Live backend** (Hugging Face Spaces) | <https://evenindividual00-sarvam-deep-research.hf.space> |
| **Source code** (GitHub) | <https://github.com/evenindividual04/sarvam-assignment> |
| **Demo video** | <https://www.loom.com/share/6c174a551046421db5b7eabd73394766> |
| **Submission git SHA** | `dc6bd18` |

### Quick local reproduction

```bash
git clone https://github.com/evenindividual04/sarvam-assignment.git
cd sarvam-assignment
cp .env.example .env       # fill in API keys
pip install -r requirements.txt
python -c "from agent.memory import init_db; import asyncio; asyncio.run(init_db())"
uvicorn main:app --port 7860
# in another shell:
cd frontend && npm install && npm run dev
```

A legacy Streamlit UI is preserved under `legacy/` for single-process reproduction:
`streamlit run legacy/streamlit_app.py`.

---

## Necessary Assumptions

These are the choices made where the assignment left judgment to the implementer. Each one shaped a specific code path and is listed so reviewers can challenge or accept them deliberately.

### 1. "Deep research" means multi-source retrieval with claim verification and conflict detection
Not single-source summarisation. Each turn issues 2–4 typed sub-queries across Parallel / Tavily / Serper, fetches 8–14 URLs, reranks with FlashRank, and runs a conflict-detection probe before synthesis. A single-source path would not exercise the conflict-handling, citation-integrity, or multi-hop logic the assignment grades on.

### 2. No orchestration frameworks
The libraries used (`rank_bm25`, `trafilatura`, `aiosqlite`, `fastembed`, `sqlite-vec`, `tenacity`, `httpx`, `flashrank`) are utilities — not orchestration frameworks. No LangChain, LangGraph, CrewAI, LlamaIndex, or Haystack anywhere in the dependency tree. State machine and provider routing are plain async functions with type hints.

### 3. Session ownership is client-generated + ownership-checked
Session IDs are UUIDs generated in browser `localStorage`. Sessions persist in SQLite until manually deleted. `POST /research/cancel/{turn_id}` and `POST /research/approve/{turn_id}` require `session_id` (JSON body or `?session_id=…`) to prove caller ownership — mismatched or missing IDs return 404. This prevents one session from cancelling another's in-flight turn.

### 4. Citation format is `[Title — domain](URL)` per assignment line 62
Internal `[doc_N]` markers are used during generation for stable referencing, then converted post-synthesis by `agent/citation_guard.py`. Unsupported claims are marked `[UNVERIFIED]` inline. The sentence-level quote-anchored verification pipeline (`<quote>...</quote> <claim>...</claim> [doc_N]`) is used internally for the faithfulness audit, then stripped before display.

### 5. Eval judge is Groq Llama 3.3 70B; cross-family check uses GPT-4o-mini
The primary evaluation judge is Groq Llama 3.3 70B (Meta). A cross-family check runs GPT-4o-mini (OpenAI, accessed via GitHub Models) on a 20-question seed=42 subset to validate that results are not inflated by same-family preference. GPT-4o-mini is quota-guarded against the 150/day GitHub Models cap. Seven metrics scored independently to prevent anchor bleed: Faithfulness, Answer Relevance, Context Precision, Citation Integrity, Claim Precision, Conflict Adherence, Session Coherence.

### 6. Rolling summary trigger is turn-count-gated
Fires when `turn_count > 5`, compressing turns older than the last 3 into a Groq-generated summary, then once per 3 new turns thereafter. Below 5 turns, full verbatim history is kept. This is the threshold where compression starts paying for itself in token budget without losing context.

### 7. Iterative re-search is planner-confidence-gated, not uncertainty-marker-driven
`FAILURE_POLICY_MAX_HOPS=2` (env var). Hop 2 fires when the adaptive stopping check doesn't short-circuit at hop 1 — specifically when planner confidence is `low` or `medium`, context didn't saturate the budget (selected < 80% of web context budget), and token budget isn't exhausted. The adaptive stopping check is the smart terminator; deterministic rules (hop cap, AGENT_MAX_ITER=5) are the safety net.

### 8. Cross-family judge sample uses a deterministic 20-question subset
Seed=42 for reproducibility. Configurable via `--cross-family-sample-size`. Quota-guarded against the 150/day GitHub Models cap so a misconfigured run cannot exhaust the day's budget. Full 76-question dataset runs on the primary Groq judge.

### 9. Search provider hierarchy is Parallel → Tavily → Serper
Parallel is the primary (16,000 free queries, AI-native excerpts). Tavily is the fallback (1,000/month, raw_content). Serper is the last resort (2,500, snippet-only, requires separate Trafilatura fetch). Order chosen by free quota × extraction-step economy.

### 10. Frontend / backend split
Next.js 16 frontend (Vercel) talks to FastAPI backend (Hugging Face Spaces) over CORS-allowed SSE. Frontend is deliberately stateless beyond `localStorage` UI preferences — all session/turn/eval state lives in the backend SQLite. This split is what made deploying to two free hosting tiers feasible.

---

## Evaluation Summary

- **Dataset:** 76 questions across factual (21) / multi-hop (13) / comparison (10) / insufficient-evidence (12) / conflicting (10) / multi-turn (10); languages: English 44, Hindi 17, Tamil 5, Bengali 5, Marathi 5
- **Metrics:** 7 per-question scores judged independently to prevent anchor bleed
- **Judge:** Groq Llama 3.3 70B (primary); GPT-4o-mini via GitHub Models (cross-family check on 20-question subset)
- **Methodology details:** `docs/EVAL_METHODOLOGY.md` in the repo

Run the harness yourself:

```bash
python eval/eval_runner.py                          # full 76-question run
python eval/eval_runner.py --ablate                 # BM25-only vs hybrid RRF
RETRIEVAL_MODE=hybrid python eval/eval_runner.py    # fail-loud if sqlite-vec absent
```

Results land in `eval/results/eval_<timestamp>.jsonl` with the full per-question scoring trace.

---

## What's in the Repo

Pointers to the canonical implementation files, so reviewers can jump straight to the load-bearing code:

| Concern | File |
|---|---|
| State machine (planning → search → fetch → select → probe → generate → verify) | `agent/orchestrator.py` |
| Adaptive stopping check + 6-rule deterministic gate | `agent/termination_policy.py` |
| Context pipeline (BM25 → FlashRank → 5-factor selection) | `agent/context_engine.py` |
| Citation conversion + claim-level quote audit | `agent/citation_guard.py` |
| Conflict probe (self / pair / conditional conflict types) | `agent/context_engine.py` (`probe_contradictions`) |
| Provider router (Sarvam → Gemini → OpenRouter → Cerebras fallback chain) | `utils/provider_router.py` |
| Storage + FTS5 | `agent/memory.py` |
| Eval harness | `eval/eval_runner.py`, `eval/judge.py` |
| API surface (FastAPI + SSE) | `main.py` |
| Frontend chat UI | `frontend/app/page.tsx` |

---

## Submission Manifest

This document and the live system together comprise the submission. Nothing else is required to evaluate it — every link above is reachable, every assumption above is testable by reading the file path next to it.

**Git SHA at submission:** `dc6bd18` (branch: `main`).
