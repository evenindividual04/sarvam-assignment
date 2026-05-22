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
`streamlit run app.py`.

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
Internal `[doc_N]` markers are used during generation for stable referencing, then converted post-synthesis by `agent/citation_guard.py`. Unsupported claims are marked `[UNVERIFIED]` inline (assignment line 91: "state uncertainty AND propose next steps"). The `<quote>...</quote> <claim>...</claim> [doc_N]` ReClaim triplet structure is used internally for the faithfulness audit pipeline, then stripped before display.

### 5. Judge model family discipline
The evaluation harness uses GPT-4o-mini (OpenAI family, accessed via GitHub Models) to judge Gemini-generated answers. Different model family from the generator to prevent same-family self-preference bias (Panickssery et al. 2024, Zheng et al. 2023). Each metric is a separate judge call (no anchor bleed): Faithfulness, Answer Relevance, Context Precision, Citation Integrity, Claim Precision, Conflict Adherence, Session Coherence.

### 6. Rolling summary trigger is turn-count-gated
Fires when `turn_count > 5`, compressing turns older than the last 3 into a Groq-generated summary, then once per 3 new turns thereafter. Below 5 turns, full verbatim history is kept. This is the threshold where compression starts paying for itself in token budget without losing context.

### 7. Iterative re-search is planner-confidence-gated, not uncertainty-marker-driven
`MAX_HOPS=2`. Hop 2 fires when the deterministic termination policy doesn't short-circuit at hop 1 — specifically when planner.confidence is `low` or `medium`, context didn't saturate the budget (selected < 80% of web context budget), STOP-RAG LLM judge doesn't terminate, and token budget isn't exhausted. The Stop-RAG value gate (Park et al. 2025) is the smart terminator; deterministic rules are the safety net.

### 8. Cross-family judge sample uses a deterministic 20-question subset
Seed=42 for reproducibility. Configurable via `--cross-family-sample-size`. Quota-guarded against the 150/day GitHub Models cap so a misconfigured run cannot exhaust the day's budget. Full 17-question dataset still runs on the primary judge.

### 9. Search provider hierarchy is Parallel → Tavily → Serper
Parallel is the primary (16,000 free queries, AI-native excerpts). Tavily is the fallback (1,000/month, raw_content). Serper is the last resort (2,500, snippet-only, requires separate Trafilatura fetch). Order chosen by free quota × extraction-step economy, not by benchmark score alone.

### 10. Frontend / backend split
Next.js 16 frontend (Vercel) talks to FastAPI backend (Hugging Face Spaces) over CORS-allowed SSE. Frontend is deliberately stateless beyond `localStorage` UI preferences — all session/turn/eval state lives in the backend SQLite. This split is what made deploying to two free hosting tiers feasible.

---

## Evaluation Summary

- **Dataset:** 17 questions across factual / multi-hop / conflict / cross-script / Hindi categories
- **Metrics:** 7 per-question scores (5 generation, 1 retrieval, 1 cross-turn) judged independently to prevent anchor bleed
- **Judge:** GPT-4o-mini (different family from generator, per assumption #5)
- **Methodology details:** `docs/EVAL_METHODOLOGY.md` in the repo

Run the harness yourself:

```bash
python eval/eval_runner.py                          # full 17-question run
python eval/eval_runner.py --ablate                 # BM25-only vs hybrid RRF
RETRIEVAL_MODE=hybrid python eval/eval_runner.py    # fail-loud if sqlite-vec absent
```

Results land in `eval/results/eval_<timestamp>.jsonl` with the full per-question scoring trace.

---

## What's in the Repo

Pointers to the canonical implementation files, so reviewers can jump straight to the load-bearing code:

| Concern | File |
|---|---|
| State machine (planning → search → fetch → select → synthesize) | `agent/orchestrator.py` |
| Termination policy (Stop-RAG + 6-rule deterministic gate) | `agent/termination_policy.py` |
| Context pipeline (BM25 → FlashRank → 3-factor selection) | `agent/context_engine.py` |
| Citation conversion + ReClaim audit | `agent/citation_guard.py` |
| Conflict probe (DRAGged taxonomy: self / pair / conditional) | `agent/context_engine.py` (`probe_contradictions`) |
| Provider router (Gemini → DeepSeek → Cerebras fallback) | `utils/provider_router.py` |
| Storage + FTS5 | `agent/memory.py` |
| Eval harness | `eval/eval_runner.py`, `eval/judge.py` |
| API surface (FastAPI + SSE) | `main.py` |
| Frontend chat UI | `frontend/app/page.tsx` |

---

## Submission Manifest

This document and the live system together comprise the submission. Nothing else is required to evaluate it — every link above is reachable, every assumption above is testable by reading the file path next to it.

**Git SHA at submission:** `dc6bd18` (branch: `main`).
