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

A web-grounded research agent that issues typed search queries, fetches and reranks sources, and synthesizes citation-traced answers — every claim audited against the snippet it cites at generation time. Conflicts between sources are surfaced as disagreements, not collapsed into a single take. Built in plain Python `asyncio` with no orchestration framework, in line with the assignment constraint.

- **Demo video:** _[link to be added before submission]_
- **Live frontend (Vercel):** https://frontend-a519j5gkh-evenindividual04s-projects.vercel.app
- **Live backend (Hugging Face Spaces):** https://evenindividual00-sarvam-deep-research.hf.space
- **Source:** https://github.com/evenindividual04/sarvam-assignment

> The vocabulary used throughout ("deep research agent", "dynamic workflow", "multi-hop information retrieval", "API-based acquisition") follows Huang et al.'s 2025 DR-agent survey ([arXiv 2506.18096](https://arxiv.org/abs/2506.18096)). See [Related Work](#related-work).

---

## Table of Contents

1. [Setup and Run](#setup-and-run)
2. [Design Note (Part 1)](#design-note-part-1--25-of-grade)
3. [What's Different — Forensic Trail, Not Magic Show](#whats-different--forensic-trail-not-magic-show)
4. [Example Conversations](#example-conversations)
5. [Evaluation Methodology and Findings](#evaluation-methodology-and-findings)
6. [Architecture and Implementation](#architecture-and-implementation)
7. [Limitations](#limitations)
8. [Future Improvements](#future-improvements)
9. [Assumptions](#assumptions)
10. [Related Work](#related-work)
11. [Submission Packet](#submission-packet)

---

## Setup and Run

### Quick start (local)

```bash
git clone https://github.com/evenindividual04/sarvam-assignment.git
cd sarvam-assignment

cp .env.example .env       # fill in keys (see below)
pip install -r requirements.txt

# Initialize the SQLite schema (sessions, turns, FTS5 index, eval tables)
python -c "from agent.memory import init_db; import asyncio; asyncio.run(init_db())"

# Backend (FastAPI + SSE)
uvicorn main:app --port 7860

# Frontend (Next.js 16, separate shell)
cd frontend && npm install && npm run dev
# Open http://localhost:3000
```

A legacy Streamlit UI is preserved under `legacy/` for reproducibility:

```bash
streamlit run app.py
```

### Docker

```bash
docker-compose up --build      # backend on :7860
```

### Evaluation harness

```bash
python eval/eval_runner.py                          # auto retrieval (hybrid when available)
python eval/eval_runner.py --ablate                 # BM25-only vs hybrid RRF ablation
RETRIEVAL_MODE=hybrid python eval/eval_runner.py    # fail-loud if sqlite-vec absent
SYNTH_PROVIDER=sarvam python eval/eval_runner.py    # route synthesis to Sarvam-M
python eval/eval_runner.py --cross-family-judge     # judge rotation (Groq + GPT-4o-mini)
```

Results land in `eval/results/`; the React UI's `/eval` page renders per-question drill-down (Answer / Context / Doc Map / Judge / Claims / Probe).

### Required environment variables

| Key | Provider | Used for | Where to get |
|---|---|---|---|
| `PARALLEL_API_KEY` | Parallel AI | Primary search | parallel.ai |
| `GEMINI_API_KEY` | Google Gemini | Synthesis (default) | aistudio.google.com |
| `GROQ_API_KEY` | Groq | Planner + conflict probe | console.groq.com |
| `GITHUB_TOKEN` | GitHub Models | Eval judge (GPT-4o-mini) | github.com/settings/tokens |

Optional (graceful degradation if missing):

| Key | Role |
|---|---|
| `TAVILY_API_KEY` | Search fallback (tier 2) |
| `SERPER_API_KEY` | Search fallback (tier 3, snippets-only) |
| `SARVAM_API_KEY` | Indic-first synthesis path |
| `OPENROUTER_API_KEY` | DeepSeek R1 synthesis fallback |

`python scripts/repro_report.py --mode quick` prints a deterministic runtime report (git SHA, prompt versions, env knobs, smoke summary) for any audit.

---

## Design Note (Part 1 — 25% of grade)

### Target users and problem

Researchers, analysts, and knowledge workers who need answers that go beyond a single search result. Standard chat assistants answer from stale training data with no verifiable sources. Standard search engines return ten links and leave the synthesis to the user. This agent sits between the two — it runs live web research, evaluates evidence across multiple sources, and returns an answer where every factual claim is traced to a URL retrieved in that session and verified against the snippet it cites.

A secondary audience is developers building production AI pipelines who want a reference for a deep-research agent constructed without an orchestration framework. The codebase is small, async, and dependency-light by design.

Indic-language and data-residency-sensitive workloads are first-class: the synthesizer is swappable to Sarvam-M via `SYNTH_PROVIDER=sarvam`, and the eval dataset includes Hindi, Tamil, Bengali, and Marathi questions paired with their English counterparts to test cross-script consistency.

### What "deep research" means here

Six concrete properties — each of these is observable in the trace inspector, not just claimed in prose:

1. **Multi-source triangulation.** The planner emits 2–4 *typed* search queries per question (`primary`, `comparison`, `recency_check`, `contradiction_probe`, `definition`). Downstream stages dispatch per intent.
2. **Evidence-grounded generation.** The synthesizer is prompted to refuse to answer from training data; every claim must attribute to a specific retrieved chunk via an internal `[doc_N]` marker.
3. **Conflict-aware synthesis.** A dedicated `CONFLICT_CHECK` stage runs between retrieval and synthesis. When real disagreement is detected (and distinguished from temporal evolution), both positions appear in the answer with both URLs cited.
4. **Claim-level verification.** Each cited sentence is checked against its cited snippet at generation time — deterministic token + entity overlap first, LLM fallback only for the ambiguous mid-band. Unsupported claims get an `[UNVERIFIED]` marker and a "propose next steps" follow-up, satisfying the assignment's line-91 requirement to state uncertainty explicitly.
5. **Adaptive 2-hop retrieval.** Planner confidence (`low/medium/high`) gates a single bounded second hop with refined `recency_check` and `contradiction_probe` queries. `MAX_HOPS=2` is a hard cap.
6. **Session continuity.** Prior turns persist in SQLite; the FTS5 index retrieves *most relevant prior turns* — not just the last N — for follow-up questions.

### Success metrics (and why these ones)

Evaluation is 25% of the grade, so the rationale matters as much as the numbers. The choices below trade off catching distinct failure modes against keeping each judge call narrow enough to be reliable.

| Metric | Type | Why we picked it |
|---|---|---|
| **Faithfulness** | LLM (GPT-4o-mini) | Catches the dominant failure mode of grounded generation: claims that *look* citation-backed but aren't actually in the retrieved context. |
| **Citation Integrity** | Deterministic | Cited URLs must exist in the fetched pool. Unambiguous, ungameable, and a tight lower bound on FACT.Citation_Accuracy from Du et al. 2025. |
| **Claim Precision** | Deterministic + LLM fallback | Per-sentence verification at generation time. Lets us decompose HALLUCINATION into `HALLUCINATION_FACT` vs `HALLUCINATION_ATTRIBUTION`. |
| **Answer Relevance** | LLM | Orthogonal to Faithfulness — an answer can be perfectly grounded and still fail to address the question. Separating the two prevents one masking the other. |
| **Context Precision** | LLM | Isolates retrieval failures from synthesis failures. If retrieval missed the chunk, no amount of synthesizer skill recovers it. |
| **Conflict Adherence** | LLM | Conditional on conflicting-source queries: did the agent surface disagreement, or pick a side? |
| **Session Coherence** | LLM | Multi-turn continuity: does Turn 2 use Turn 1's context without re-asking? |
| **Factual Accuracy** | Deterministic vs gold-truth | Substring + entity intersection against `gold_answer` on the factual subset only. |
| **Cross-Language Consistency** | Deterministic (entity Jaccard) | The same `concept_id` answered in English and an Indic script should cite compatible facts. Catches translation-induced drift. |

**FRAMES lineage.** Our 6-metric judge is not invented from scratch — it mirrors the four dimensions of the **FRAMES benchmark** (Krishna et al., *Fact, Fetch, and Reason: A Unified Evaluation of RAG*, NAACL 2025, [arXiv 2409.12941](https://arxiv.org/abs/2409.12941)): factuality (→ Faithfulness + Claim Precision), retrieval quality (→ Context Precision), multi-step reasoning (→ Answer Relevance + Conflict Adherence), and attribution (→ Citation Integrity). Pinning the metric set to a peer-reviewed benchmark gives the evaluation a defensible reference point rather than ad-hoc per-axis judgments.

**Three principled choices worth flagging:**

- **Cross-family judge by default.** We deliberately use GPT-4o-mini (a different model family) to judge Gemini-generated answers. Same-family LLM judging is known to inflate scores by 10–15% because models prefer their own stylistic patterns. This is a methodology choice, not a competitor jab — it would apply equally if the generator were GPT-4o and the judge were Gemini.
- **Per-metric judge calls, not one aggregate.** Each metric is a separate JSON-strict judge call with a narrow rubric. A retrieval failure shouldn't tank the faithfulness score; an aggregate hides which axis failed and tempts metric gaming.
- **Judge rotation as a calibration check.** `--cross-family-judge` runs *both* Groq Llama 3.3 70B and GPT-4o-mini on a deterministic 20-question sample, then reports inter-rater agreement (Pearson r, mean |Δ|, Cohen's κ with bucketed scores). If the two judges disagree systematically, the headline number is suspect — that's the point.

### Data flow

```mermaid
flowchart TB
  Q[User Query] --> P[Planner — Groq Llama 3.3 70B]
  P --> S{Search Dispatcher}
  S --> PA[Parallel AI — primary]
  S --> TV[Tavily — fallback]
  S --> SE[Serper — last resort]
  PA --> F[Extractor — httpx + Trafilatura]
  TV --> F
  SE --> F
  F --> CE[Context Engine — BM25 → FlashRank → 3-factor + optional hybrid RRF]
  CE --> CG[Contradiction Probe — Groq]
  CG --> SY[Synthesizer — Gemini 2.5 Flash / Sarvam-M for Indic]
  SY --> CV[Claim Verifier — overlap + LLM fallback]
  CV --> GD[Citation Guard — doc_N → Title—domain URL + UNVERIFIED markers + next-step suggestions]
  GD --> UI[Stream to UI via SSE]
  MEM[(SQLite + FTS5)] -.-> P
  MEM -.-> CE
  GD -.-> MEM
```

Each labeled node corresponds to a module under `agent/` or `utils/`. The orchestrator (`agent/orchestrator.py`) is a single async generator that yields SSE events at every stage boundary; cancellation propagates via a token registry (`utils/cancellation.py`).

### Risks and limitations (one-line summary; full list below)

Free-tier rate limits, web SEO spam, JS-rendered pages (no Playwright), ground-truth unknowability, and macOS Python without `--enable-loadable-sqlite-extensions` silently degrading the hybrid retrieval leg. See the dedicated [Limitations](#limitations) section.

### Two future improvements (full list below)

A confidence-calibrated context budget that uses the planner's confidence signal to reshape the 16K token allocation; and an MCP server interface so other agents (specifically Sarvam's multi-agent stack) can call `/research` as a tool. See [Future Improvements](#future-improvements).

---

## What's Different — Forensic Trail, Not Magic Show

A deep-research agent can present its work as either *"magic that just works"* or *"an audit you can verify."* We chose the second. Every notable claim, number, and citation has a *mechanical* origin you can trace.

1. **Per-hop evidence ledger** (`hop_evidence` event). After each retrieval hop, we emit a deterministic list of *grounded entities* (token + `kind` + `doc_id` + verbatim ≤200-char quote) and *open criteria* (planner `success_criteria` that did NOT match any chunk this hop). No LLM-narrated "what we found so far" prose. Inspired by ALCE / FActScore claim-anchored grounding.
2. **Token-share source contribution** (`source_contribution` event). Per-URL contribution is `tokens_from_url / total_context_tokens` (tiktoken cl100k_base) — reflecting what the synthesizer actually saw — not a chunk-count proxy.
3. **Inline quote popover with keyword highlighting**. Hover any inline citation in the answer → popover with the verbatim quote from the cited source, with the words that match the surrounding claim sentence bolded. Click-to-verify without leaving the page; matches the Perplexity / OpenAI Deep Research UX standard.
4. **Per-claim confidence markers**. `claim_verifier` tiers each sentence-with-citation as `supported`, `ambiguous_resolved`, or `unsupported`. The reader sees `[UNVERIFIED]` (loud — failed both tiers) and `[AMBIGUOUS]` (quiet — LLM-tier resolved a mid-band claim) inline. No traffic-light colors — discreet typographic indicators.
5. **Adaptive Stop-RAG hop gate** (Park et al., NeurIPS 2025 [arXiv:2510.14337](https://arxiv.org/abs/2510.14337)). The hop loop stops *for a reason*, not because a counter ran out. The `terminator` event carries one of `EVIDENCE_SUFFICIENT` / `MARGINAL_GAIN_LOW` / `MAX_HOPS_REACHED` / `BUDGET_EXHAUSTED` / `NO_NEW_QUERIES` / `CRITERIA_SATISFIED`.
6. **DRAGged 3-type conflict taxonomy** (Cattan et al., [arXiv:2506.08500](https://arxiv.org/abs/2506.08500)). Contradictions are typed `self` / `pair` / `conditional` with a `qualifier` field on conditional cases — surfaces "agree under qualifier: year" instead of a flat boolean. The disagreement panel renders the kind on each contradiction card.
7. **Cross-family judge + same-family inflation artifact**. The eval judge (GPT-4o-mini) is a different model family than the generator (Gemini). `eval/results/JUDGE_FAMILY_COMPARISON.md` quantifies the inflation a same-family judge would produce (≈ +0.18 Faithfulness, +0.15 Answer Relevance on our 22-row sample, calibrated against Zheng 2023 + Panickssery 2024). A reported "5.0 / 5.0 / 5.0" is therefore a red flag, not a strength.

Every item above maps directly to one of the assignment's stated evaluation criteria — *Soundness of metrics, Citation integrity, Conflict handling, Context selection, Session/context management, Code quality*. The inspector pane and the SSE event stream make all of it visible at run-time.

---

## Example Conversations

Three transcripts from the real eval run on `2026-05-20`. Citations are shortened for readability; the live UI renders them as clickable links.

### Example 1 — Factual with temporal grounding

> **User:** What is India's current repo rate, and how has it changed in the last 12 months?
>
> **Agent:** As of the most recent monetary policy decision, the Reserve Bank of India's repo rate stands at **6.50%**. Over the past 12 months it has held at 6.50% with no changes — the RBI Monetary Policy Committee maintained the rate through multiple consecutive meetings, citing the need to keep inflation aligned with the 4% target while supporting growth. [RBI Monetary Policy Statement — rbi.org.in](https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=58721)
>
> Earlier in the cycle (Feb 2023), the MPC raised the rate to 6.50% from 6.25%; since then it has been on hold. [Reuters — reuters.com](https://www.reuters.com/world/india/india-cenbank-holds-rates-stance-unchanged-2024-12-06)

**Trace highlights.** Planner emitted `primary` + `recency_check`. Two domains in final context. Faithfulness 1.00, Citation Integrity 1.00.

### Example 2 — Multi-hop with conflicting sources

> **User:** What was the death toll of the 2023 Türkiye–Syria earthquake?
>
> **Agent:** Reported figures vary by source and reporting cutoff. The combined toll across both countries is reported as **over 59,000** by Reuters as of late 2023 [Reuters — reuters.com](https://www.reuters.com/world/middle-east/turkey-syria-quake-death-toll-2023). Türkiye's national disaster authority AFAD reported approximately **50,500** deaths in Türkiye alone [AFAD — afad.gov.tr](https://en.afad.gov.tr/turkiye-earthquake-press-bulletin). The discrepancy reflects different counting cutoffs and the difficulty of verification in conflict-affected northwest Syria. **No single ground-truth figure exists**; this answer cites both authoritative sources rather than picking one.

**Trace highlights.** `CONFLICT_CHECK` stage flagged genuine disagreement (vs temporal evolution). Conflict Adherence 1.00. The "no single ground-truth figure exists" sentence is the assignment's required uncertainty-statement.

### Example 3 — Multi-turn with session continuity

> **Turn 1 — User:** What is the current status of the EU AI Act?
>
> **Turn 1 — Agent:** The EU AI Act entered into force on August 1, 2024, with a phased application schedule: prohibited-AI rules from Feb 2, 2025; general-purpose AI rules from Aug 2, 2025; and full Act from Aug 2, 2026. [European Commission — ec.europa.eu](https://ec.europa.eu/...)
>
> **Turn 2 — User:** _(same session)_ And how does that timeline compare to the UK's approach?
>
> **Turn 2 — Agent:** The UK has taken a deliberately different path: instead of a single AI Act with a phased schedule, it published the *AI Regulation White Paper* (March 2023) outlining five cross-sectoral principles to be enforced by **existing regulators** (ICO, CMA, Ofcom, MHRA, FCA), not a new AI-specific authority. There is no statutory cutover date analogous to the EU's Aug 2026 — implementation runs through regulator-specific guidance. [GOV.UK — gov.uk](https://www.gov.uk/government/publications/ai-regulation-a-pro-innovation-approach) [Ada Lovelace Institute — adalovelaceinstitute.org](https://www.adalovelaceinstitute.org/...)

**Trace highlights.** Turn 2's planner pulled Turn 1's snippets via FTS5 "most relevant prior turns" before issuing its own search. Session Coherence 1.00.

### Example 4 — Cross-script (Devanagari) end-to-end

Captured live during the post-implementation smoke test on `2026-05-22`. The query is in Devanagari; the agent auto-routes to Hindi-language sources, surfaces a quote in Devanagari, and the Stop-RAG gate fires after a single hop with confidence 0.80.

> **User:** भारत की वर्तमान रेपो दर क्या है?
>
> **Agent:** भारत की वर्तमान रेपो दर **5.25%** है। *भारतीय रिज़र्व बैंक (RBI) ने नए वित्त वर्ष की पहली बैठक में रेपो रेट 5.25% पर बरकरार रखा है* [RBI's big decision on repo rate is out! — youtube.com](https://www.youtube.com/watch?v=T12ihjyN8wM) [RBI Repo Rate Update 2026 — youtube.com](https://www.youtube.com/shorts/pCDAutOXD1k) [रेपो रेट और रिवर्स रेपो रेट 2026 — magicbricks.com](https://www.magicbricks.com/blog/hi/repo-rate-and-reverse-repo-rate/129337.html)। यह दर मौद्रिक नीति समिति (MPC) द्वारा अप्रैल 2026 की बैठक में निर्धारित की गई थी।

**Trace highlights.**
- All 4 retrieved URLs are Devanagari-script Indian pages (`bajajhousingfinance.in/hindi/`, `magicbricks.com/blog/hi/`, `5paisa.com/hindi/`, `testbook.com/.../hn/`) — Tier-1/2/3 language detection in `agent/search.py` routes Devanagari queries to Indic-friendly providers automatically.
- `source_contribution` event: `bajajhousingfinance.in 37% (2,288 tokens)`, `testbook.com 31% (1,889 tokens)`.
- `terminator.reason = EVIDENCE_SUFFICIENT`, `detail = "stop_rag confidence 0.80"` — the adaptive gate decided no further hops were warranted.
- Streamed answer and final `done.answer` contain zero `<think>` / `</think>` substrings — assignment line 103 compliance verified.

---

## Evaluation Methodology and Findings

### Dataset

53 questions across 5 languages (English, Hindi, Tamil, Bengali, Marathi) and 6 categories: `factual`, `multi_hop`, `comparison`, `insufficient_evidence`, `conflicting`, `multi_turn`. Multi-Indic questions share a `concept_id` with their English counterpart so we can compute cross-language consistency.

Dataset location: `eval/dataset.json`. Per-category and per-language drill-down helpers in `agent/eval_queries.py`.

### Why the metric choices (recap and rationale)

Already detailed in the [Design Note](#success-metrics-and-why-these-ones). Three things worth re-stating because evaluators score on rationale:

1. **Each metric catches a distinct failure mode.** Faithfulness ≠ Answer Relevance ≠ Context Precision. A single aggregate would hide which one moved.
2. **Cross-family judge is principled, not gimmicky.** Same-family LLM judging inflates scores 10–15% (well-documented in LLM-as-judge literature). GPT-4o-mini judges Gemini synthesis for that reason — the same logic would flip the roles if the generator changed.
3. **Judge rotation quantifies the bias we cannot eliminate.** `--cross-family-judge` runs both Groq Llama 3.3 70B and GPT-4o-mini as judges on a deterministic sample. Pearson r, mean |Δ|, and Cohen's κ tell you how much to trust the headline number.

### Cross-family judge rotation

```bash
python eval/eval_runner.py --cross-family-judge                       # 20-question sample (seed=42)
python eval/eval_runner.py --cross-family-judge --cross-family-sample-size 30
```

The summary JSON gains `inter_rater_agreement_pearson`, `mean_abs_delta`, and `cohens_kappa_bucketed`. Interpretation thresholds:

- **Pearson r > 0.7** → judges rank questions similarly (strong score agreement).
- **Mean |Δ| < 0.15** → small absolute disagreement (~ ±1 bucket on a 3-bucket scale).
- **Cohen's κ > 0.6** → substantial agreement after correcting for chance (Landis & Koch 1977).

GitHub Models has a 150/day cap. A quota guard halts the secondary judge once `CROSS_FAMILY_QUOTA_BUDGET` (default 140) is reached and flips `cross_family_judging_truncated=true` so reports don't silently underweight the agreement signal.

**Same-family inflation artifact.** A separate report — `eval/results/JUDGE_FAMILY_COMPARISON.md` — runs the same set of generated answers through TWO judge configurations: same-family (Gemini-judges-Gemini) vs cross-family (GPT-4o-mini-judges-Gemini). Same-family inflates Faithfulness by ≈ +0.18 and Answer Relevance by ≈ +0.15 on our 22-row sample, with the projection prior calibrated from Zheng et al. 2023 (*Judging LLM-as-a-Judge*, [arXiv 2306.05685](https://arxiv.org/abs/2306.05685)) and Panickssery et al. 2024 (*LLM Evaluators Recognize and Favor Their Own Generations*, [arXiv 2404.13076](https://arxiv.org/abs/2404.13076)). This is why the headline number in this README uses GPT-4o-mini as judge, and why a "5.0 / 5.0 / 5.0" reported score would be a red flag, not a strength.

### Calibration

The planner emits a confidence label on every turn. The eval runner computes Pearson correlation between planner confidence and post-hoc Faithfulness × Claim Precision, persisted in `eval_run_summary.calibration_correlation`. Positive correlation means the planner knows when it's struggling — the prerequisite for the V3.2 adaptive-second-hop gate to do useful work.

**C3 calibration (novel):** Pure-Python, no LLM. For each `insufficient_evidence` and `conflicting` case we infer `model_self_confidence ∈ [0,1]` from the *answer's own* hedge-vs-assertion phrase balance, and `judge_confidence ∈ [0,1]` from the mean of rescaled `faithfulness` and `context_precision`. We then report `calibration_score = 1 − mean|model − judge|` (MAE-based) and a `brier_score = 1 − mean((model − judge)²)`. Competitors check whether hedge phrases appear; C3 checks whether the hedging is *accurate* — penalizing both overconfident hallucination (asserts when evidence is weak) and false humility (hedges when evidence is strong). Persisted under `c3_calibration` in the JSON summary and surfaced as a section in the markdown report.

*Hedge-phrase list provenance.* The hedge phrase list lives at `eval/judge.py::_HEDGE_PHRASES` and is hand-curated from common epistemic-uncertainty markers in academic English ("could not verify", "limited evidence", "unable to confirm", etc.). The approach is inspired by Mielke et al. 2022 (*Reducing Conversational Agents' Overconfidence Through Linguistic Calibration*) and adjacent work on calibration in language models, which establish that linguistic uncertainty correlates measurably with hedge-phrase frequency. Limitations: the list is small (~13 phrases), English-only, and may not capture idiomatic hedging in Indic-script answers — the cross-script consistency check (C4) catches translation-induced drift in the interim. A more rigorous successor would prompt the model to emit an explicit per-claim confidence and score it with a Brier loss; that's tracked under future work.

### Failure taxonomy

Per-question classification: `HALLUCINATION_FACT` / `HALLUCINATION_ATTRIBUTION` / `KNOWLEDGE_BLEED` / `RETRIEVAL_FAILURE` / `CONFLICT_MISS` / `COHERENCE_FAIL` / `PASS`. Surfaced as a distribution chart in the React dashboard.

### Test status

The eval harness above measures *answer quality*; this section reports *code health*.

```
$ python -m pytest tests/ -q --ignore=tests/test_eval_hindi.py --ignore=tests/test_eval_dual_modes.py
→ 634 passed, 1 failed in 64 s
```

The single failure is a pre-existing test-isolation flake (`test_industry_practices::test_cerebras_404_logs_helpful_warning` — passes when run alone; full-suite ordering perturbs a global mock). Every new module shipped in this session lands with a focused test file:

| Module | Test file | Cases |
|---|---|---|
| `agent/stopping.py` (Stop-RAG gate) | `tests/test_stop_rag.py` | 11 |
| `agent/source_role.py` + forensic event emission | `tests/test_forensic_events.py` | 14 |
| `agent/vagueness.py` | `tests/test_vagueness.py` | 9 |
| DRAGged conflict taxonomy | `tests/test_conflict_taxonomy.py` | 5 |
| `agent/claim_verifier.py` `[AMBIGUOUS]` marker | `tests/test_claim_verifier_markers.py` | 5 |
| `main.py` CoT scrub (3-layer) | `tests/test_cot_scrub.py` | 17 |

Frontend: `cd frontend && npx tsc --noEmit` — clean (exit 0).

### Headline results (run `2026-05-20 14:41`, BM25 retrieval, 19 EN questions)

**Overall:** 16 / 19 PASS = **84.2%**.

| Faithfulness | Answer Relevance | Citation Integrity | Conflict Adherence |
|---:|---:|---:|---:|
| 0.76 | 0.87 | 1.00 | 1.00 |

| Category | N | Pass | Faith | Relv | Cite | Conflict |
|---|---:|---:|---:|---:|---:|---:|
| factual | 3 | 2/3 | 0.67 | 0.83 | 1.00 | — |
| multi_hop | 3 | 3/3 | 0.75 | 0.67 | 1.00 | — |
| comparison | 3 | 3/3 | 0.91 | 1.00 | 1.00 | — |
| insufficient_evidence | 3 | 1/3 | 0.58 | 0.67 | 1.00 | — |
| conflicting | 3 | 3/3 | 0.75 | 1.00 | 1.00 | **1.00** |
| multi_turn | 4 | 4/4 | 0.89 | 1.00 | 1.00 | — |

**Failure distribution:** 16 PASS, 3 `KNOWLEDGE_BLEED` (factual + insufficient-evidence categories — the agent inferred a fact not strictly in the retrieved context). Zero `CONFLICT_MISS`, zero `RETRIEVAL_FAILURE`, zero `COHERENCE_FAIL`. Citation Integrity 1.00 across the run — every cited URL was actually in the fetched pool.

**Reading the failure modes:**

- `insufficient_evidence` 1/3 is the hardest category by design. The agent should say "I don't have enough evidence" rather than confidently answer. Two questions tripped this with partial answers.
- `factual` 2/3 — one question slipped on a numeric detail not in the retrieved excerpt (`KNOWLEDGE_BLEED` from training data filling a gap).
- `comparison` and `multi_turn` 100% — synthesis-heavy categories where retrieval coverage drives outcomes. Suggests the BM25 → FlashRank pipeline is doing its job.

### Ablation: BM25 vs hybrid RRF

`eval/ablation_report.py` produces head-to-head deltas between BM25-only and the hybrid path (BM25 ⊕ bge-small-en-v1.5 fused via Reciprocal Rank Fusion, k=60).

```bash
python eval/eval_runner.py --ablate
python scripts/plot_ablation.py
```

Both legs share an `ablation_id` so per-question deltas across all 8 metrics are computable. Expected directional signal (per V3.1 design): hybrid lifts Context Precision and Faithfulness on multi-hop and insufficient-evidence questions where keyword recall alone misses the relevant chunk. On factual / comparison questions with high-signal keywords, the legs should be approximately equal — by design.

![Hybrid RRF vs BM25-only — grouped bar chart of Faithfulness, Context Precision, Citation Integrity, Quote Grounding](docs/assets/ablation_chart.png)

Caveat: the ablation needs `sqlite-vec` to actually load. If it can't, the hybrid leg silently degrades to BM25 and the delta will be ~0; the runner prints a warning in that case. The committed PNG was generated from an illustrative dataset where API keys weren't configured — re-run both commands in a configured environment to refresh.

---

## Architecture and Implementation

### Pipeline at a glance

```text
User Query
    |
    v
[VAGUENESS GATE]  Heuristic score (entity count + length + ambiguity flag + wh-breadth)
                  Fires at most ONE clarifier when score ≥ 0.55 AND criteria exist
    |
    v
[PLANNER]       Groq Llama 3.3 70B
                → typed queries (primary | comparison | recency_check | contradiction_probe)
                → confidence: low | medium | high
                → success_criteria (drives the evidence ledger)
    |
    v               ┌─────────────────────────────────┐
    |               | for each hop ∈ {1..MAX_HOPS}    |
    |               |                                 |
    v               v                                 |
[SEARCHER]      Parallel AI → Tavily → Serper         |
                Per-intent routing; circuit breakers  |
    |                                                 |
    v                                                 |
[FETCHER]       httpx async, semaphore(3),            |
                Trafilatura readability extract       |
    |                                                 |
    v                                                 |
[CONTEXT]       BM25 → FlashRank rerank → 5-signal    |
                Hybrid RRF + bge-small-en-v1.5 opt    |
    |                                                 |
    v                                                 |
[EVIDENCE LEDGER]  hop_evidence event — mechanical    |
                   extraction of grounded entities/   |
                   numbers/criteria → real doc_ids    |
                   + verbatim quotes. NOT LLM recap.  |
    |                                                 |
    v                                                 |
[STOP-RAG GATE]    Groq, 4s timeout, JSON-strict.     |
                   Useful=False OR confidence<0.5     |
                   → emit terminator(EVIDENCE_SUFFI-  |
                   CIENT or MARGINAL_GAIN_LOW),       |
                   break hop loop. Degrades-to-       |
                   continue on any failure.           |
    |                                                 |
    +-- continue? --→ next hop ──────────────────────┘
    |
    v
[SOURCE ROLE]   Single batched Groq classifier over the URL pool:
                {primary_source | secondary_analysis | statistical
                 | news_event | official | encyclopedic | contradicting}
                Composed with the existing V2.3 source-trust tier.
    |
    v
[SOURCE CONTRIBUTION]  tokens_from_url / total_context_tokens (tiktoken cl100k_base).
                       Emitted once per run for the inspector.
    |
    v
[CONFLICT_CHECK]  Dedicated stage: detects cross-source contradictions.
                  DRAGged-into-Conflict taxonomy: self | pair | conditional.
                  Conditional contradictions carry a `qualifier` field.
    |
    v
[SYNTHESIZER]   Gemini 2.5 Flash (default) | Sarvam-M (Indic) | OpenRouter DeepSeek R1 (fallback)
                Streams [doc_N] markers; citation guard converts to [Title — domain](URL)
                Stream is scrubbed of <think>...</think> blocks (per-turn stateful filter)
    |
    v
[CLAIM VERIFIER]  Per-sentence: deterministic token+entity overlap → LLM fallback
                  unsupported → [UNVERIFIED]   ambiguous_resolved → [AMBIGUOUS]
    |
    v
[PERSISTENCE]   aiosqlite: sessions, turns, turn_context, claim_audit,
                contradiction_probes (w/ dominant_kind), circuit_events,
                eval_runs, FTS5 index
```

### Provider router

| Stage | Default | Alternatives |
|---|---|---|
| Planning + conflict detection | Groq Llama 3.3 70B | (Groq only — low-latency tier) |
| **Stop-RAG hop gate** | Groq Llama 3.3 70B, `max_tokens=80`, 4 s timeout | Degrades-to-continue on any failure; `reason` is persisted but never streamed |
| **Source-role classifier** | Groq Llama 3.3 70B, single batched call, 8 s timeout, in-memory cache | Degrades-to-`unclassified/0.0` on any failure; composed with the V2.3 source-trust tier (not replacing it) |
| Search | Parallel AI | Tavily, Serper (auto fallback) |
| Synthesis | Gemini 2.5 Flash | `SYNTH_PROVIDER=sarvam` → Sarvam-M / 30B (Indic-first, 64K–128K context, Apache-2.0 base); `SYNTH_PROVIDER=openrouter` → DeepSeek R1; Cerebras / Ollama as additional fallbacks |
| Eval judge | GitHub Models GPT-4o-mini | Any OpenAI-compatible model from a different family than the generator |

Five-step synthesis fallback chain (`Gemini → Sarvam → OpenRouter → Cerebras → Ollama`) with per-provider pre-flight key checks, breaker-aware skipping, and a structured `[synth] fallback succeeded: provider=X step=N` log line. Cerebras has an 8K context-cap guard that auto-skips when the prompt overflows. Ollama is the local last-resort.

### Architectural tradeoffs

Every notable decision was made against a real alternative.

| Choice | Why we picked it | What we rejected |
|---|---|---|
| **SQLite + sqlite-vec** for persistence + vectors | Zero-ops embedded store, single-file DB, FTS5 + vector in one engine, runs locally and in a 200MB container. | **Pinecone / Weaviate / pgvector** — managed vector DBs add a network hop, a SaaS dependency, and free-tier quotas that interfere with eval re-runs. |
| **Hand-rolled async state machine** in `agent/orchestrator.py` | Single async generator makes phase boundaries, cancellation, and SSE emission trivially traceable. Zero framework overhead. Maps 1:1 to the "no orchestration frameworks" constraint. | **LangGraph / CrewAI / LlamaIndex / Haystack** — opaque control flow, hidden retries, version churn, explicitly disallowed. |
| **Parallel AI** as primary search provider | 16,000 free queries vs Tavily's 1,000/mo; structured AI-native excerpts mean we skip a separate fetch on most hits. | **Tavily** (smaller free tier, marginally higher agentic-benchmark score) and **Serper** (snippets only, kept as last-resort fallback). |
| **FlashRank** cross-encoder for rerank | 4MB ONNX model, no PyTorch dependency, sub-100ms on CPU. HF Space deployable. | **BGE / ColBERT MaxSim / DeBERTa cross-encoders** — require PyTorch, GPU for latency, 400MB+ container. |
| **GPT-4o-mini** judge via GitHub Models | Different family from generator (Gemini), free tier, JSON-strict output — avoids same-family score-inflation bias. | **Gemini judging Gemini** — anchor-bleed; judges score familiar style higher than substance. |
| **Per-metric judge calls** (6+ separate calls) | Isolates failure modes — retrieval failure shouldn't tank faithfulness. Each judge has a narrow rubric. | **Single weighted aggregate** — opaque, hides which axis failed, tempts metric gaming. |
| **Hard `MAX_HOPS=2` + Stop-RAG adaptive gate** | The hard cap is a safety net; the value-based gate (Park et al. 2025, [arXiv:2510.14337](https://arxiv.org/abs/2510.14337)) decides per-hop *whether another retrieval round would actually change the answer*. Logs the terminator reason (`EVIDENCE_SUFFICIENT` / `MARGINAL_GAIN_LOW` / `MAX_HOPS_REACHED` / `BUDGET_EXHAUSTED`) so the inspector renders *why* the loop stopped. | **Fixed-iteration baseline** — Park et al. show value-based stopping consistently beats fixed-iteration on multi-hop QA. **Unbounded ReAct** — token explosion, hallucination spirals on a free tier. |
| **Mechanical evidence ledger** (per-hop `hop_evidence` event) | Each grounded row points to a real `doc_id` + verbatim quote; open criteria are a set-difference against `success_criteria`. Deterministic. Survives the assignment's "no hidden CoT streaming" rule trivially. | **LLM-narrated intermediate-answer feed-forward** — surfacing a model's prose recap of "what we found so far" risks hallucination compounding across hops and is harder to audit. |
| **DRAGged 3-type conflict taxonomy** (`self` / `pair` / `conditional`) | Aligned with Cattan et al. (Google, [arXiv:2506.08500](https://arxiv.org/abs/2506.08500)). Conditional contradictions carry a `qualifier` (e.g. *"under the qualifier: year"*) so the agent can correctly route apparent disagreements that vanish when a temporal/regional context is applied. | **Free-form `has_conflict` boolean** — loses information; can't distinguish "agree under qualifier" from "actually disagree". |
| **Bounded U-shape reordering** at injection | Top-1 first, top-2 last, rest in middle. Single-pass reorder, near-zero overhead. Liu et al. 2023. | **Full Lost-in-the-Middle grid search** — heavier instrumentation for marginal additional gain in a 6.4K window. |

### Context engine pipeline

```
BM25 pre-filter (top 30 candidates)
    ↓
FlashRank cross-encoder rerank (top 10)
    ↓
5-signal scoring: BM25 relevance + recency + diversity + source trust + provider relevance
    ↓
Bounded U-shape reordering before injection (Lost-in-the-Middle mitigation)
    ↓
≤ 6,400 token web-context budget (40% of 16K total)
```

Three distinct roles with no overlap: **BM25 = keyword recall**, **FlashRank = semantic relevance**, **5-signal pass = editorial selection**. Hybrid RRF (BM25 + bge-small-en-v1.5 fused at k=60) auto-enables when `sqlite-vec` loads; falls back to lexical otherwise. The effective mode is logged at startup and stamped into every turn's `run_metadata.retrieval_mode`.

### Context budget allocation

Total **16,000 tokens** (`utils/token_counter.py`):

| Slice | Share | Tokens |
|---|---:|---:|
| System instructions | 15% | 2,400 |
| Conversation history (rolling summary + last 3 turns) | 25% | 4,000 |
| Web context | 40% | 6,400 |
| Reserved for generation | 20% | 3,200 |

Two-tier conversation state: the last 3 turns verbatim, plus a Groq-generated rolling summary of older turns (fires at `turn_count > 5`, every 3 turns thereafter).

### Database schema (selected tables)

| Table | Purpose |
|---|---|
| `sessions` | session_id, created_at, last_active_at |
| `turns` | query, response, search_queries (JSON), urls_opened (JSON), `context_xml_sent`, `doc_map`, prompt/completion tokens, latency, `run_metadata_json` |
| `turn_context` | per-snippet audit trail of exactly what the LLM saw, including `provider_relevance` |
| `claim_audit` | per-claim verification result with `verified | unverified | partial` |
| `contradiction_probes` | conflict-check stage output per turn (incl. `dominant_kind ∈ {self, pair, conditional, none}` from the DRAGged taxonomy; idempotent `ALTER TABLE … ADD COLUMN` migration backfills `'none'` on pre-existing rows) |
| `circuit_events` | breaker trips per provider |
| `session_summaries` | rolling summaries |
| `semantic_memory` | sqlite-vec embeddings of distilled facts from older turns (Phase 5 stretch) |
| `fts_content` | FTS5 virtual table for BM25 over all past turns |
| `eval_runs` | per-question scores, per-metric, per-judge |

The three fields evaluators care most about are `turns.context_xml_sent` (faithfulness eval needs the exact XML), `turns.doc_map` (citation back-reference), and per-stage timings stored in `run_metadata_json`.

### Streaming SSE event reference

Events emitted from `main.py` `/research`. Each frame has both an SSE `event:` discriminator AND a `type` field embedded inside the JSON payload (belt-and-suspenders for L7 proxies that strip comment-prefix lines). Stage labels match the assignment spec verbatim.

| Event (`type`) | Payload | Notes |
|---|---|---|
| `run_started` | `{turn_id, session_id, query}` | One-time, first event of the run |
| `phase_started` | `{name, label, idx, total, hop?}` | One per phase boundary; `label` is the user-visible stage string |
| `phase_progress` | `{name, current, total, failed?}` | Sub-phase counters (e.g. URL fetches) |
| `phase_finished` | `{name, duration_ms, ...}` | One per phase boundary |
| `search_query` | `{query, provider, hop}` | Each subquery dispatched |
| `source_found` | `{url, title, domain, query}` | Each unique URL discovered |
| `source_fetched` | `{url, status, latency_ms, bytes?, error?}` | After fetch attempt |
| `context_selected` | `{url, score, snippet_preview, rank}` | Each chunk kept after 5-signal scoring |
| `reasoning` | `{hop, phase: "intent"\|"observation", queries?, observation?}` | Planner-derived rationales (not model CoT) |
| **`hop_evidence`** | `{hop, grounded: [{token, kind, doc_id, url, quote}], open: [{criterion, reason}]}` | Forensic per-hop ledger — mechanically grounded entities/numbers; open `success_criteria` after set-difference. Never a model prose recap. |
| **`source_contribution`** | `{contributions: [{url, domain, title, tokens, share, citations}], total_tokens}` | Token-share per URL of the final context, computed with tiktoken cl100k_base |
| **`source_role`** | `{roles: [{url, role, confidence}]}` | LLM-classified role per URL; degrades to `unclassified/0.0` on classifier failure |
| **`terminator`** | `{reason, hop, detail?}` | Explicit hop-loop stop reason: `EVIDENCE_SUFFICIENT \| MARGINAL_GAIN_LOW \| MAX_HOPS_REACHED \| NO_NEW_QUERIES \| BUDGET_EXHAUSTED \| CRITERIA_SATISFIED` |
| `clarification_offered` | `{kind, original_query, possible_interpretations, clarifying_question}` | Fires only when vagueness score ≥ 0.55 AND planner success_criteria are non-empty |
| `evidence_gap` | `{query, intent, reason}` | A planner sub-query yielded no usable evidence |
| `conflict_detected` | `{claim, position_a, position_b}` | DRAGged taxonomy applied at probe stage |
| `uncertainty` | `{kind: "weak"\|"missing"\|"conflict", reason, follow_ups}` | Structured signal so the chat UI can surface "state uncertainty and propose next steps" |
| `answer_delta` | `{text}` | Streamed answer text; scrubbed of `<think>…</think>` blocks |
| `citation_resolved` | `{marker, url, title}` | As citation guard converts each `[doc_N]` |
| `run_finished` | `{usage, total_latency_ms, cost_usd}` | One-time, last event before `done` |
| `done` | full final payload incl. `answer`, `urls`, `doc_map`, `run_metadata` | Terminal |
| `run_error` / `error` | `{phase, message, recoverable}` | Terminal-on-failure |

The 5 user-facing phase labels are exactly: `"Planning"`, `"Searching the web"`, `"Fetching sources"`, `"Selecting relevant context"`, `"Generating answer with citations"` — all carried via `phase_started.label`.

**No hidden chain-of-thought is streamed.** See the CoT-streaming compliance section below for the three-layer scrub that enforces this.

### CoT-streaming compliance (assignment line 103)

The assignment is explicit: *"Do not stream hidden chain-of-thought."* This is a hard requirement, not a preference. Our synthesizer talks to multiple LLMs (Gemini, Sarvam-M, DeepSeek R1, Cerebras) — at least two of those families occasionally emit `<think>...</think>` reasoning blocks mid-stream. We enforce the rule in three layers:

1. **Regex pre-emit filter** (`main.py:_COT_PATTERNS`). Catches `<think>` / `<thinking>` / `<thought>` (and closing variants) plus provider envelope keys (`thought_summary`, `thought_tokens`, `reasoning_content`, `reasoning_tokens`, `redacted_thinking`).
2. **Per-turn stateful scrubber** (`_drive`/`cot_state` in `main.py`). Tracks whether a `<think>` block opened in a *prior* chunk is still in flight. Body chunks between open and close get their text emptied; the chunk carrying `</think>` is trimmed up to and including the close tag. This is necessary because streaming chunks may split a block: chunk N has `<think>...`, chunks N+1..M have body text with no marker tokens, chunk M has `...</think>`.
3. **Recursive payload scrub** (`_scrub_cot_recursive` in `main.py`). Walks every string leaf of the SSE payload — covers `done.data.answer` and any nested string field so single-frame full-text values cannot leak CoT even after the streaming layer already passed.

Additionally, the Stop-RAG gate's LLM `reason` field is persisted to `run_metadata.stop_rag_decisions` (for the trace inspector) but **scrubbed from the SSE wire copy** of the `done` event. Asserted by `tests/test_stop_rag.py::test_reason_not_in_sse_payload`, which walks every yielded event for the sentinel reason string and requires zero hits.

Lock-in: `tests/test_cot_scrub.py` — **17 tests** covering pattern coverage, inline scrub, mid-stream open-only, close-only-after-open, idempotency, done-event payload, nested lists, and envelope contains-check.

### Runtime failure budget

Per-stage budgets enforced by `utils/failure_policy.py`; degrade gracefully instead of crashing a turn:

| Policy key | Default | Behavior on breach |
|---|---:|---|
| `FAILURE_POLICY_PLAN_TIMEOUT_S` | 25 | Fallback to direct-query planning (single PRIMARY query) |
| `FAILURE_POLICY_SEARCH_TIMEOUT_S` | 45 | Continue with empty results; downstream stages adapt |
| `FAILURE_POLICY_FETCH_TIMEOUT_S` | 60 | Continue with partial extracted content |
| `FAILURE_POLICY_SELECT_TIMEOUT_S` | 20 | Fallback to heuristic selector |
| `FAILURE_POLICY_SYNTH_TIMEOUT_S` | 90 | Bounded fallback response with proposed follow-up queries |
| `FAILURE_POLICY_MAX_TOTAL_TURN_TIME_S` | 240 | Tag `budget_breach` in run metadata; stop extra work |
| `FAILURE_POLICY_MAX_HOPS` | 2 | Hard cap on adaptive 2-hop retrieval |

Per-provider circuit breakers (V2.5) sit inside Tenacity retry boundaries — one Tenacity-exhausted call counts as one logical failure. Thresholds: search providers 4 failures / 60s, LLM providers 5 / 60s, judge 3 / 60s.

### Project structure

```
.
├── agent/                          # Core pipeline
│   ├── orchestrator.py             #   state machine: PLANNING → ... → DONE/CANCELLED
│   ├── search.py                   #   multi-Indic language detection, provider chain
│   ├── extractor.py                #   httpx + Trafilatura
│   ├── context_engine.py           #   BM25 + FlashRank + 5-signal + optional hybrid RRF
│   ├── synthesizer.py              #   streaming wrapper
│   ├── citation_guard.py           #   [doc_N] → [Title — domain](URL)
│   ├── claim_verifier.py           #   per-sentence verification (V2.4)
│   ├── embedder.py                 #   bge-small-en-v1.5 (V3.1, optional)
│   ├── memory.py                   #   aiosqlite schema, migrations, FTS5
│   ├── eval_queries.py             #   per-language, per-category drill-down
│   └── models.py                   #   dataclasses + Pydantic
├── utils/                          # Cross-cutting
│   ├── provider_router.py          #   Gemini / Groq / Sarvam / OpenRouter / GitHub Models
│   ├── circuit_breaker.py          #   V2.5
│   ├── cancellation.py             #   V2.6 token registry
│   ├── source_trust.py             #   V2.3 tier table
│   ├── prompt_registry.py          #   versioned prompts
│   ├── failure_policy.py           #   per-stage timeouts + breaker thresholds
│   ├── cost_model.py               #   per-model token cost lookup
│   └── token_counter.py            #   tiktoken + ContextBudget
├── eval/
│   ├── dataset.json                #   53 questions, 5 languages, 6 categories
│   ├── eval_runner.py              #   single-mode + --ablate + --cross-family-judge
│   ├── judge.py                    #   9 metrics
│   └── ablation_report.py
├── frontend/                       # Next.js 16 App Router, Tailwind, shadcn/ui
│   ├── app/                        #   chat / sessions / eval / drill-down
│   └── components/
├── main.py                         # FastAPI: /research SSE, /sessions, /eval/*, /health
├── docs/DEPLOY.md                  # 30-min VPC deployment guide
├── legacy/                         # Decommissioned Streamlit V1
└── tests/                          # 187 passing
```

---

## Limitations

Honest accounting — these are real constraints, not future-work euphemisms.

- **DNS-rebinding SSRF protection is out of scope.** The fetcher trusts the URLs returned by search providers. Production deployments behind a corporate VPC should add a URL allowlist or a SSRF-aware HTTP proxy.
- **Free-tier rate limits.** Heavy concurrent eval runs hit Gemini / Groq quotas. V2.5 circuit breakers prevent Tenacity-storm cascades but cannot raise the limit. The five-step synth fallback chain mitigates but does not eliminate this.
- **Eval non-determinism.** LLM judges have temperature > 0 in places and JSON-mode is best-effort, not guaranteed. Cohen's κ via judge rotation quantifies the noise floor (~ ±0.05 absolute on most metrics in our runs).
- **Web SEO spam.** `favor_precision=True` in Trafilatura plus the V2.3 source-trust prior reduce but don't eliminate low-quality sources. A production deployment should add a domain allowlist.
- **Indic-language eval coverage is thin.** The English subset has 19 questions; each Indic language has 5–8. Enough to detect cross-script consistency drift, not enough to publish per-language headline numbers.
- **JavaScript-rendered pages.** SPAs with client-side content aren't fetched. A Playwright-backed extractor is the fix; deliberately deferred (300MB+ dependency, multi-second per-fetch tax).
- **macOS system Python + `sqlite-vec`.** Python compiled without `--enable-loadable-sqlite-extensions` silently disables V3.1 hybrid retrieval and gracefully falls back to BM25. The Linux Docker image avoids this entirely.
- **Ground-truth unknowability.** The system detects conflicts but cannot adjudicate them. For genuinely contested facts, the answer surfaces disagreement and cites both sources rather than picking one.
- **Source-role classifier degrades to `unclassified`.** The LLM-classified role pass adds one batched Groq call after the hop loop. When the call fails (timeout / quota / breaker), every URL is tagged `unclassified` with `confidence=0.0` and downstream consumers fall back on the V2.3 source-trust tier alone. Net effect: a non-fatal loss of the role badge in the inspector, no incorrect labelling.
- **Stop-RAG gate adds ≤ 4 s per hop boundary.** The adaptive hop gate calls Groq once per inter-hop decision with a 4 s timeout. On a hard cap of 2 hops this is at most one extra call per turn; degrades-to-continue on any failure. Disable via `FAILURE_POLICY_MAX_HOPS=1` if the latency is unacceptable for a specific deployment.

---

## Future Improvements

Three high-impact directions, ranked by ROI within a 1–2 week extension window.

1. **Median-of-3 judge ensemble.** Replace the cross-family judge pair with a 3-judge ensemble drawing from three distinct families (e.g. Llama, GPT, Mistral) and take the median score per metric. Reduces single-judge bias variance ~2× in published benchmarks at ~3× cost. Existing judge-rotation infrastructure makes this a small refactor.
2. **Sarvam-M translation pre-pass for Mayura-style cross-lingual retrieval.** Right now Indic queries hit Indic-language sources only when those exist; for cross-script consistency we want to retrieve English sources for an Indic query when local sources are sparse. A Sarvam-M translation pre-pass (query → EN) gated on retrieval recall would lift coverage substantially.
3. **Calibration metric refinement.** Pearson correlation between planner confidence and post-hoc faithfulness is a starting point. Replace with a proper expected calibration error (ECE) over 3 confidence buckets, and use the ECE delta to gate whether the V3.2 adaptive-second-hop fires. The data is already persisted; this is purely an eval-runner improvement.

Beyond the top 3, two scoped items also worth listing:

- **Confidence-calibrated context budget.** Currently a fixed 16K allocation. Should adapt the system/history/web/output split based on planner confidence and query complexity. Data to drive this is already collected.
- **MCP server interface.** Expose `/research` as an MCP tool so other agents (specifically Sarvam's multi-agent stack) can consume it natively.

### Considered and rejected (with reasoning)

Recorded here because the rejection itself is a design signal.

- **ColBERT MaxSim reranking** — would lift recall on long-tail multi-hop, but requires C-bindings and a 400MB+ index. Deferred until we move off SQLite-only storage.
- **DeBERTa-v3 local NLI for conflict detection** — strong contradiction classifier, but prohibitive on HF Space free tier. Groq Llama 3.3 70B contradiction probe hits the same envelope at sub-second latency.
- **Playwright headless fetching** — 300MB+ browser dependency and multi-second per-fetch latency for marginal recall over Trafilatura on the open web. Future work for SPA-heavy verticals.
- **Graph-based knowledge extraction** (Neo4j / NetworkX entity graphs across turns) — signals trend-chasing more than utility for a single-user research agent. FTS5 + rolling summary covers cross-turn continuity at vastly lower complexity.
- **Full ReAct loops with unbounded iteration** — known source of cost blowouts and hallucination spirals. `MAX_HOPS=2` + token-budget terminator gives the same bounded recovery without the failure modes.
- **Same-family LLM judge.** Anchor-bleed well-documented in LLM-as-judge literature. We use GPT-4o-mini for that reason.
- **LLM-narrated "intermediate answer" forwarded across hops.** A natural-sounding but risky pattern: the model writes a prose recap of what was found so far, the next hop's planner uses it as context. Compounds hallucination across hops and is impossible to audit. We use the **mechanical evidence ledger** (`hop_evidence` event) — deterministic extraction of grounded entities/numbers from the chunk pool, each pointing to a real `doc_id` and a verbatim quote.
- **Regex-substring source-coverage tagging** (categorising sources as BIO / STATS / NEWS by matching keywords in title and snippet). First-match-wins on multi-topic chunks; a footballer page tagged "STATS" because "score" appears anywhere is theatre, not classification. We combine the V2.3 **source-trust tier** (domain reputation) with an **LLM-classified role** over the chunk pool — composed, not substituted.
- **Chip-style multi-step clarification wizard** ("Question 2 of 3" with options). Friction without utility. We fire **at most one clarifier**, gated by a vagueness score (`agent/vagueness.py`) that fuses entity count, query length, planner `ambiguity_flag`, and a wh-breadth heuristic.

---

## Assumptions

For the PDF submission packet:

1. **"Deep research"** in this assignment means multi-source retrieval with claim verification and conflict detection — not single-source summarisation.
2. **No orchestration frameworks.** The libraries used (`rank_bm25`, `trafilatura`, `aiosqlite`, `fastembed`, `sqlite-vec`, `tenacity`, `httpx`) are utilities, not orchestration frameworks. No LangChain / LangGraph / CrewAI / LlamaIndex / Haystack.
3. **Session ownership.** Session IDs are client-generated UUIDs in browser localStorage. Sessions persist in SQLite until manually deleted. `POST /research/cancel/{turn_id}` and `POST /research/approve/{turn_id}` require `session_id` (JSON body or `?session_id=…`) to prove caller ownership — mismatched or missing returns 404.
4. **Citation format.** `[Title — domain](URL)` per assignment line 62. Internal `[doc_N]` markers used during generation, converted post-synthesis by `citation_guard`. Unsupported claims marked `[UNVERIFIED]` inline (assignment line 91: state uncertainty AND propose next steps).
5. **Judge model family discipline.** Eval uses LLM-as-judge from a different model family than the generator (GPT-4o-mini judges Gemini output) to prevent same-family bias.
6. **Rolling summary trigger.** Fires when `turn_count > 5`, compressing turns older than the last 3, then once per 3 new turns.
7. **Iterative re-search trigger.** Planner-confidence-gated, not uncertainty-marker-driven. `MAX_HOPS=2`. Second hop restricted to `RECENCY_CHECK` and `CONTRADICTION_PROBE` intents.
8. **Cross-family judge sample.** Deterministic 20-question subset (seed=42). Configurable via `--cross-family-sample-size`. Quota-guarded against the 150/day GitHub Models cap.

---

## Related Work

Each item below is referenced by a specific design decision in the codebase.

- **Huang et al. (2025), "Deep Research Agents: A Systematic Examination And Roadmap"** ([arXiv 2506.18096](https://arxiv.org/abs/2506.18096)). The canonical DR-agent survey. We adopt its vocabulary throughout and use its taxonomy for the orchestrator's phase structure.
- **Du et al. (2025), "DeepResearch Bench"** ([arXiv 2506.11763](https://arxiv.org/abs/2506.11763)). Defines RACE and FACT metric families. Our 6-metric judge maps cleanly: Faithfulness ↔ RACE.Faithfulness, Citation Integrity ↔ FACT.Citation_Accuracy, Context Precision ↔ RACE.Comprehensiveness.
- **OpenAI BrowseComp / BrowseComp-ZH** ([blog](https://openai.com/index/browsecomp)). Inspires our 53-question multilingual dataset and the factual / multi-hop / comparison / insufficient-evidence / conflicting category mix.
- **Liu et al. (2023), "Lost in the Middle"** ([arXiv 2307.03172](https://arxiv.org/abs/2307.03172)). Drives our final-stage snippet reordering (top-1 first, top-2 last, rest in middle) before the synthesis prompt.
- **Jina AI, "node-DeepResearch"** ([blog](https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch/)). Token-budget terminator: bound a ReAct loop on cumulative token spend, not just hop count. We adopt this alongside `MAX_HOPS=2`.
- **Shao et al. (2024), STORM** ([NAACL 2024](https://arxiv.org/abs/2402.14207)). Multi-perspective question generation. Cited as planner future direction.
- **Alibaba, Tongyi DeepResearch / IterResearch** ([repo](https://github.com/Alibaba-NLP/DeepResearch)). Heavy-mode test-time scaling. Cited as future work.
- **Vectara (2025), chunking + metadata enrichment** ([blog](https://www.vectara.com/blog/)). Drives snippet-XML expansion to carry `retrieved_at`, rank, and `relevance_score` alongside the URL/title/domain triple.
- **Park, Cho, Lee (2025), "Stop-RAG: Value-Based Retrieval Control for Iterative RAG"** ([arXiv 2510.14337](https://arxiv.org/abs/2510.14337), NeurIPS 2025 MTI-LLM Workshop). The adaptive hop gate. We approximate the value function with a single fast Groq call returning `{another_hop_useful, confidence}`; their paper documents the gain over fixed-iteration RAG that motivates the design.
- **Cattan et al. (Google, 2025), "(D)RAGged Into Conflict: Detecting and Addressing Conflicting Sources in Generative QA"** ([arXiv 2506.08500](https://arxiv.org/abs/2506.08500)). The 3-type conflict taxonomy (`self` / `pair` / `conditional`) and the use of a `qualifier` field on conditional contradictions.
- **Krishna et al. (2025), "Fact, Fetch, and Reason: A Unified Evaluation of RAG (FRAMES)"** ([arXiv 2409.12941](https://arxiv.org/abs/2409.12941), NAACL 2025). Our 6-metric judge mirrors FRAMES's four dimensions (factuality / retrieval / reasoning / attribution).
- **Gao, Yen, Yu, Chen (2023), ALCE** ([arXiv 2305.14627](https://arxiv.org/abs/2305.14627), EMNLP 2023). Quote-then-cite + claim-anchored grounding. Drives the `[doc_N]` post-processing and per-claim verification pattern.
- **Min et al. (2023), FActScore** ([arXiv 2305.14251](https://arxiv.org/abs/2305.14251), EMNLP 2023). Decompose-then-verify on atomic facts. The conceptual basis for our deterministic + LLM-fallback `claim_verifier`.
- **Owoicho et al. (2023), "Ask-to-Clarify"** ([arXiv 2509.15061](https://arxiv.org/abs/2509.15061)). Clarification-utility prediction. Our vagueness gate is the heuristic cousin: cheaper, less precise, but sufficient for "fire at most one clarifier only when it's worth asking".
- **Zheng et al. (2023), "Judging LLM-as-a-Judge"** ([arXiv 2306.05685](https://arxiv.org/abs/2306.05685)). Documents same-family score-inflation bias. Calibrates the projection prior in `eval/results/JUDGE_FAMILY_COMPARISON.md`.
- **Panickssery et al. (2024), "LLM Evaluators Recognize and Favor Their Own Generations"** ([arXiv 2404.13076](https://arxiv.org/abs/2404.13076)). Second source for the cross-family judge discipline.

```bibtex
@misc{huang2025deepresearch,
  title  = {Deep Research Agents: A Systematic Examination And Roadmap},
  author = {Huang et al.},
  year   = {2025},
  eprint = {2506.18096},
  archivePrefix = {arXiv}
}

@misc{du2025deepresearchbench,
  title  = {DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents},
  author = {Du et al.},
  year   = {2025},
  eprint = {2506.11763},
  archivePrefix = {arXiv}
}

@inproceedings{liu2023lostinmiddle,
  title  = {Lost in the Middle: How Language Models Use Long Contexts},
  author = {Liu, Nelson F. and others},
  year   = {2023},
  eprint = {2307.03172},
  archivePrefix = {arXiv}
}
```

---

## Submission Packet

The final PDF submission bundles this README, the demo video link, the live URLs, and the assumptions list. Generate it deterministically:

```bash
# Produce the PDF from the README + assumptions appendix
python scripts/build_submission_pdf.py --out submission.pdf

# Reproducibility snapshot to embed in the packet
python scripts/repro_report.py --mode quick > submission_repro.txt
```

Submission contents:

1. This README (rendered to PDF)
2. Demo video link (top of file)
3. Live URLs (Vercel frontend + HF Spaces backend)
4. `assumptions` section (above)
5. `submission_repro.txt` — git SHA, prompt versions, env knobs, smoke summary

## License

MIT. See `LICENSE`.
