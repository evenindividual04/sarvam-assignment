# Evaluation Harness

This directory contains the runnable evaluation script that scores the Deep
Research Agent on a curated question set. It produces a JSONL of raw rows, a
JSON summary, and a human-readable markdown report per run.

> **Reproducing a run:** from the repo root,
> `python eval/eval_runner.py` (sequential, ~15–25 min depending on rate
> limits). Add `--ablate` to run BM25-only and BM25+vector hybrid back-to-back
> and emit a delta report.

---

## What we measure and why

Seven core metrics, each scored by a **separate** judge call so the model
can't anchor one score on another. Citation Integrity and Claim Precision are
**deterministic** — no LLM needed.

| Metric | What it asks | Why it's here |
|---|---|---|
| **Faithfulness** | Is every claim supported by the retrieved context? | The single biggest failure mode of RAG systems is plausible-sounding sentences that aren't in the source material. This is the headline metric. |
| **Answer Relevance** | Does the answer address the user's actual question? | Faithfulness alone rewards a system that copies the source without answering. Relevance catches that. |
| **Context Precision** | Did retrieval surface the *necessary* information? | Separates retrieval failures from synthesis failures. If context_precision is low but faithfulness is high, the agent honestly said "I don't know" — that's a pass for the *agent* but a fail for the *retriever*. |
| **Citation Integrity** _(deterministic)_ | Do all `[N]` markers resolve to URLs that were actually fetched? | LLM judges over-credit citation work. A regex + URL set membership check is cheaper and more reliable. |
| **Claim Precision** _(deterministic + LLM fallback)_ | Do cited sentences actually appear in the cited source chunks? | Distinguishes HALLUCINATION_ATTRIBUTION (valid URL, wrong claim) from HALLUCINATION_FACT (made-up content) — each has a different fix. |
| **Conflict Adherence** | When sources contradict, does the answer surface the disagreement? | The assignment explicitly calls out "conflicting sources" as a behaviour to test. A confident answer that hides disagreement is worse than a hedged one that shows it. |
| **Session Coherence** | In multi-turn scenarios, does turn N use turn N-1's context correctly? | Tests the rolling-summary + recent-turns memory design, not just single-shot QA. |

**Auxiliary grounding signals** computed per-run (not part of the core seven,
but reported for diagnostics):

- **Factual Accuracy** — substring/entity match against gold answers where available.
- **Quote Grounding Ratio** — fraction of inline quotes that appear verbatim in their cited source.
- **Numeric Grounding Ratio** — fraction of numeric tokens (years, percentages) found in cited sources.
- **Criteria Coverage Ratio** — fraction of planner's success criteria present in the answer.
- **Script Preservation Ratio** — for Indic-language answers, fraction of alphabetic chars in the expected Unicode script.
- **Cross-language consistency** — Jaccard over cited URL sets for paired EN/Indic questions on the same concept.
- **Confidence calibration** — whether the agent's hedging language tracks actual evidence quality.

**Failure taxonomy** — every row is bucketed into one of seven mutually
exclusive classes: PASS / HALLUCINATION_FACT / HALLUCINATION_ATTRIBUTION /
KNOWLEDGE_BLEED / RETRIEVAL_FAILURE / CONFLICT_MISS / COHERENCE_FAIL.
Splitting HALLUCINATION into two sub-classes reveals whether the root cause is
bad retrieval (FACT: the information was never in the context) or bad synthesis
(ATTRIBUTION: the citation format is valid but the claim isn't in the cited doc).

### Metrics we considered and rejected

- **BLEU / ROUGE.** Surface n-gram similarity to a gold answer doesn't
  measure grounding. Two answers can have identical BLEU and opposite truth
  values. Useless for a research agent.
- **Single combined "quality" score.** Hides the diagnostic signal. The whole
  point of separate metrics is that "faithfulness 0.9, citation 0.3" tells you
  exactly where to dig — averaging them tells you nothing.
- **Human eval.** Out of scope for a 48h assignment and not reproducible by
  graders. LLM-as-judge with a different model family is the realistic
  substitute.

### Why the judge is a different model family

A model tends to rate its own family's style more favorably, inflating scores
by 5–15 percentage points compared to a cross-family judge. The judge defaults
to **Groq Llama 3.3 70B** (Meta family) — cross-family when the synthesizer
is Gemini, free at 14,400 req/day, fast.

**Configurable via env:** `JUDGE_PROVIDER` in `{groq, github}` (default
`groq`) and `JUDGE_MODEL` (default `llama-3.3-70b-versatile` for groq,
`gpt-4o-mini` for github). If you switch the synthesizer to Sarvam (which is
also Llama/Meta-based), set `JUDGE_PROVIDER=github` to preserve the
cross-family invariant with GPT-4o-mini (OpenAI family) as the secondary judge.

The GPT-4o-mini judge via GitHub Models has a 150 req/day ceiling, which
can't finish a single 76-question × 7-metric eval run on a fresh day
(~532 calls needed). Groq's 14,400/day removes the bottleneck and is the
recommended default.

---

## Dataset

76 questions in `dataset.json`, distributed across the categories the
assignment calls out:

| Category | n | What it tests |
|---|---:|---|
| `factual` | 21 | Single-hop, single-source answers. Baseline for "does retrieval work at all." |
| `multi_hop` | 13 | Answer requires combining two+ documents. Tests context_engine selection. |
| `comparison` | 10 | "X vs Y" — tests whether the agent retrieves balanced sources. |
| `conflicting` | 10 | Two sources disagree. Tests conflict detection + conflict adherence metric. |
| `insufficient_evidence` | 12 | Web doesn't have the answer. Tests honest "I don't know" behaviour. |
| `multi_turn` | 10 | Two turns where turn 2 depends on turn 1's context. Tests memory. |

**Language coverage** (cross-lingual robustness — explicitly asked for in the
FDSE assignment because Sarvam is Indic-first):

| Language | n |
|---|---:|
| English | 44 |
| Hindi | 17 |
| Tamil | 5 |
| Bengali | 5 |
| Marathi | 5 |

A subset of questions shares a `concept_id` tag linking English and Indic
versions of the same question, used to compute **cross-language consistency**
(Jaccard over the cited URL sets — do EN and HI converge on the same evidence?).

---

## Outputs per run

Files are emitted to `eval/results/`:

| File | Purpose |
|---|---|
| `eval_<ts>.jsonl` | One row per question with all judge scores, latency breakdown, token counts, cost, failure class. |
| `summary_<ts>.json` | Aggregates (means, P50/P95), failure taxonomy counts, calibration, cross-language pairs. |
| `report_<ts>.md` | Human-readable report. Open this first. Includes a "worst three rows" section with concrete failure cases. |
| `calibration_<ts>.json` | Confidence ↔ faithfulness correlation and per-row pairs. |
| `ablation_<id>.json` | _Only with `--ablate`._ BM25 vs hybrid delta report. |

The same rows are also persisted to the `eval_runs` SQLite table, which the
Next.js `/eval` page reads from for browsing past runs.

---

## Interpreting the report

For an evaluator opening a fresh `report_<ts>.md`:

1. **Overall pass rate.** If <70%, the dominant failure class in the taxonomy
   tells you where to look.
2. **By category.** Compare `factual` (should be high) against `conflicting`
   and `insufficient_evidence` (the interesting ones). Most systems pass
   factual and collapse on the rest.
3. **By language.** EN ≥ HI ≥ other Indic is the expected gradient. A
   surprisingly *flat* curve means either the retrieval works cross-lingually
   or all languages are equally bad — check `context_precision` to
   disambiguate.
4. **Worst three rows.** Three concrete failure cases with question + agent
   answer + scores, so you can read the actual output instead of trusting an
   aggregate.
5. **Calibration.** A negative correlation here is a red flag: it means the
   agent is most confident when it's most wrong.

---

## Known limitations

- **Judge variance.** Single judge call per metric → ~±0.05 noise on means.
  Acceptable for relative comparisons across runs, not for absolute claims.
- **Dataset size.** 76 questions is enough to catch dominant failure modes
  but not enough to detect subtle regressions <5pp.
- **Cost.** Each full run hits the generator stack 76× and the judge stack
  ~5×76 ≈ 380× (LLM-judged metrics only). Budget ~$0.05–0.15 per run
  depending on synthesis provider.
- **Generator temperature.** The synthesizer runs at ~0.1–0.3 (not fully
  deterministic); the judge runs at temperature 0. Repeat runs will produce
  slightly different per-row scores.
- **No human ground truth on Indic questions.** The judge is multilingual
  but its calibration on Tamil/Bengali/Marathi is weaker than English.
