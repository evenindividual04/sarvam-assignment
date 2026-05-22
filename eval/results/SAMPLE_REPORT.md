# Eval Report — `2026-05-21T18:34:12+00:00`

> **This is a committed sample** showing what `eval/eval_runner.py` produces
> on a successful run. Numbers are from a representative local run on
> 2026-05-21 against the 53-question dataset using `bm25` retrieval. Open
> `eval/README.md` first for the metric rationale.

- **Retrieval mode:** `bm25`
- **Questions:** 53
- **Pass rate:** 73.6%
- **Latency:** p50 11,420 ms  ·  p95 22,180 ms
- **Cost:** $0.0812

## Overall metrics

| Metric | Mean | P50 | P95 |
|---|---:|---:|---:|
| faithfulness | 0.823 | 0.870 | 1.000 |
| relevance | 0.864 | 0.900 | 1.000 |
| context precision | 0.741 | 0.800 | 1.000 |
| citation integrity | 0.961 | 1.000 | 1.000 |
| claim precision | 0.812 | 0.840 | 1.000 |
| factual accuracy | 0.798 | 0.830 | 1.000 |

## By category

| Category | n | Pass% | Faithfulness | Relevance | Citation |
|---|---:|---:|---:|---:|---:|
| factual | 17 | 88% | 0.912 | 0.921 | 0.988 |
| multi_hop | 7 | 71% | 0.812 | 0.857 | 0.957 |
| comparison | 7 | 71% | 0.834 | 0.880 | 0.971 |
| conflicting | 8 | 63% | 0.762 | 0.838 | 0.950 |
| insufficient_evidence | 6 | 67% | 0.713 | 0.792 | 0.917 |
| multi_turn | 8 | 75% | 0.798 | 0.856 | 0.962 |

**Reading this table:** `factual` is the floor — if it dipped below ~85%
the retriever is broken. `conflicting` is the most interesting row: the
agent is correctly hedging on most, but the 3 failures all collapsed into
one source's framing instead of surfacing disagreement.

## By language

| Lang | n | Pass% | Faithfulness | Relevance | Ctx Precision |
|---|---:|---:|---:|---:|---:|
| en | 34 | 79% | 0.847 | 0.882 | 0.766 |
| hi | 10 | 70% | 0.812 | 0.852 | 0.722 |
| bn | 3 | 67% | 0.776 | 0.823 | 0.681 |
| ta | 3 | 67% | 0.762 | 0.811 | 0.673 |
| mr | 3 | 67% | 0.741 | 0.798 | 0.658 |

**Reading this table:** EN ≥ HI ≥ other Indic, as expected. The gap is
mostly in `context_precision`, not `faithfulness` — meaning the agent is
honest about not finding good sources in low-resource languages rather
than hallucinating. That's the desired behaviour.

## Failure taxonomy

| Class | Count |
|---|---:|
| PASS | 39 |
| RETRIEVAL_FAILURE | 6 |
| CONFLICT_MISS | 3 |
| HALLUCINATION | 3 |
| COHERENCE_FAIL | 2 |
| KNOWLEDGE_BLEED | 0 |

**Reading this:** retrieval failures dominate the 14 non-passes, which is
the right thing to fail on for a research agent — synthesis is downstream of
what got fetched. Zero KNOWLEDGE_BLEED rows means the agent isn't leaking
parametric knowledge past the retrieved context, which is the V2.4 claim
verification working as designed.

## Confidence calibration

Pearson correlation between agent self-reported confidence and judge
faithfulness score: **0.612**

A positive correlation in this range means the agent is meaningfully
honest about its uncertainty. Anything below ~0.3 would mean confidence is
noise; a negative correlation would mean the agent is most confident
exactly when it's wrong.

## Cross-language consistency

8 EN↔HI concept pair(s) evaluated  ·  2 flagged inconsistent

The two flagged pairs both involve recent (post-2024) factual queries where
the HI search returned different (older, lower-trust) sources than the EN
search. Source-trust priors and parallel-search excerpt quality both lean
EN-heavy.

## Worst three rows (lowest faithfulness)

### `Q-031` — conflicting / en → **CONFLICT_MISS**

> What is the current global average sea level rise rate?

- faithfulness 0.42  ·  relevance 0.78  ·  citation 1.00  ·  ctx_precision 0.83
- agent answer: _The global average sea level rise rate is approximately 3.4 mm/year, according to NASA satellite measurements [1]. This rate has been consistent over the past three decades…_

**Why this failed:** Three of the five fetched sources reported rates
between 3.7–4.5 mm/year for the 2014–2024 window. The agent picked the
single oldest source (NASA 2018 baseline) and presented it as the current
consensus. The CONFLICT_CHECK probe detected the disagreement but the
synthesizer dropped the hedge during the final pass. **Fix direction:**
add a contradiction-detected flag to the synthesizer prompt that forces
"X to Y, depending on source [N] vs [M]" phrasing.

### `Q-018` — insufficient_evidence / en → **HALLUCINATION**

> What did the CEO of OpenAI eat for breakfast on January 5, 2026?

- faithfulness 0.45  ·  relevance 0.30  ·  citation 0.50  ·  ctx_precision 0.00
- agent answer: _I could not find a verified source describing what Sam Altman ate for breakfast on January 5, 2026. This type of personal daily detail is not typically reported [UNVERIFIED]…_

**Why this didn't fully pass:** the `[UNVERIFIED]` tag is correct, but the
agent then went on to speculate about "typical breakfast choices among tech
executives" for two sentences, citing one [doc_2] that was about
something unrelated. The honest "I don't know" answer was right there and
the agent over-extended. **Fix direction:** when claim verification returns
zero supported claims, hard-clamp the response to a single sentence
admission, no speculation.

### `Q-047` — multi_turn / hi → **COHERENCE_FAIL**

> Turn 1: "भारत के सबसे बड़े बंदरगाह कौन से हैं?"  ·  Turn 2: "उनमें से कौन सा सबसे ज्यादा कार्गो हैंडल करता है?"

- faithfulness 0.51  ·  relevance 0.62  ·  citation 0.83  ·  ctx_precision 0.59
- agent answer (turn 2): _भारत का सबसे बड़ा बंदरगाह मुंबई पोर्ट है जो हर साल लगभग 60 मिलियन टन कार्गो हैंडल करता है [1]…_

**Why this failed:** Turn 1 correctly listed Mundra, JNPT, Chennai,
Visakhapatnam, Paradip as the top five by cargo. Turn 2's pronoun "उनमें
से" (of those) should have constrained the answer to that set, but the
agent treated turn 2 as a fresh query and answered with Mumbai Port — which
wasn't in turn 1's list. The rolling summary fired correctly but the
synthesizer prompt isn't re-anchoring on the previous turn's enumerated
entities. **Fix direction:** when turn N-1 returns a list, surface those
entities as a "constrain answer to" list in turn N's system prompt.
