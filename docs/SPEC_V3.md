# Spec V3 — Citation Hover-Preview + Sarvam Alignment Cluster

**Status**: Refined draft for approval
**Estimated effort**: ~9 hrs across 4 phases (refined from 7.5 after adding telemetry + caching + a11y specifics)
**Target**: Submission-ready demo with strong Sarvam-specific evaluator signals
**Author**: Interactive design session, 2026-05-22 (refined pass)

---

## Context

The Deep Research Agent is at v2 — full provider stack, 461 tests passing, frontend polished, deployed to Vercel + HF Spaces. Two improvement areas surfaced during review:

1. **Citations are first-class in the architecture (provenance audit, quote-first prompt, claim verification) but visually inert in the UI** — they're plain markdown links. Hovering them should expose the provenance work we've done.
2. **This is Sarvam AI's assignment** — the eval rubric and the cultural read both reward features that lean into Sarvam's own stack (Sarvam-M for Indic synthesis, Mayura for translation, source-trust for Bharat-relevance, eval-visible Indic performance).

V3 ships 7 surgical changes — 1 universal (hover cards), 6 Sarvam-aligned — that together create a coherent demo narrative without scope creep into voice/personas/branching.

---

## Feature Specs

### F1 — Citation Hover-Preview Cards

**User story**: As a user reading an answer, I want to hover any `[Title — domain](URL)` citation and see *where the claim came from* — the snippet excerpt, which provider found it, its trust tier — without losing my place by opening a new tab.

**Functional requirements**:
- Hover (or focus via keyboard) on any inline citation in a rendered answer → preview card appears within 200ms.
- Card displays:
  - Title (Instrument Serif italic, max 2 lines)
  - Domain + trust tier badge (Mono)
  - Snippet excerpt (≤200 chars, sans, with `…` ellipsis if truncated)
  - Provenance line (Mono): "Found via Parallel · Reranked by Cohere · Rank #2"
  - "Open ↗" button → same as click on the citation itself
- Card dismisses on pointer leave with 100ms grace period (avoid flicker)
- Keyboard accessible: Tab to citation → Enter or Space opens card → Esc closes
- Card position auto-flips above/below citation based on viewport (no clipping)
- Works inside Trace Inspector's Answer tab AND on the live chat stream

**Differentiator vs Perplexity**: Ours surfaces *why* the source ranked where it did — provenance and trust tier are visible, not just the snippet.

**What's NEW vs already-visible in `[Title — domain](URL)` link text**:
- Title + domain are ALREADY in the inline link. Hover ADDS: snippet excerpt (the actual quoted text from the source), provenance chain (Parallel/Tavily/Serper which provider found it, Cohere/FlashRank which reranker promoted it), rank position (was this top-1 or top-9?), trust tier (tier-1 official / tier-2 press / tier-3 misc), retrieval intent (was this found via PRIMARY query or CONTRADICTION_PROBE?). This is the *audit trail* the rest of the architecture invested in but never surfaced.

**Mobile/touch behavior**:
- No `:hover` on touch devices. Tap citation → card slides up from bottom (≤320px height), persists until outside-tap or Esc. Same data shown.

**Accessibility**:
- Citation gets `role="button" tabindex="0" aria-describedby="hover-card-{doc_id}"`
- Card has `role="tooltip"` on desktop, `role="dialog"` on touch
- Esc closes card and returns focus to citation
- `prefers-reduced-motion` → no slide/fade, instant show/hide
- Keyboard: Tab to citation → Enter or Space opens, Esc closes
- Screen reader announces card content on open

**Performance budget**:
- Card render <16ms (single frame)
- No network call (all data already in turn state)
- Debounce hover ingress at 150ms to avoid card-flashing when sweeping cursor across citations

**Edge case — close-together citations**:
- If multiple citations are adjacent (`[1] [2] [3]`), hover-ingress on each schedules a render but a 100ms grace period after pointer-leave prevents flicker
- Only one card is open at a time; opening a new citation closes the prior

**Interaction with F4 (Devanagari numerals)**: hover card always shows source title/excerpt in the source's original language; the trigger numeral is whatever F4 emits (`[१]` or `[1]`)

**Data sources** (already available, no backend work):
- `doc_map: {doc_N: (title, url, domain)}` — already persisted per turn
- `turn.context_xml_sent` — contains snippet excerpts
- `run_metadata.reranker_used`, `extraction_fallbacks`, `supplementary_sources` — provenance flags
- Trust tier — derived client-side via existing `utils/source_trust.py` API or shipped as part of doc_map

**Implementation**:
- `frontend/components/chat/citation-hover-card.tsx` (new, ~120 lines) — Radix `HoverCard` primitive
- `frontend/lib/markdown.tsx` — augment `RichMarkdown` to wrap `[text](url)` link nodes in `<CitationHoverCard>` when the URL is in the current turn's `doc_map`
- `frontend/lib/types.ts` — extend `Turn` and `TurnDetail` types if needed for provenance fields
- `frontend/components/trace/trace-inspector.tsx` — uses same component on the Answer tab

**Testing**:
- Render an answer with 3 citations → hover each → assert correct snippet renders
- Hover with no `doc_map` entry → card shows minimal fallback (just title + URL)
- Keyboard nav: Tab to citation → Enter → card open → Esc → card closed, focus returns to citation
- Reduce motion: respect `prefers-reduced-motion`

**Acceptance**:
- Demo video can show "hover any citation to see provenance" — this becomes a 5-second wow moment
- All 461 existing tests still pass
- TSC clean

**Effort**: ~2 hrs

---

### F2 — Sarvam Translate (Mayura) Query Expansion for Indic Queries

**User story**: As a Hindi-speaking user, I want my Hindi query to find the best English sources too (English web has 100× more content) — without me having to translate manually — and still get the answer in Hindi.

**Functional requirements**:
- When `detect_language(query)` returns `hi`/`ta`/`bn`/`mr`, AND `SARVAM_API_KEY` is set, AND `SARVAM_TRANSLATE_ENABLED=1` (default):
  - Before running planner, translate the Indic query to English via Sarvam Translate API.
  - Add the English translation to the planner's input as `original_query_en`.
  - Planner emits typed queries in BOTH the original Indic AND English.
  - Search executes both sets; results dedup by URL.
- Synthesis stays on Sarvam (auto-routed for Indic) — answers in the original Indic language.
- Quote-first audit (Phase 1) continues to verify against the original English source text — citations preserved.
- All translation behavior logged in `run_metadata["translate_expansion"]: {enabled: bool, source_query: str, translated_query: str, latency_ms: int}` for the trace inspector.

**Why option 1 (not option 2)**:
- Phase 1 quote-grounding audit verifies quoted text appears in source. Translating sources breaks that — citations would point at Mayura-generated text not originals.
- Sarvam-M is explicitly trained for English-input → Indic-output.
- One translate call per query vs N per snippet — much lower latency + quota.

**Caching strategy** (avoid duplicate translate calls):
- Per-session in-memory LRU keyed on `sha256(text + source_lang)`, capped at 100 entries per process
- Optional persistent cache via existing `utils/search_cache.py` pattern (same TTL semantics) gated on `SARVAM_TRANSLATE_CACHE_ENABLED=1`
- Cache hit logged in `run_metadata["translate_expansion"]["cached"] = true` so trace inspector can show it
- 24h TTL — translations rarely change

**Fallback behavior** (each path tested):
1. `SARVAM_TRANSLATE_ENABLED=0` → translation skipped entirely
2. `SARVAM_API_KEY` missing → translation skipped, log info
3. Mayura returns 429 → retry once with 2s backoff, then skip
4. Mayura returns nonsense (e.g., echo of source, empty string) → detect via length-ratio sanity check (translated > 50% of source length, max 3× source length); reject if outside
5. Mayura timeout (5s) → skip, fall back to Indic-only search
6. Any 5xx → skip, log warning

**Eval-visible value metric** (so we can SHOW translate-expansion improved retrieval):
- New eval column: `translate_expansion_added_sources: int` — count of search-result URLs found ONLY via the translated query (not the original Indic)
- Aggregated in summary: `mean_translate_expansion_added_sources_per_indic_q`
- Displayed in eval report markdown under "Sarvam Translate impact"
- This becomes the concrete proof that Mayura earned its keep

**Planner query budget interaction**:
- Today planner emits ≤4 typed queries
- With translate-expansion: planner sees BOTH original-Indic AND translated-English as input context, but still emits ≤4 typed queries total
- Translate-expansion does NOT double the search call count; planner is asked to allocate the 4-query budget across both languages intelligently
- If planner ignores the English and emits only Indic, no harm — search still runs the original

**Implementation**:
- `utils/sarvam_translate.py` (new, ~80 lines):
  ```python
  async def translate_to_english(text: str, source_lang: str, timeout_s: float = 5.0) -> Optional[str]:
      """Sarvam Translate API. Returns translated text or None on failure. Never raises."""
  ```
- `agent/orchestrator.py` — in the PLANNING phase, if Indic detected, run translate-to-en in parallel with planner call; on success splice into planner input.
- `agent/search.py` — dedup logic already handles cross-language queries via URL hash.
- New env vars in `.env.example`:
  - `SARVAM_TRANSLATE_ENABLED=1` (default on; flip to 0 to disable)
  - `SARVAM_TRANSLATE_MODEL=mayura:v1` (model selector — pin to specific Mayura version)
- Trace Inspector new field: "Translation expansion" panel showing source → translated query when active.

**Testing**:
- Hindi query → translate fires → English query added to planner output → both searched
- Translation API timeout → falls back to Indic-only search, no error
- English query → translation skipped (no Indic detected)
- `SARVAM_TRANSLATE_ENABLED=0` → translation skipped
- Test fixture mock for Mayura response

**Acceptance**:
- Demo: ask "RBI ki latest repo rate kya hai" → trace shows Hindi query translated to "What is the latest RBI repo rate" → both queries searched → answer in Hindi with English-source citations
- New `translate_expansion_used` aggregate in eval summary
- Sarvam Translate visible in `/status` page as a new probe (status: ok/down/not_configured)

**Effort**: ~2.5 hrs (including new provider_health probe + tests)

---

### F3 — Sarvam Usage Spotlight in Trace + Per-Turn Provider Attribution

**User story**: As a Sarvam evaluator, I want to see at a glance which of my turns were handled by Sarvam — both as a session-level summary and per-turn.

**Functional requirements**:
- **Per-turn badge**: Below the answer, render a small attribution row:
  ```
  Synthesized by Sarvam-M · Indic-auto · 936ms
  ```
  Or for English:
  ```
  Synthesized by Gemini 2.5 Flash · 569ms
  ```
  Pulled from `run_metadata.synth_provider_chain[0]` and existing latency data.
- **Session-level Sarvam stat**: In the sidebar or session header, when ≥1 turn used Sarvam, show a small "🇮🇳 Sarvam handled N/M turns" pill.
- **Trace Inspector "Routing" tab** (already exists from Tier A) gets a "Sarvam attribution" callout when applicable — what % of work Sarvam did for this turn.

**Implementation**:
- `frontend/components/chat/provider-attribution.tsx` (new, ~50 lines)
- `frontend/app/page.tsx` — render the attribution row below each answer
- `frontend/components/shell/sidebar.tsx` — compute Sarvam-stat for the active session, render pill if applicable
- Use existing data from `run_metadata.synth_provider_chain` and `run_metadata.planner_provider`

**Testing**:
- English query → "Synthesized by Gemini" badge
- Hindi query → "Synthesized by Sarvam-M" badge
- Session with mix → sidebar pill shows count

**Acceptance**:
- Visible Sarvam attribution wherever provider routing happened
- No new backend work — all data already persisted

**Effort**: ~1 hr

---

### F4 — Indic-Script Citation Numerals

**User story**: As a Hindi reader, I want citation markers in my Hindi answer to feel native — `[१]` instead of `[1]`.

**Functional requirements**:
- After `convert_citations()` rewrites `[doc_N]` → `[Title — domain](URL)`, an additional optional transform converts the leading `[N]` portion of compact-citation format `[N]` → Indic numeral when answer language is Indic.
- Apply ONLY when:
  - Answer language is Indic (Hindi/Tamil/Bengali/Marathi)
  - Citation format used is the compact `[N]` numeric style (not the full `[Title — domain]` link)
- Conversion table:
  - Hindi/Marathi (Devanagari): 0123456789 → ०१२३४५६७८९
  - Tamil: 0123456789 → ௦௧௨௩௪௫௬௭௮௯
  - Bengali: 0123456789 → ০১২৩৪৫৬৭৮৯
- Citations still link to the same URLs (hover card from F1 still works)
- The full-format `[Title — domain](URL)` stays in Latin (URL and domain are ASCII regardless)

**Implementation**:
- `agent/citation_guard.py` — add `localize_citation_numerals(answer: str, lang: str) -> str` helper called after `convert_citations` when lang is Indic
- Pure string transform; tests are deterministic

**Testing**:
- Hindi answer with `[1] [2] [3]` → `[१] [२] [३]`
- Tamil answer → `[௧] [௨] [௩]`
- English answer → unchanged
- Mixed-script answer (rare) → only `[N]` outside of code blocks is converted

**Acceptance**:
- Demo: Hindi answer visibly cites with Devanagari numerals. Small detail, big distinctiveness.

**Effort**: ~1 hr

---

### F5 — Bharat-Friendly Source Trust + sarvam.ai Boost

**User story**: As a researcher on Indian topics, I want India-relevant domains (rbi.org.in, gov.in TLDs, indianexpress.com, hindustantimes.com, etc.) to rank higher when the query is Indic OR India-related.

**Functional requirements**:
- Extend `utils/source_trust.py` `TRUST_TABLE` with India-focused tier-1/2/3 domains:
  - Tier 1 (official): `rbi.org.in`, `sebi.gov.in`, `gov.in` (and `*.gov.in` subdomains), `pib.gov.in`, `mygov.in`, `niti.gov.in`, `mea.gov.in`
  - Tier 2 (high-quality Indian press): `indianexpress.com`, `thehindu.com`, `livemint.com`, `business-standard.com`, `hindustantimes.com`, `scroll.in`, `theprint.in`
  - Tier 3 (research/think tanks): `sarvam.ai`, `research.sarvam.ai`, `iitb.ac.in` and IIT TLDs, `idfcinstitute.org`, `prsindia.org`
- The sarvam.ai entry is a small easter egg ("we trust their research") — visible if you inspect the trust table
- When Indic query OR `expected_source_types=["official"]` from planner, apply a small +0.05 trust boost to all .in TLDs (not enough to override real signal, enough to break ties)

**Implementation**:
- `utils/source_trust.py` — extend TRUST_TABLE dict
- Optional: small `_INDIA_TLD_BOOST` constant gated on `IS_INDIA_BOOST_ENABLED=1` env var (default on for Indic queries)

**Testing**:
- `trust_for("rbi.org.in")` returns expected high tier
- `trust_for("sarvam.ai")` returns tier-3 boost
- Random non-Indian domain unchanged

**Acceptance**:
- Indic queries surface Indian sources more reliably (visible in trace `domain_blocklist_drops` vs `selected` URLs)

**Effort**: ~30 min

---

### F6 — Eval Dashboard "Indic Spotlight" Panel

**User story**: As a Sarvam evaluator, the first thing I want to see on the eval dashboard is how the agent performs on Indic queries vs English.

**Functional requirements**:
- New panel at the top of `/eval/[runAt]/page.tsx` (above the existing metrics grid):
  ```
  ┌─ Indic Spotlight ────────────────────────────────────────┐
  │  Hindi (17 q):  Faithfulness 0.84  Relevance 0.89        │
  │  Tamil (5 q):   Faithfulness 0.81  Relevance 0.87        │
  │  Bengali (5 q): Faithfulness 0.79  Relevance 0.85        │
  │  Marathi (5 q): Faithfulness 0.82  Relevance 0.86        │
  │  English (44 q): Faithfulness 0.87 Relevance 0.91        │
  │                                                          │
  │  Sarvam-routed: 32 of 76 turns (42%)                     │
  │  Cross-language consistency (EN↔HI pairs): 0.83          │
  └──────────────────────────────────────────────────────────┘
  ```
- Pulls from existing per-language aggregates in summary.json + cross_language_consistency
- New stat: count of turns where `synth_provider_chain[0] == "sarvam"` from the persisted JSONL

**Implementation**:
- `frontend/app/eval/[runAt]/page.tsx` — new `IndicSpotlight` component
- Backend `eval/eval_runner.py` `_compute_aggregates()` — add `sarvam_routed_count` to summary JSON (already have per-language slices)

**Testing**:
- Mock summary JSON with all 5 languages → panel renders all rows
- Single-language run → panel hides empty language rows
- Sarvam-routed-count present and accurate

**Acceptance**:
- First-impression panel on eval dashboard speaks directly to Sarvam's evaluation lens

**Effort**: ~1.5 hrs

---

### F7 — Bharat-Themed Suggested Queries

**User story**: As a Sarvam team member opening the demo for the first time, the suggested queries should feel relevant to my work, not generic.

**Functional requirements**:
- Replace the current SUGGESTED list in `frontend/app/page.tsx` with Sarvam-relevant defaults:
  1. "What is India's current repo rate, and how has it changed in the last 12 months?" (keep — already there)
  2. "भारत में मानसून कब आता है और इस वर्ष कैसा रहा?" (keep — already there)
  3. "Compare India's UPI and ONDC adoption — what's the current state of DPI exports?" (new — Bharat policy)
  4. "DPI exports kaha kaha ho rahe hain abhi?" (new — Hinglish, tests our 3-tier lang detection live)
  5. "What's the latest from IndiaAI mission — funding allocated, deliverables shipped?" (new — IndiaAI-aligned)
  6. "Sarvam-30B vs Llama 3.3 70B Indic benchmark comparison" (new — meta, model-comparison)
- Random rotation: pick 4 from the pool per page-load so demo can be re-run with variety

**Implementation**:
- `frontend/app/page.tsx` — replace static SUGGESTED array with a pool + random-sample on mount
- Add a small "↻ Shuffle suggestions" affordance for live demos

**Testing**: Mount → 4 suggestions rendered; click shuffle → different 4

**Acceptance**:
- First-impression alignment with Sarvam's domain interests; demonstrates the Hinglish + Sarvam-routing without forcing the user to think of queries

**Effort**: ~30 min

---

## Phased Implementation Plan

### Phase A — Foundations (parallel-safe, ~3 hrs total)

Dispatch in parallel — disjoint file ownership:

| Agent | Files owned | Duration |
|---|---|---|
| A1 | `frontend/components/chat/citation-hover-card.tsx` (new), `frontend/lib/markdown.tsx`, `frontend/components/trace/trace-inspector.tsx` (F1) | ~2 hrs |
| A2 | `utils/sarvam_translate.py` (new), `utils/provider_health.py` (new probe entry), `agent/orchestrator.py` (translate-expansion plumbing) (F2 backend) | ~2 hrs |
| A3 | `agent/citation_guard.py` (F4), `utils/source_trust.py` (F5) | ~1 hr |

### Phase B — Frontend integration (sequential after A, ~2.5 hrs)

| Step | Files | Duration |
|---|---|---|
| B1 | `frontend/components/chat/provider-attribution.tsx` (new), `frontend/app/page.tsx` (F3) | ~1 hr |
| B2 | `frontend/app/page.tsx` SUGGESTED replacement (F7) | ~30 min |
| B3 | `frontend/app/eval/[runAt]/page.tsx` IndicSpotlight (F6 frontend) | ~1 hr |

### Phase C — Eval aggregate update (~30 min)

| Step | Files | Duration |
|---|---|---|
| C1 | `eval/eval_runner.py` — add `sarvam_routed_count` aggregate (F6 backend) | ~30 min |

### Phase D — Verification + commit (~1.5 hrs)

| Step | What | Duration |
|---|---|---|
| D1 | Run full test suite; verify all 461+ tests pass with new ones added | ~10 min |
| D2 | tsc clean, ESLint not increased, npm build green | ~10 min |
| D3 | Smoke-test in dev: each of 7 features hand-verified at the URL | ~30 min |
| D4 | Commit + push to origin + hf | ~30 min |
| D5 | Vercel auto-deploys via webhook; HF Spaces rebuilds | wall-clock only |

---

## Risks + Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Mayura API quota exhausted during eval | MED | Cache translations per session_id × query; cap retries; env-toggle disable |
| Hover card breaks on mobile (no hover) | LOW | Fall back to tap-to-show; dismiss on outside tap |
| Indic numerals confuse downstream Markdown renderers | LOW | Only transform in user-facing answer text; trace inspector shows raw Latin |
| Sarvam trust-boost over-weights .in domains for non-India queries | LOW | Boost gated on Indic-query OR planner intent — not blanket |
| Backend translate adds 200-500ms latency on Indic queries | LOW | Runs in parallel with planner call; net latency impact ~100ms |

---

## Out of scope (deliberately skipped, document for next cycle)

- **Voice (Saaras STT + Bulbul TTS)** — high impact but ~6 hrs of mic permissions, audio playback, browser quirks. Defer.
- **Research personas** ("Indian banking analyst", "Bharat policy researcher") — over-engineering for 2-day budget.
- **Branching conversations** — Claude-pattern; out of scope.
- **Conversation pinning + shared link** — out of scope.
- **Custom instructions per session** — RuntimeConfig overrides already exposed in /settings; per-session UI is its own feature.
- **Model picker per chat** — env-driven today; per-chat override is its own feature.

---

## Acceptance criteria (overall)

- [ ] All 7 features functionally working end-to-end (manual smoke test in dev)
- [ ] All existing tests (461+) still pass; new tests added for each non-trivial feature
- [ ] TypeScript clean, ESLint not worsened, npm build green
- [ ] Demo video can showcase: hover citation → translate expansion → Sarvam routing badge → Indic numerals → Indic spotlight panel
- [ ] No new Python or npm dependencies
- [ ] Visual additions match existing "research instrument" character (Instrument Serif editorial / Geist sans / JetBrains Mono technical / teal accent / dark default)

---

## Total estimated effort: 7.5 hrs

Phase A (3) + Phase B (2.5) + Phase C (0.5) + Phase D (1.5).

Realistic 1-day push.
