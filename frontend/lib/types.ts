// Backend-facing TypeScript types. Mirrors the FastAPI contract.

export type StreamStep =
  | "planning"
  | "searching"
  | "fetching"
  | "selecting"
  | "probing"
  | "generating"
  | "verifying"
  | "done"
  | "error";

export interface ExecutionEvent<T = unknown> {
  step: StreamStep;
  label: string;
  data?: T;
  // Phase 1.25: typed-event discriminator (AG-UI / Vercel AI SDK 5 style).
  // When set, the SSE layer carries this in the `event:` field. Frontend
  // dispatches per `type` in a renderer registry. Absent on legacy events.
  type?: TypedEventName;
}

// Phase 1.25: typed event taxonomy. The five user-facing phase names
// (planning/searching/fetching/selecting/generating) still appear via
// `phase_started.label` — assignment compliance is preserved.
export type TypedEventName =
  | "run_started"
  | "phase_started"
  | "phase_progress"
  | "phase_finished"
  | "search_query"
  | "source_found"
  | "source_fetched"
  | "context_selected"
  | "conflict_detected"
  | "answer_delta"
  | "citation_resolved"
  | "run_finished"
  | "run_error"
  // Phase 1.5: structured uncertainty signal. Three kinds:
  //   - "missing"  — retrieval returned no usable sources
  //   - "weak"     — low planner confidence OR thin context vs budget
  //   - "conflict" — sources disagree on a key fact
  | "uncertainty"
  // Phase 1.875: plan-level signals
  //   - "clarification_offered" — planner flagged ambiguity
  //   - "evidence_gap"          — a planner sub-query yielded no usable evidence
  | "clarification_offered"
  | "evidence_gap"
  // Per-turn forensic events.
  //   - "hop_evidence"        — newly-grounded tokens + still-open criteria per hop
  //   - "source_contribution" — per-URL token-share of final context + citation count
  //   - "source_role"         — LLM-classified role per URL (primary / analysis / …)
  //   - "terminator"          — explicit hop-loop stop reason
  | "hop_evidence"
  | "source_contribution"
  | "source_role"
  | "terminator"
  // Phase 2: human-in-the-loop plan-approval gate. Emitted between PLANNING
  // and SEARCHING when ChatRequest.approval_required=true. The orchestrator
  // pauses until POST /research/approve/{turn_id} or /cancel fires.
  | "plan_approval"
  // B3: retrieval-grounded reasoning. Emitted twice per hop. The `intent`
  // payload carries planner-derived rationales (NOT model-streamed CoT); the
  // `observation` payload carries title/domain/score from selected chunks.
  | "reasoning";

export type UncertaintyKind = "weak" | "missing" | "conflict";

export type StreamEvent =
  | { type: "run_started"; turn_id: string; session_id: string; query: string }
  | {
      type: "phase_started";
      name: string;
      label: string;
      idx: number;
      total: number;
      hop?: number;
    }
  | {
      type: "phase_progress";
      name: string;
      current: number;
      total: number;
      failed?: number;
      label?: string;
    }
  | {
      type: "phase_finished";
      name: string;
      duration_ms: number;
      [extra: string]: unknown;
    }
  | { type: "search_query"; query: string; provider: string; hop: number }
  | {
      type: "source_found";
      url: string;
      title: string;
      domain: string;
      query: string;
    }
  | {
      type: "source_fetched";
      url: string;
      status: "ok" | "error";
      latency_ms: number;
      bytes?: number;
      error?: string;
    }
  | {
      type: "context_selected";
      url: string;
      score: number;
      snippet_preview: string;
      rank: number;
    }
  | {
      type: "conflict_detected";
      claim: string;
      position_a: string;
      position_b: string;
    }
  | { type: "answer_delta"; text: string }
  | { type: "citation_resolved"; marker: string; url: string; title: string }
  | {
      type: "run_finished";
      usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
      total_latency_ms: number;
      cost_usd: number;
    }
  | {
      type: "run_error";
      phase: string;
      message: string;
      recoverable: boolean;
    }
  | {
      type: "uncertainty";
      kind: UncertaintyKind;
      reason?: string;
      follow_ups: string[];
    }
  | {
      type: "clarification_offered";
      kind?: "ambiguity";
      original_query: string;
      possible_interpretations: string[];
      /** Vagueness-gated clarifier: one user-facing question. May be absent
       *  on legacy events that pre-date the vagueness module. */
      clarifying_question?: string;
    }
  | {
      type: "evidence_gap";
      query: string;
      intent: string;
      reason: "no_results" | "all_filtered";
    }
  | {
      type: "plan_approval";
      turn_id: string;
      planner_output: PlannerOutput;
      sub_queries: string[];
    }
  | {
      // B3: retrieval-grounded reasoning. `phase` discriminates the two
      // emissions per hop. All fields originate from structured data
      // (planner JSON or selected chunk metadata) — no model-streamed CoT.
      type: "reasoning";
      hop: number;
      phase: "intent" | "observation";
      queries?: { text: string; intent: string; rationale: string | null }[];
      observation?: { title: string; domain: string; url: string; score: number }[];
    }
  // Forensic ledger: what the hop *mechanically* established and what remains.
  // `grounded` items come from the planner's success_criteria intersected with
  // claim_verifier+numeric_audit outputs — NOT model-generated prose. `open`
  // items are the planner criteria that did not get grounded this hop.
  | {
      type: "hop_evidence";
      hop: number;
      grounded: {
        token: string;
        kind: "entity" | "number" | "criterion";
        doc_id?: string;
        url?: string;
        quote?: string;
      }[];
      open: {
        criterion: string;
        reason: "no_evidence" | "partial" | "conflicting";
      }[];
    }
  // Token-share of final context per URL — replaces the chunk-count ratio.
  // `tokens` is the tiktoken count, `share` is tokens / total_context_tokens.
  | {
      type: "source_contribution";
      contributions: {
        url: string;
        domain: string;
        title: string;
        tokens: number;
        share: number;
        citations: number;
      }[];
      total_tokens: number;
    }
  // LLM-classified source role over the chunk pool. One classifier call,
  // results applied per URL.
  | {
      type: "source_role";
      roles: {
        url: string;
        role:
          | "primary_source"
          | "secondary_analysis"
          | "statistical"
          | "news_event"
          | "official"
          | "encyclopedic"
          | "contradicting"
          | "unclassified";
        confidence: number;
      }[];
    }
  // Why the hop loop stopped. `reason` mirrors `run_metadata.terminator_fired`.
  | {
      type: "terminator";
      reason:
        | "MAX_HOPS_REACHED"
        | "EVIDENCE_SUFFICIENT"
        | "MARGINAL_GAIN_LOW"
        | "NO_NEW_QUERIES"
        | "BUDGET_EXHAUSTED"
        | "CRITERIA_SATISFIED";
      hop: number;
      detail?: string;
    };

// Phase 1.875: planner-output surfaced to UI (see PlanCard).
export type PlannerTimeSensitivity = "live" | "recent" | "static";
export type PlannerDifficulty = "easy" | "medium" | "hard";
export type PlannerConfidence = "low" | "medium" | "high";
export type PlannerSourceType =
  | "news"
  | "academic"
  | "official"
  | "wiki"
  | "forum";

export interface PlannerTypedQuery {
  text: string;
  intent: string;
  rationale?: string | null;
}

export interface PlannerOutput {
  strategy: string;
  confidence?: PlannerConfidence;
  time_sensitivity?: PlannerTimeSensitivity;
  expected_source_types?: PlannerSourceType[];
  difficulty?: PlannerDifficulty;
  ambiguity_flag?: boolean;
  success_criteria?: string[];
  queries: PlannerTypedQuery[];
}

export type PlanQueryPhase = "pending" | "searching" | "fetching" | "done";

export interface EvidenceGap {
  query: string;
  intent: string;
  reason: "no_results" | "all_filtered";
}

// Forensic payload shapes used by `run_metadata`.
// Mirror the backend constants emitted by the orchestrator.
export interface RunMetadataHopEvidence {
  hop: number;
  grounded: Array<{
    token: string;
    kind: "entity" | "number" | "criterion";
    doc_id?: string;
    url?: string;
    quote?: string;
  }>;
  open: Array<{
    criterion: string;
    reason: "no_evidence" | "partial" | "conflicting";
  }>;
}

export interface RunMetadataSourceContribution {
  url: string;
  domain: string;
  title: string;
  tokens: number;
  share: number;
  citations: number;
}

export interface UnreachablePage {
  url: string;
  status?: string;
  reason?: string;
}

/**
 * Structured shape for `Turn.run_metadata_json` (a.k.a. `run_metadata` on
 * the DoneEventData payload). All fields are optional — older turns and
 * forward-compatible additions are preserved via the index signature.
 */
export interface RunMetadata {
  retrieval_mode?: {
    requested?: string;
    effective?: string;
    reason?: string | null;
  };
  uncertainty_kind?: "none" | "weak" | "missing" | "conflict" | null;
  follow_up_queries?: string[];
  evidence_gaps_reason?: string | null;
  budget_distribution?: {
    system?: number;
    history?: number;
    web_context?: number;
    output_reserved?: number;
  };
  context_fallbacks?: string[];
  numeric_grounding_ratio?: number | null;
  criteria_coverage?: boolean[];
  terminator_fired?: string | null;
  evidence_gaps?: Array<{ query: string; intent: string; reason: string }>;
  hop_evidence?: RunMetadataHopEvidence[];
  source_contributions?: RunMetadataSourceContribution[];
  total_context_tokens?: number;
  source_roles?: Record<string, { role: string; confidence: number }>;
  terminator_hop?: number | null;
  terminator_detail?: string | null;
  unreachable_pages?: UnreachablePage[];
  cite_quote_map?: Record<string, string>;
  next_step_suggestions?: string[];
  refinement_triggered?: boolean;
  refinement_count?: number;
  refinement_reason?: string;
  conflict_table_missing?: boolean;
  terminator_source?: "stop_rag" | "deterministic";
  domain_blocklist_drops?: string[];
  extraction_fallbacks?: Record<string, number>;
  reasoner_used?: string;
  [k: string]: unknown;
}

export interface NumericAuditEntry {
  token: string;
  kind: "number" | "year" | "date";
  position: number;
  grounded: boolean;
  matched_doc_id?: string | null;
}

export interface StreamLabelsResponse {
  labels: Record<string, string>;
  order: string[];
}

export type DocMap = Record<string, [string, string, string]>; // doc_id -> [title, url, domain]

export interface DoneEventData {
  turn_id: string;
  answer: string;
  internal_answer?: string;
  citation_integrity_score: number;
  claim_precision_score: number;
  urls: string[];
  doc_map: DocMap;
  prompt_tokens: number;
  completion_tokens: number;
  latency_ms: number;
  planning_ms: number;
  search_ms: number;
  fetch_ms: number;
  select_ms: number;
  probe_ms: number;
  synthesize_ms: number;
  run_metadata: RunMetadata;
  context_xml: string;
  selection_strategy: string;
  planning_strategy: string;
}

export interface SessionListItem {
  session_id: string;
  updated_at: string;
  turn_count: number;
  /**
   * Optional human-friendly title (e.g. derived from the first user query).
   * The backend may populate this; the sidebar falls back to a session-id
   * slug when absent. See `lib/sessions.ts::sessionDisplayTitle`.
   */
  title?: string | null;
  /**
   * Verbatim text of the session's first turn query. Used as the display
   * title fallback when no explicit `title` has been stamped. Capped at
   * 200 chars on the backend; frontend truncates further for layout.
   */
  first_query?: string | null;
  /** Optional URL count rolled up across all turns; rendered as metadata. */
  source_count?: number | null;
}

export interface Turn {
  turn_id: string;
  session_id: string;
  query: string;
  response: string;
  created_at: string;
  search_queries?: string[];
  urls_opened?: string[];
  doc_map?: DocMap;
  context_xml_sent?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  latency_ms?: number;
  citation_integrity_score?: number;
  claim_precision_score?: number;
  state_trace?: string[];
  selection_strategy?: string;
  planning_strategy?: string;
  planning_ms?: number;
  search_ms?: number;
  fetch_ms?: number;
  select_ms?: number;
  probe_ms?: number;
  synthesize_ms?: number;
  cost_usd?: number;
}

export interface ContextSnippetRow {
  doc_id: string;
  url: string;
  title: string;
  domain: string;
  snippet: string;
  bm25_score?: number;
  recency_score?: number;
  final_score?: number;
  trust_score?: number;
  trust_tier?: string;
  provider_relevance?: number | null;
  provider_relevance_source?: string | null;
}

export interface TurnDetail extends Turn {
  claim_verification_json?: unknown;
  run_metadata_json?: RunMetadata;
  claim_audit?: ClaimAuditRow[];
  contradiction_probes?: ContradictionProbeRow | null;
  context_snippets?: ContextSnippetRow[];
}

export interface ClaimAuditRow {
  claim_id?: string;
  turn_id: string;
  claim_text: string;
  cited_doc_ids: string[];
  method: "deterministic" | "llm" | "skip" | string;
  score: number;
  status: "supported" | "unsupported" | "ambiguous_resolved" | string;
  reasoning?: string;
}

/**
 * DRAGged-into-Conflict taxonomy (Cattan et al., arXiv:2506.08500).
 * Surfaced from the backend `contradiction_probes.dominant_kind` column
 * and per-contradiction `kind` field. "none" means no real conflict.
 */
export type ConflictKind = "self" | "pair" | "conditional" | "none";

export interface Contradiction {
  topic?: string;
  position_a: string;
  position_a_doc_ids: string[];
  position_b: string;
  position_b_doc_ids: string[];
  is_temporal_evolution?: boolean;
  // P3: DRAGged-into-Conflict taxonomy (arXiv:2506.08500).
  kind?: ConflictKind;
  qualifier?: string | null;
}

export interface ContradictionProbeRow {
  turn_id: string;
  has_conflict: boolean;
  is_temporal_evolution?: boolean;
  confidence: number;
  contradictions?: Contradiction[];
  skip_reason?: "timeout" | "parse_fail" | "lt_2_chunks" | "breaker_open" | string;
  raw_response?: string;
  // P3: overall verdict across all contradictions.
  dominant_kind?: ConflictKind;
}

export interface EvalRun {
  run_at: string;
  n_questions: number;
  pass_rate: number;
  retrieval_mode?: "bm25" | "hybrid" | string;
  avg_faithfulness: number;
  avg_relevance: number;
  avg_context_precision: number;
  avg_citation_integrity: number;
  avg_claim_precision: number;
}

export interface EvalSummaryCategoryRow {
  category: string;
  faithfulness: number;
  answer_relevance: number;
  context_precision: number;
  citation_integrity: number;
  claim_precision: number;
  conflict_adherence?: number;
}

export interface EvalSummary {
  run_at: string;
  pass_rate: number;
  avg_faithfulness: number;
  avg_relevance: number;
  avg_context_precision: number;
  avg_citation_integrity: number;
  avg_claim_precision: number;
  avg_conflict_adherence?: number;
  by_category: EvalSummaryCategoryRow[];
  failure_class_distribution: Record<string, number>;
  calibration?: {
    buckets: Array<{
      confidence: "low" | "medium" | "high" | string;
      mean_faithfulness: number | null;
      mean_claim_precision: number | null;
      n: number;
    }>;
    correlation: number | null;
  };
  run_summary?: {
    total_cost_usd?: number;
    p50_latency_ms?: number;
    p95_latency_ms?: number;
    calibration_correlation?: number | null;
  };
  // Tier A (Phase 1+) — per-turn quality summary across the run. Optional
  // because older runs (pre-migration) don't have these columns; UI renders
  // "—" when missing.
  per_turn_quality?: {
    mean_quote_grounding_ratio: number | null;
    mean_numeric_grounding_ratio: number | null;
    mean_criteria_coverage_ratio: number | null;
  };
  cross_language?: {
    rows: CrossLanguageRow[];
    flagged: CrossLanguageRow[];
    mean_jaccard: number | null;
  } | null;
}

export interface CrossLanguageRow {
  concept_id: string;
  en_question_id: string;
  hi_question_id: string;
  jaccard_score: number;
  /** SQLite stores 0/1 — kept as numeric literal type, truthy-checked at use sites. */
  flagged_inconsistent: 0 | 1;
  reasoning?: string | null;
}

export interface EvalQuestion {
  question_id: string;
  category: string;
  question: string;
  language?: string;
  faithfulness?: number;
  answer_relevance?: number;
  context_precision?: number;
  citation_integrity?: number;
  claim_precision?: number;
  conflict_adherence?: number;
  factual_accuracy_score?: number;
  failure_class?: string;
  pass?: boolean;
  latency_ms?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  cost_usd?: number;
}

export interface EvalQuestionDetail extends EvalQuestion {
  turn_id?: string;
  agent_answer: string;
  judge_reasoning?: string;
  context_xml_sent?: string;
  doc_map?: DocMap;
  claim_audit?: ClaimAuditRow[];
  contradiction_probes?: ContradictionProbeRow | null;
  // Tier A (Phase 1+) — promoted per-row metrics and routing decisions.
  // Optional because older rows (pre-migration) don't have these columns.
  quote_grounding_ratio?: number | null;
  numeric_grounding_ratio?: number | null;
  criteria_coverage_ratio?: number | null;
  terminator_fired?: string | null;
  planner_provider?: string | null;
  reranker_used?: string | null;
  language_method?: string | null;
  // Refactor #3: terminator-policy trace. `terminator_source` is which gate
  // emitted the primary terminator ("stop_rag" or "deterministic").
  // `terminator_history` is the ordered list of gate decisions across hops
  // (1 entry per hop that terminated). `stop_rag_decisions` is the per-hop
  // log of the LLM gate, including degraded (continue-by-default) entries.
  terminator_source?: "stop_rag" | "deterministic" | null;
  terminator_history?: Array<{
    source: "stop_rag" | "deterministic";
    reason: string;
    hop: number;
  }>;
  stop_rag_decisions?: Array<{
    hop: number;
    useful: boolean | null;
    confidence: number | null;
    reason: string;
    degraded_reason: string | null;
  }>;
}
