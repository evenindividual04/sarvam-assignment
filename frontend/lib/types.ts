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
  // Phase 2: human-in-the-loop plan-approval gate. Emitted between PLANNING
  // and SEARCHING when ChatRequest.approval_required=true. The orchestrator
  // pauses until POST /research/approve/{turn_id} or /cancel fires.
  | "plan_approval";

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
  run_metadata: Record<string, unknown>;
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
  run_metadata_json?: Record<string, unknown>;
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

export interface Contradiction {
  topic?: string;
  position_a: string;
  position_a_doc_ids: string[];
  position_b: string;
  position_b_doc_ids: string[];
  is_temporal_evolution?: boolean;
}

export interface ContradictionProbeRow {
  turn_id: string;
  has_conflict: boolean;
  is_temporal_evolution?: boolean;
  confidence: number;
  contradictions?: Contradiction[];
  skip_reason?: "timeout" | "parse_fail" | "lt_2_chunks" | "breaker_open" | string;
  raw_response?: string;
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
}
