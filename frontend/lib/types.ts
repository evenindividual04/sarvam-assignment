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
}

export interface TurnDetail extends Turn {
  claim_verification_json?: unknown;
  claim_audit?: ClaimAuditRow[];
  contradiction_probes?: ContradictionProbeRow | null;
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
}

export interface EvalQuestion {
  question_id: string;
  category: string;
  question: string;
  faithfulness?: number;
  answer_relevance?: number;
  context_precision?: number;
  citation_integrity?: number;
  claim_precision?: number;
  conflict_adherence?: number;
  failure_class?: string;
  pass?: boolean;
  latency_ms?: number;
}

export interface EvalQuestionDetail extends EvalQuestion {
  turn_id?: string;
  agent_answer: string;
  judge_reasoning?: string;
  context_xml_sent?: string;
  doc_map?: DocMap;
  claim_audit?: ClaimAuditRow[];
  contradiction_probes?: ContradictionProbeRow | null;
}
