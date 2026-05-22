// Runtime-tunable knobs the chat page forwards on every /research POST as
// `overrides`. The backend's RuntimeConfig.from_overrides layers these on top
// of process env, so eval_runner.py (which doesn't send overrides) stays
// reproducible from env vars alone.
//
// All state lives in localStorage — no backend persistence. This sidesteps
// the source-of-truth problem (DB shadow vs env) entirely.

export interface RuntimeOverrides {
  // V3.8: tri-state retrieval mode supersedes the boolean HYBRID_RETRIEVAL.
  // Backend still honors HYBRID_RETRIEVAL=0|1 as a deprecated alias.
  RETRIEVAL_MODE?: "auto" | "hybrid" | "lexical";
  FAILURE_POLICY_MAX_HOPS?: number;
  CONTEXT_SELECTION_STRATEGY?: "heuristic" | "mmr";
  // Phase 5b: comma-separated extra blocked domains. Default behavior blocks
  // reddit/twitter/x/tiktok/quora/instagram/facebook/pinterest; pass "none"
  // to disable the default blocklist entirely.
  RETRIEVAL_DOMAIN_BLOCKLIST?: string;
}

const KEY = "dra:overrides";

export function loadOverrides(): RuntimeOverrides {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as RuntimeOverrides;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export function saveOverrides(overrides: RuntimeOverrides): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(overrides));
  } catch {
    /* quota / private mode — silently no-op */
  }
}

export function clearOverrides(): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(KEY);
}

export interface KnobDef {
  key: keyof RuntimeOverrides;
  label: string;
  type: "boolean" | "integer" | "enum" | "string" | "float";
  min?: number;
  max?: number;
  choices?: string[];
  available?: boolean;
  note?: string | null;
  // V3.8: for RETRIEVAL_MODE, the backend reports what *would* happen under
  // each user choice given the current capability probe. Lets the UI label
  // "hybrid → unavailable on this host" without a separate API round trip.
  effective_per_choice?: Record<string, string>;
}

export interface DefaultsResponse {
  effective: {
    retrieval_mode: string;
    max_hops: number;
    selection_strategy: string;
    mmr_lambda?: number;
    approval_required?: boolean;
    domain_blocklist?: string[];
  };
  knobs: KnobDef[];
}
