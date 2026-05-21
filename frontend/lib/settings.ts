// Runtime-tunable knobs the chat page forwards on every /research POST as
// `overrides`. The backend's RuntimeConfig.from_overrides layers these on top
// of process env, so eval_runner.py (which doesn't send overrides) stays
// reproducible from env vars alone.
//
// All state lives in localStorage — no backend persistence. This sidesteps
// the source-of-truth problem (DB shadow vs env) entirely.

export interface RuntimeOverrides {
  HYBRID_RETRIEVAL?: "0" | "1";
  FAILURE_POLICY_MAX_HOPS?: number;
  CONTEXT_SELECTION_STRATEGY?: "heuristic" | "mmr";
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
  type: "boolean" | "integer" | "enum";
  min?: number;
  max?: number;
  choices?: string[];
  available?: boolean;
  note?: string | null;
}

export interface DefaultsResponse {
  effective: {
    hybrid_retrieval: boolean;
    max_hops: number;
    selection_strategy: string;
  };
  knobs: KnobDef[];
}
