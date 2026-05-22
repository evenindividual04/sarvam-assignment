import type {
  EvalQuestion,
  EvalQuestionDetail,
  EvalRun,
  EvalSummary,
  SessionListItem,
  StreamLabelsResponse,
  Turn,
  TurnDetail,
} from "./types";
import type { DefaultsResponse } from "./settings";

export const BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:7860";

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BACKEND}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text || res.statusText);
  }
  return (await res.json()) as T;
}

export async function getHealth(): Promise<{ status: string; version?: string }> {
  return getJson("/health");
}

export interface ProviderProbe {
  name: string;
  role: string;
  status: "ok" | "degraded" | "down" | "missing_key" | "not_configured";
  latency_ms: number | null;
  detail: string;
}

export interface ProviderUsageRow {
  provider: string;
  usage_date: string;
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  request_limit?: number | null;
  token_limit?: number | null;
}

export interface ProviderHealth {
  checked_at: number;
  overall: "ok" | "degraded" | "down";
  providers: ProviderProbe[];
  cache_ttl_s: number;
  usage_today?: ProviderUsageRow[];
}

export async function getProviderHealth(force = false): Promise<ProviderHealth> {
  return getJson(`/health/providers${force ? "?force=1" : ""}`);
}

export async function getSettingsDefaults(): Promise<DefaultsResponse> {
  return getJson("/settings/defaults");
}

export async function listSessions(): Promise<SessionListItem[]> {
  return getJson("/sessions");
}

export async function getSessionHistory(sessionId: string): Promise<Turn[]> {
  return getJson(`/sessions/${encodeURIComponent(sessionId)}/history`);
}

export async function getTurnDetail(
  sessionId: string,
  turnId: string,
): Promise<TurnDetail> {
  return getJson(
    `/sessions/${encodeURIComponent(sessionId)}/turns/${encodeURIComponent(turnId)}`,
  );
}

export async function listEvalRuns(): Promise<EvalRun[]> {
  return getJson("/eval/runs");
}

export async function getRunSummary(runAt: string): Promise<EvalSummary> {
  return getJson(`/eval/runs/${encodeURIComponent(runAt)}/summary`);
}

export async function getRunQuestions(runAt: string): Promise<EvalQuestion[]> {
  return getJson(`/eval/runs/${encodeURIComponent(runAt)}/questions`);
}

export async function getQuestionDetail(
  runAt: string,
  questionId: string,
): Promise<EvalQuestionDetail> {
  return getJson(
    `/eval/runs/${encodeURIComponent(runAt)}/questions/${encodeURIComponent(questionId)}`,
  );
}

// Phase 1.25: fetch the canonical phase label dictionary so the pipeline
// component doesn't duplicate the Python STREAM_LABELS constant. Cached on
// the module so a single network round-trip per page load suffices.
// Promise-level cache so multiple concurrent mounts of StreamProgress
// (one per ChatTurn) share a single in-flight fetch instead of each one
// hitting `/stream/labels` independently. On failure the cache resets so
// a later mount can retry.
let _labelsPromise: Promise<StreamLabelsResponse> | null = null;
export function getStreamLabels(): Promise<StreamLabelsResponse> {
  if (!_labelsPromise) {
    _labelsPromise = getJson<StreamLabelsResponse>("/stream/labels").catch(
      (e) => {
        _labelsPromise = null;
        throw e;
      },
    );
  }
  return _labelsPromise;
}

// S1 fix: session_id is required by the backend to prove the caller owns the
// turn. Without it the backend returns 404 (anti-enumeration).
export async function cancelResearch(turnId: string, sessionId: string): Promise<void> {
  await fetch(`${BACKEND}/research/cancel/${encodeURIComponent(turnId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  }).catch(() => {
    /* fire-and-forget */
  });
}

// Phase 2: resolve a paused plan-approval gate. Pass null to accept the plan
// as-is, or an array of edited sub_queries (length-capped at 6 server-side).
// S1 fix: sessionId required for ownership verification.
export async function approveResearchPlan(
  turnId: string,
  sessionId: string,
  editedSubQueries: string[] | null,
): Promise<{ approved: boolean; edited: boolean; sub_queries: string[] | null }> {
  const res = await fetch(
    `${BACKEND}/research/approve/${encodeURIComponent(turnId)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sub_queries: editedSubQueries, session_id: sessionId }),
    },
  );
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text || res.statusText);
  }
  return (await res.json()) as {
    approved: boolean;
    edited: boolean;
    sub_queries: string[] | null;
  };
}

export { ApiError };
