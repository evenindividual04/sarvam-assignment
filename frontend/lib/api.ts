import type {
  EvalQuestion,
  EvalQuestionDetail,
  EvalRun,
  EvalSummary,
  SessionListItem,
  Turn,
  TurnDetail,
} from "./types";

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

export async function cancelResearch(turnId: string): Promise<void> {
  await fetch(`${BACKEND}/research/cancel/${encodeURIComponent(turnId)}`, {
    method: "POST",
  }).catch(() => {
    /* fire-and-forget */
  });
}

export { ApiError };
