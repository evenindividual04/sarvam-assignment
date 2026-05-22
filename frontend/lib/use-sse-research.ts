"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { approveResearchPlan, BACKEND, cancelResearch } from "./api";
import { loadOverrides } from "./settings";
import type {
  DoneEventData,
  EvidenceGap,
  ExecutionEvent,
  PlanQueryPhase,
  PlannerOutput,
  TypedEventName,
  UncertaintyKind,
} from "./types";

// Phase 2: plan-approval gate state surfaced to the chat UI. When non-null,
// the orchestrator is paused awaiting POST /research/approve|cancel.
export interface PlanApprovalPending {
  turnId: string;
  plannerOutput: PlannerOutput;
  subQueries: string[];
}

export interface UncertaintySignal {
  kind: UncertaintyKind;
  reason?: string;
  follow_ups: string[];
}

export type SseStatus = "idle" | "streaming" | "done" | "cancelled" | "error";

export interface UseSseResearchReturn {
  events: ExecutionEvent[];
  status: SseStatus;
  finalData: DoneEventData | null;
  error: string | null;
  turnId: string | null;
  currentText: string;
  // Phase 1.5: latest structured uncertainty signal for the current turn.
  uncertainty: UncertaintySignal | null;
  // Phase 1.875: planner-level state surfaced live for the PlanCard.
  plan: PlannerOutput | null;
  phaseProgress: Record<string, PlanQueryPhase>;
  evidenceGaps: EvidenceGap[];
  // Phase 2: when set, the chat UI must render <PlanApprovalPanel/> and pause
  // any further UI updates for the turn until approve/cancel resolves.
  approvalPending: PlanApprovalPending | null;
  start: (
    query: string,
    sessionId: string,
    options?: { approvalRequired?: boolean },
  ) => Promise<void>;
  approvePlan: (editedSubQueries: string[] | null) => Promise<void>;
  cancel: () => void;
  reset: () => void;
}

// Parses SSE `data:` frames out of a streaming POST response body.
export function useSseResearch(): UseSseResearchReturn {
  const [events, setEvents] = useState<ExecutionEvent[]>([]);
  const [status, setStatus] = useState<SseStatus>("idle");
  const [finalData, setFinalData] = useState<DoneEventData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [turnId, setTurnId] = useState<string | null>(null);
  const [currentText, setCurrentText] = useState<string>("");
  const [uncertainty, setUncertainty] = useState<UncertaintySignal | null>(null);
  const [plan, setPlan] = useState<PlannerOutput | null>(null);
  const [phaseProgress, setPhaseProgress] = useState<
    Record<string, PlanQueryPhase>
  >({});
  const [evidenceGaps, setEvidenceGaps] = useState<EvidenceGap[]>([]);
  const [approvalPending, setApprovalPending] =
    useState<PlanApprovalPending | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const turnIdRef = useRef<string | null>(null);
  // S1 fix: keep the active session_id so cancel/approve can pass it to the
  // backend for ownership verification. Without it the server returns 404.
  const sessionIdRef = useRef<string | null>(null);
  // Phase 1.875: shadow ref so we can mutate per-query progress immutably
  // without losing concurrent updates from rapid SSE bursts.
  const phaseProgressRef = useRef<Record<string, PlanQueryPhase>>({});
  // url → query text mapping so source_fetched (URL only) can mark the right
  // plan step as "fetching".
  const urlToQueryRef = useRef<Record<string, string>>({});

  const reset = useCallback(() => {
    setEvents([]);
    setStatus("idle");
    setFinalData(null);
    setError(null);
    setTurnId(null);
    setCurrentText("");
    setUncertainty(null);
    setPlan(null);
    setPhaseProgress({});
    setEvidenceGaps([]);
    setApprovalPending(null);
    phaseProgressRef.current = {};
    urlToQueryRef.current = {};
    turnIdRef.current = null;
  }, []);

  const cancel = useCallback(() => {
    const tid = turnIdRef.current;
    const sid = sessionIdRef.current;
    if (tid && sid) void cancelResearch(tid, sid);
    abortRef.current?.abort();
    setApprovalPending(null);
  }, []);

  // On hook unmount (e.g. navigating to /sessions or /eval mid-stream),
  // abort the in-flight fetch so we don't leak a detached reader/decoder.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const start = useCallback(
    async (
      query: string,
      sessionId: string,
      options?: { approvalRequired?: boolean },
    ) => {
    reset();
    setStatus("streaming");
    sessionIdRef.current = sessionId;
    const ac = new AbortController();
    abortRef.current = ac;

    try {
      // Snapshot overrides at request time so a settings flip mid-stream
      // doesn't retroactively change what this turn ran under.
      const overrides = loadOverrides();
      const hasOverrides = Object.keys(overrides).length > 0;
      const approvalRequired = options?.approvalRequired === true;
      const res = await fetch(`${BACKEND}/research`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          session_id: sessionId,
          ...(hasOverrides ? { overrides } : {}),
          ...(approvalRequired ? { approval_required: true } : {}),
        }),
        signal: ac.signal,
      });
      if (!res.ok || !res.body) {
        throw new Error(`Server returned ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          // Phase 1.25: SSE frames may include `id:`, `event:`, and `data:`
          // lines (or just `: ping` heartbeat comments). Parse them all so
          // the typed-event discriminator is preserved.
          let dataLine: string | null = null;
          let eventLine: string | null = null;
          for (const l of part.split("\n")) {
            if (l.startsWith("data: ")) dataLine = l;
            else if (l.startsWith("event: ")) eventLine = l;
          }
          if (!dataLine) continue;
          const raw = dataLine.slice(6);
          let ev: ExecutionEvent;
          try {
            ev = JSON.parse(raw) as ExecutionEvent;
          } catch {
            continue;
          }
          if (eventLine) {
            ev = { ...ev, type: eventLine.slice(7).trim() as TypedEventName };
          }

          // Capture turn_id from the first planning event.
          if (ev.step === "planning") {
            const d = ev.data as { turn_id?: string } | undefined;
            if (d?.turn_id) {
              turnIdRef.current = d.turn_id;
              setTurnId(d.turn_id);
            }
          }

          if (ev.step === "generating" && typeof ev.data === "string") {
            setCurrentText((prev) => prev + ev.data);
          }

          // Phase 1.875: capture planner output from phase_finished:planning
          // so the PlanCard can render live.
          if (
            ev.type === "phase_finished" &&
            ev.data &&
            typeof ev.data === "object"
          ) {
            const d = ev.data as {
              name?: string;
              planner_output?: PlannerOutput;
            };
            if (d.name === "planning" && d.planner_output) {
              setPlan(d.planner_output);
              const initProgress: Record<string, PlanQueryPhase> = {};
              for (const q of d.planner_output.queries ?? []) {
                initProgress[q.text] = "pending";
              }
              phaseProgressRef.current = initProgress;
              setPhaseProgress(initProgress);
            }
            // When fetching finishes, mark every known plan query as "done"
            // (we don't have per-query fetch boundaries — phase is global).
            if (d.name === "fetching") {
              const next: Record<string, PlanQueryPhase> = {
                ...phaseProgressRef.current,
              };
              for (const k of Object.keys(next)) {
                if (next[k] !== "done") next[k] = "done";
              }
              phaseProgressRef.current = next;
              setPhaseProgress(next);
            }
          }

          // Phase 1.875: per-query progress from typed events.
          if (
            ev.type === "source_found" &&
            ev.data &&
            typeof ev.data === "object"
          ) {
            const d = ev.data as { url?: string; query?: string };
            if (d.query) {
              const cur = phaseProgressRef.current[d.query];
              if (cur === "pending" || cur === undefined) {
                const next: Record<string, PlanQueryPhase> = {
                  ...phaseProgressRef.current,
                  [d.query]: "searching",
                };
                phaseProgressRef.current = next;
                setPhaseProgress(next);
              }
              if (d.url) {
                urlToQueryRef.current = {
                  ...urlToQueryRef.current,
                  [d.url]: d.query,
                };
              }
            }
          }
          if (
            ev.type === "source_fetched" &&
            ev.data &&
            typeof ev.data === "object"
          ) {
            const d = ev.data as { url?: string };
            const q = d.url ? urlToQueryRef.current[d.url] : undefined;
            if (q && phaseProgressRef.current[q] !== "done") {
              const next: Record<string, PlanQueryPhase> = {
                ...phaseProgressRef.current,
                [q]: "fetching",
              };
              phaseProgressRef.current = next;
              setPhaseProgress(next);
            }
          }

          // Phase 1.875: aggregate evidence-gap events.
          if (
            ev.type === "evidence_gap" &&
            ev.data &&
            typeof ev.data === "object"
          ) {
            const d = ev.data as EvidenceGap;
            if (d.query && d.intent && d.reason) {
              setEvidenceGaps((prev) => {
                if (prev.some((g) => g.query === d.query && g.reason === d.reason)) {
                  return prev;
                }
                return [...prev, d];
              });
            }
          }

          // Phase 2: plan-approval gate — render PlanApprovalPanel and hold
          // further UI updates until approve/cancel resolves.
          if (
            ev.type === "plan_approval" &&
            ev.data &&
            typeof ev.data === "object"
          ) {
            const d = ev.data as {
              turn_id?: string;
              planner_output?: PlannerOutput;
              sub_queries?: string[];
            };
            if (d.turn_id && d.planner_output) {
              setApprovalPending({
                turnId: d.turn_id,
                plannerOutput: d.planner_output,
                subQueries: Array.isArray(d.sub_queries)
                  ? d.sub_queries
                  : d.planner_output.queries.map((q) => q.text),
              });
            }
          }

          // Phase 1.5: capture structured uncertainty signal.
          if (ev.type === "uncertainty" && ev.data && typeof ev.data === "object") {
            const u = ev.data as {
              kind?: UncertaintyKind;
              reason?: string;
              follow_ups?: string[];
            };
            if (u.kind) {
              setUncertainty({
                kind: u.kind,
                reason: u.reason,
                follow_ups: Array.isArray(u.follow_ups) ? u.follow_ups : [],
              });
            }
          }

          if (ev.step === "done") {
            setFinalData(ev.data as DoneEventData);
            setStatus("done");
          }

          if (ev.step === "error") {
            if (ev.label === "cancelled") {
              setStatus("cancelled");
            } else {
              setStatus("error");
              setError(
                typeof ev.data === "string"
                  ? ev.data
                  : ev.label || "Stream error",
              );
            }
          }

          setEvents((prev) => [...prev, ev]);
        }
      }
    } catch (e: unknown) {
      // AbortController.abort() → DOMException name 'AbortError'.
      if (e instanceof DOMException && e.name === "AbortError") {
        setStatus((s) => (s === "cancelled" || s === "done" ? s : "cancelled"));
        return;
      }
      setStatus("error");
      setError(e instanceof Error ? e.message : "Unknown error");
    }
    },
    [reset],
  );

  const approvePlan = useCallback(
    async (editedSubQueries: string[] | null) => {
      const pending = approvalPending;
      if (!pending) return;
      const sid = sessionIdRef.current;
      if (!sid) return;
      try {
        await approveResearchPlan(pending.turnId, sid, editedSubQueries);
      } catch (e) {
        // Surface a soft error but don't abort the stream — the server will
        // resolve via /cancel or timeout if approve never lands.
        // eslint-disable-next-line no-console
        console.warn("approveResearchPlan failed", e);
      } finally {
        setApprovalPending(null);
      }
    },
    [approvalPending],
  );

  return {
    events,
    status,
    finalData,
    error,
    turnId,
    currentText,
    uncertainty,
    plan,
    phaseProgress,
    evidenceGaps,
    approvalPending,
    start,
    approvePlan,
    cancel,
    reset,
  };
}
