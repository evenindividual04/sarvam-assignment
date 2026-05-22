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
  StreamEvent,
  TypedEventName,
  UncertaintyKind,
} from "./types";

export type ReasoningEvent = Extract<StreamEvent, { type: "reasoning" }>;

// Per-turn forensic event payloads.
// These shapes mirror the backend constants emitted by the orchestrator.
export interface HopEvidenceItem {
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

export interface SourceContributionRow {
  url: string;
  domain: string;
  title: string;
  tokens: number;
  share: number;
  citations: number;
}

export interface SourceContributionBundle {
  contributions: SourceContributionRow[];
  total_tokens: number;
}

export type SourceRoleByUrl = Record<
  string,
  { role: string; confidence: number }
>;

export interface TerminatorPayload {
  reason: string;
  hop?: number;
  detail?: string | null;
}

// Vagueness-gated clarifier payload (one per turn, fires post-planning).
// Non-blocking: the turn continues; the panel is purely informational and
// offers click-to-resubmit shortcuts to one of the suggested interpretations.
export interface ClarificationPayload {
  kind?: string;
  original_query: string;
  possible_interpretations: string[];
  clarifying_question?: string;
}

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
  // Forensic-differentiation events (one per hop / once per run).
  hopEvidence: HopEvidenceItem[];
  sourceContribution: SourceContributionBundle | null;
  sourceRoles: SourceRoleByUrl;
  terminator: TerminatorPayload | null;
  // Vagueness-gated clarifier (non-blocking; informational only).
  clarification: ClarificationPayload | null;
  reasoningEvents: ReasoningEvent[];
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
  // Forensic-differentiation state.
  const [hopEvidence, setHopEvidence] = useState<HopEvidenceItem[]>([]);
  const [sourceContribution, setSourceContribution] =
    useState<SourceContributionBundle | null>(null);
  const [sourceRoles, setSourceRoles] = useState<SourceRoleByUrl>({});
  const [terminator, setTerminator] = useState<TerminatorPayload | null>(null);
  const [clarification, setClarification] = useState<ClarificationPayload | null>(
    null,
  );
  const [reasoningEvents, setReasoningEvents] = useState<ReasoningEvent[]>([]);

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
    setHopEvidence([]);
    setSourceContribution(null);
    setSourceRoles({});
    setTerminator(null);
    setClarification(null);
    setReasoningEvents([]);
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

          // Synthesis text streaming. The orchestrator emits the same chunk
          // twice: once as a legacy `generating` frame with `data = string`,
          // and once as a typed `answer_delta` frame with `data = {text}`.
          // Consume EITHER shape (not both — otherwise the text doubles).
          // Defaulting to the typed shape when present makes us robust to a
          // proxy that strips the legacy frame.
          if (ev.type === "answer_delta" && ev.data && typeof ev.data === "object") {
            const txt = (ev.data as { text?: string }).text;
            if (typeof txt === "string" && txt.length > 0) {
              setCurrentText((prev) => prev + txt);
            }
          } else if (
            ev.step === "generating" &&
            typeof ev.data === "string"
          ) {
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

          // Forensic-differentiation event capture.
          // `hop_evidence` fires once per hop, after the hop's selecting phase.
          if (
            ev.type === "hop_evidence" &&
            ev.data &&
            typeof ev.data === "object"
          ) {
            const d = ev.data as Partial<HopEvidenceItem>;
            if (typeof d.hop === "number") {
              const row: HopEvidenceItem = {
                hop: d.hop,
                grounded: Array.isArray(d.grounded) ? d.grounded : [],
                open: Array.isArray(d.open) ? d.open : [],
              };
              setHopEvidence((prev) => {
                // Replace by hop number (idempotent on re-runs).
                const next = prev.filter((h) => h.hop !== row.hop);
                next.push(row);
                next.sort((a, b) => a.hop - b.hop);
                return next;
              });
            }
          }
          // `source_contribution` fires once per run after the hop loop.
          if (
            ev.type === "source_contribution" &&
            ev.data &&
            typeof ev.data === "object"
          ) {
            const d = ev.data as Partial<SourceContributionBundle>;
            if (Array.isArray(d.contributions)) {
              setSourceContribution({
                contributions: d.contributions as SourceContributionRow[],
                total_tokens:
                  typeof d.total_tokens === "number" ? d.total_tokens : 0,
              });
            }
          }
          // `source_role` fires once per run after the hop loop.
          if (
            ev.type === "source_role" &&
            ev.data &&
            typeof ev.data === "object"
          ) {
            const d = ev.data as {
              roles?: Array<{ url: string; role: string; confidence: number }>;
            };
            if (Array.isArray(d.roles)) {
              const map: SourceRoleByUrl = {};
              for (const r of d.roles) {
                if (r && typeof r.url === "string" && typeof r.role === "string") {
                  map[r.url] = {
                    role: r.role,
                    confidence:
                      typeof r.confidence === "number" ? r.confidence : 0,
                  };
                }
              }
              setSourceRoles(map);
            }
          }
          // `terminator` fires once per run when the hop loop exits.
          if (
            ev.type === "terminator" &&
            ev.data &&
            typeof ev.data === "object"
          ) {
            const d = ev.data as Partial<TerminatorPayload>;
            if (typeof d.reason === "string") {
              setTerminator({
                reason: d.reason,
                hop: typeof d.hop === "number" ? d.hop : undefined,
                detail: typeof d.detail === "string" ? d.detail : null,
              });
            }
          }
          // `clarification_offered` fires at most once per turn, gated by
          // the vagueness score + ambiguity heuristic. Non-blocking — the
          // turn continues; the panel is informational.
          if (
            ev.type === "clarification_offered" &&
            ev.data &&
            typeof ev.data === "object"
          ) {
            const d = ev.data as Partial<ClarificationPayload>;
            if (
              typeof d.original_query === "string" &&
              Array.isArray(d.possible_interpretations)
            ) {
              setClarification({
                kind: typeof d.kind === "string" ? d.kind : undefined,
                original_query: d.original_query,
                possible_interpretations: d.possible_interpretations.filter(
                  (s): s is string => typeof s === "string",
                ),
                clarifying_question:
                  typeof d.clarifying_question === "string"
                    ? d.clarifying_question
                    : undefined,
              });
            }
          }

          // B3: retrieval-grounded reasoning. Two emissions per hop
          // (intent + observation). Captured for ReasoningChip rendering.
          if (
            ev.type === "reasoning" &&
            ev.data &&
            typeof ev.data === "object"
          ) {
            const d = ev.data as {
              hop?: number;
              phase?: "intent" | "observation";
              queries?: ReasoningEvent["queries"];
              observation?: ReasoningEvent["observation"];
            };
            if (
              typeof d.hop === "number" &&
              (d.phase === "intent" || d.phase === "observation")
            ) {
              const row: ReasoningEvent = {
                type: "reasoning",
                hop: d.hop,
                phase: d.phase,
                queries: d.queries,
                observation: d.observation,
              };
              setReasoningEvents((prev) => [...prev, row]);
            }
          }

          if (ev.step === "done") {
            setFinalData(ev.data as DoneEventData);
            setStatus("done");
          }

          if (ev.step === "error") {
            if (ev.label === "cancelled") {
              setStatus("cancelled");
            } else if (ev.label === "approval_timeout") {
              setStatus("error");
              setError(
                "Plan approval timed out (no response within 5 minutes). Re-submit the query to try again.",
              );
              setApprovalPending(null);
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
      // Reader exited cleanly via `done: true` but the server never emitted
      // `{step:"done"}` (or `{step:"error"}`). Without this guard the row
      // stays stuck in `"streaming"` forever — the spinner keeps spinning,
      // the Cancel button keeps showing, and the user can't submit again.
      // Treat a silent close as an error so the UI surfaces a regenerate
      // affordance and `inFlight` releases.
      if (abortRef.current === ac) {
        setStatus((s) =>
          s === "done" || s === "error" || s === "cancelled" ? s : "error",
        );
        setError((prev) =>
          prev ?? "Stream closed unexpectedly before completion.",
        );
      }
    } catch (e: unknown) {
      // Critical: a new start() may have replaced abortRef.current by the
      // time this catch runs (e.g. clarification "pick interpretation"
      // cancels the current turn AND immediately starts a new one). If our
      // controller is no longer the active one, the abort belongs to a
      // superseded turn — don't touch state, the new turn owns it now.
      const stillCurrent = abortRef.current === ac;
      // AbortController.abort() → DOMException name 'AbortError'.
      if (e instanceof DOMException && e.name === "AbortError") {
        if (stillCurrent) {
          setStatus((s) => (s === "cancelled" || s === "done" ? s : "cancelled"));
        }
        return;
      }
      if (stillCurrent) {
        setStatus("error");
        setError(e instanceof Error ? e.message : "Unknown error");
      }
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
    hopEvidence,
    sourceContribution,
    sourceRoles,
    terminator,
    clarification,
    reasoningEvents,
    start,
    approvePlan,
    cancel,
    reset,
  };
}
