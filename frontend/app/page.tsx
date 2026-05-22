"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown } from "lucide-react";
import { ChatInput, type ChatInputHandle } from "@/components/chat/chat-input";
import { StreamProgress } from "@/components/chat/stream-progress";
import { MetricBar } from "@/components/chat/metric-bar";
import { UncertaintyBadge } from "@/components/chat/uncertainty-badge";
import { PlanCard } from "@/components/chat/plan-card";
import { PlanApprovalPanel } from "@/components/chat/plan-approval-panel";
import { ClarificationPanel } from "@/components/chat/clarification-panel";
import { RichMarkdown } from "@/lib/markdown";
import {
  useSseResearch,
  type ClarificationPayload,
  type PlanApprovalPending,
  type UncertaintySignal,
} from "@/lib/use-sse-research";
import {
  TraceInspector,
  doneToTraceData,
  turnToTraceData,
  type TraceInspectorData,
} from "@/components/trace/trace-inspector";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { formatMs } from "@/lib/format";
import { getSessionHistory, getTurnDetail } from "@/lib/api";
import type {
  DoneEventData,
  EvidenceGap,
  ExecutionEvent,
  PlannerOutput,
  PlanQueryPhase,
  Turn,
} from "@/lib/types";
import type {
  SseStatus,
  HopEvidenceItem,
  SourceContributionBundle,
  SourceRoleByUrl,
  TerminatorPayload,
  ReasoningEvent,
} from "@/lib/use-sse-research";

interface ChatRow {
  query: string;
  events: ExecutionEvent[];
  status: SseStatus;
  liveText?: string;
  final?: DoneEventData;
  error?: string | null;
  /** Phase 1.5: structured uncertainty signal for this turn, if emitted. */
  uncertainty?: UncertaintySignal | null;
  /** Phase 1.875: planner-level state surfaced as a PlanCard above the answer. */
  plan?: PlannerOutput | null;
  phaseProgress?: Record<string, PlanQueryPhase>;
  evidenceGaps?: EvidenceGap[];
  /** Forensic-differentiation events captured live from this turn. */
  hopEvidence?: HopEvidenceItem[];
  sourceContribution?: SourceContributionBundle | null;
  sourceRoles?: SourceRoleByUrl;
  terminator?: TerminatorPayload | null;
  reasoningEvents?: ReasoningEvent[];
  /** Plan-approval gate, when `approval_required=true` was set on the request. */
  approvalPending?: PlanApprovalPending | null;
  /** Vagueness-gated clarifier, non-blocking, at most once per turn. */
  clarification?: ClarificationPayload | null;
  /** Set when row was rehydrated from /sessions history (not from a live SSE stream). */
  historyTurn?: Turn;
}

// F7: Bharat-themed query pool. The empty state samples 4 at random per
// mount; ↻ Shuffle re-rolls. Mix of English / Hindi / Hinglish, mix of
// factual / multi-hop / comparison / recent. Curly quotes match the
// typography pass; Devanagari and Hinglish keep their native punctuation.
const SUGGESTED_POOL = [
  "What is India’s current repo rate, and how has it changed in the last 12 months?",
  "भारत में मानसून कब आता है और इस वर्ष कैसा रहा?",
  "Compare India’s UPI and ONDC adoption — what’s the state of DPI exports?",
  "DPI exports kaha kaha ho rahe hain abhi?",
  "What’s the latest from IndiaAI mission — funding allocated, deliverables shipped?",
  "Sarvam-30B vs Llama 3.3 70B Indic benchmark comparison",
  "How is Aadhaar enabling DPI globally? Recent country adoptions?",
  "What is the current status of India’s semiconductor mission and SemiconIndia program?",
];

const SUGGESTED_COUNT = 4;

function pickSuggestions(pool: readonly string[], n: number): string[] {
  // Fisher–Yates over a copy; deterministic only within a single render.
  const copy = [...pool];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, n);
}

function newSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `s-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function ChatPage() {
  const [sessionId, setSessionId] = useState<string>("");
  const [rows, setRows] = useState<ChatRow[]>([]);
  const [traceOpen, setTraceOpen] = useState(false);
  const [traceData, setTraceData] = useState<TraceInspectorData | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<ChatInputHandle | null>(null);
  // Scroll-to-bottom pill: pause auto-scroll when the user is reading older
  // turns mid-stream, resume silently when they return near the bottom.
  const [autoStick, setAutoStick] = useState(true);
  const [showJumpPill, setShowJumpPill] = useState(false);

  const sse = useSseResearch();

  useEffect(() => {
    const stored =
      typeof window !== "undefined"
        ? window.localStorage.getItem("dra:lastSessionId")
        : null;
    const id = stored || newSessionId();
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSessionId(id);
    if (typeof window !== "undefined") {
      window.localStorage.setItem("dra:lastSessionId", id);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setRows((prev) => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      if (
        last.status === "done" ||
        last.status === "cancelled" ||
        last.status === "error"
      ) {
        return prev;
      }
      const updated: ChatRow = {
        ...last,
        events: sse.events,
        status: sse.status,
        liveText: sse.currentText,
        final: sse.finalData ?? undefined,
        error: sse.error,
        uncertainty: sse.uncertainty,
        plan: sse.plan,
        phaseProgress: sse.phaseProgress,
        evidenceGaps: sse.evidenceGaps,
        hopEvidence: sse.hopEvidence,
        sourceContribution: sse.sourceContribution,
        sourceRoles: sse.sourceRoles,
        terminator: sse.terminator,
        reasoningEvents: sse.reasoningEvents,
        approvalPending: sse.approvalPending,
        clarification: sse.clarification,
      };
      return [...prev.slice(0, -1), updated];
    });
  }, [
    sse.events,
    sse.status,
    sse.currentText,
    sse.finalData,
    sse.error,
    sse.uncertainty,
    sse.plan,
    sse.phaseProgress,
    sse.evidenceGaps,
    sse.hopEvidence,
    sse.sourceContribution,
    sse.sourceRoles,
    sse.terminator,
    sse.reasoningEvents,
    sse.approvalPending,
    sse.clarification,
  ]);

  // Auto-stick to bottom while streaming, but only when the user hasn't
  // scrolled away. The scroll listener below flips `autoStick` when the user
  // moves >100px from the bottom and restores it once they're within 50px.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !autoStick) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [rows, sse.currentText, autoStick]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    let timer: number | null = null;
    const onScroll = () => {
      if (timer !== null) window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
        if (distance > 100) {
          setAutoStick(false);
          setShowJumpPill(true);
        } else if (distance < 50) {
          setAutoStick(true);
          setShowJumpPill(false);
        }
      }, 100);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      el.removeEventListener("scroll", onScroll);
      if (timer !== null) window.clearTimeout(timer);
    };
  }, []);

  const jumpToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    setAutoStick(true);
    setShowJumpPill(false);
  }, []);

  useEffect(() => {
    if (sse.status === "done") {
      // Broadcast so the sidebar can refresh its session list.
      window.dispatchEvent(new CustomEvent("dra:turn-done"));
    }
  }, [sse.status]);

  // Hydrate rows from backend whenever the active session changes (initial mount,
  // returning from /sessions or /eval tabs, or selecting a session from the rail).
  // Skip while a stream is in flight so we don't blow away live state.
  useEffect(() => {
    if (!sessionId) return;
    if (sse.status === "streaming") return;
    let alive = true;
    getSessionHistory(sessionId)
      .then((turns) => {
        if (!alive) return;
        if (!turns || turns.length === 0) return;
        const hydrated: ChatRow[] = turns.map((t) => ({
          query: t.query,
          events: [],
          status: "done" as SseStatus,
          liveText: t.response,
          historyTurn: t,
        }));
        setRows(hydrated);
      })
      .catch(() => {
        /* leave rows empty; backend may be unreachable or session is brand-new */
      });
    return () => {
      alive = false;
    };
    // sse.status intentionally excluded — we only want to hydrate on session change,
    // not on every status transition during a live stream.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const inFlight = sse.status === "streaming";

  const handleSubmit = (
    query: string,
    options: { approvalRequired: boolean } = { approvalRequired: false },
  ) => {
    if (!sessionId) return;
    setRows((prev) => [
      ...prev,
      { query, events: [], status: "streaming", liveText: "" },
    ]);
    void sse.start(query, sessionId, options);
  };

  const handleCancel = () => {
    sse.cancel();
    toast.info("Cancelling…");
  };

  const handleFollowUpClick = (query: string) => {
    inputRef.current?.prefill(query);
  };

  const handleNewSession = useCallback(() => {
    if (sse.status === "streaming") sse.cancel();
    sse.reset();
    const id = newSessionId();
    setRows([]);
    setSessionId(id);
    if (typeof window !== "undefined") {
      window.localStorage.setItem("dra:lastSessionId", id);
      window.dispatchEvent(
        new CustomEvent("dra:session-select", { detail: { sessionId: id } }),
      );
    }
  }, [sse]);

  const handlePickSession = useCallback(
    (id: string) => {
      if (id === sessionId) return;
      if (sse.status === "streaming") sse.cancel();
      sse.reset();
      setRows([]);
      setSessionId(id);
      if (typeof window !== "undefined") {
        window.localStorage.setItem("dra:lastSessionId", id);
      }
    },
    [sessionId, sse],
  );

  // The sidebar lives in the layout tree and cannot share state via props.
  // It broadcasts user intent through CustomEvents on `window`.
  useEffect(() => {
    const onSelect = (e: Event) => {
      const detail = (e as CustomEvent<{ sessionId: string }>).detail;
      if (detail?.sessionId) handlePickSession(detail.sessionId);
    };
    const onNew = () => handleNewSession();
    window.addEventListener("dra:session-select", onSelect);
    window.addEventListener("dra:session-new", onNew);
    return () => {
      window.removeEventListener("dra:session-select", onSelect);
      window.removeEventListener("dra:session-new", onNew);
    };
  }, [handlePickSession, handleNewSession]);

  const openTrace = (row: ChatRow) => {
    if (row.final) {
      setTraceData(
        doneToTraceData(row.query, row.final, sessionId, {
          hopEvidence: row.hopEvidence,
          sourceContribution: row.sourceContribution ?? null,
          sourceRoles: row.sourceRoles,
          terminator: row.terminator ?? null,
          reasoningEvents: row.reasoningEvents,
        }),
      );
      setTraceOpen(true);
      return;
    }
    if (row.historyTurn) {
      const turn = row.historyTurn;
      setTraceData(turnToTraceData(turn));
      setTraceOpen(true);
      // Fetch full detail (claim audit, contradiction probe, context_xml) lazily.
      getTurnDetail(sessionId, turn.turn_id)
        .then((detail) => setTraceData(turnToTraceData(detail)))
        .catch(() => {
          /* keep partial */
        });
    }
  };

  const isEmpty = useMemo(() => rows.length === 0, [rows]);

  return (
    <div className="flex flex-1 min-h-0">
      <div className="flex-1 flex flex-col min-w-0 relative">
        <div className="h-14 px-8 border-b border-border flex items-center justify-between bg-background/80 backdrop-blur-sm sticky top-0 z-10">
          <div className="flex items-baseline gap-4 min-w-0">
            <h1 className="font-sans text-[15px] font-medium tracking-tight">
              Deep Research Agent
            </h1>
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              {sessionId ? `session · ${sessionId.slice(0, 8)}` : ""}
            </span>
          </div>
        </div>

        {showJumpPill && (
          <button
            onClick={jumpToBottom}
            aria-label="Jump to latest message"
            className="absolute left-1/2 -translate-x-1/2 bottom-28 z-20 flex items-center gap-2 px-3 py-1.5 rounded-full border border-border bg-background/95 backdrop-blur-sm shadow-md hover:bg-surface-hover transition-colors font-mono text-[11px] uppercase tracking-[0.12em] text-foreground"
          >
            <ArrowDown size={12} aria-hidden />
            Jump to latest
          </button>
        )}

        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto px-6 md:px-12 py-10"
        >
          {isEmpty ? (
            <EmptyState onPick={(q) => handleSubmit(q)} />
          ) : (
            <div className="max-w-3xl mx-auto flex flex-col gap-10 pb-16">
              {rows.map((row, i) => (
                <ChatTurn
                  key={i}
                  row={row}
                  onFollowUpClick={handleFollowUpClick}
                  onCancel={
                    i === rows.length - 1 && row.status === "streaming"
                      ? handleCancel
                      : undefined
                  }
                  onOpenTrace={() => openTrace(row)}
                  onRegenerate={
                    (row.status === "cancelled" || row.status === "error") &&
                    !inFlight
                      ? () => {
                          // Drop the failed row, then re-submit the same query.
                          setRows((prev) => prev.filter((_, j) => j !== i));
                          handleSubmit(row.query);
                        }
                      : undefined
                  }
                  onApprovePlan={
                    i === rows.length - 1 ? sse.approvePlan : undefined
                  }
                  onApprovalCancel={
                    i === rows.length - 1 ? sse.cancel : undefined
                  }
                  onPickInterpretation={(q) => {
                    sse.cancel();
                    handleSubmit(q);
                  }}
                />
              ))}
            </div>
          )}
        </div>

        <div className="border-t border-border bg-background px-6 md:px-12 py-5">
          <div className="max-w-3xl mx-auto">
            <ChatInput
              ref={inputRef}
              onSubmit={handleSubmit}
              onCancel={handleCancel}
              busy={inFlight}
              autoFocus
            />
          </div>
        </div>
      </div>

      <TraceInspector
        open={traceOpen}
        onOpenChange={setTraceOpen}
        data={traceData}
      />
    </div>
  );
}

/**
 * EMPTY STATE — FIRST EDITORIAL MOMENT.
 * The chat hero is the ONE place outside the eval drill-down question where
 * Instrument Serif italic appears. Its scarcity is the point.
 */
function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  const [shuffleTick, setShuffleTick] = useState(0);
  const suggestions = useMemo(
    () => pickSuggestions(SUGGESTED_POOL, SUGGESTED_COUNT),
    [shuffleTick],
  );
  return (
    <div className="max-w-[640px] mx-auto py-20 md:py-28">
      <h1 className="font-display italic text-5xl md:text-6xl leading-[1.05] tracking-tight">
        What would you like to{" "}
        <span className="text-accent">research</span> today?
      </h1>
      <p className="mt-5 font-sans text-base text-muted-foreground max-w-prose leading-normal">
        Multi-source web research with claim verification. Every fact is
        traced back to a URL fetched in this session; conflicting sources
        are surfaced rather than hidden.
      </p>
      {/* Pipeline marquee — sentence case + horizontal scroll on narrow
          viewports so the row never wraps awkwardly mid-arrow. */}
      <p className="mt-2 font-mono text-[11px] text-subtle-foreground whitespace-nowrap overflow-x-auto -mx-2 px-2">
        Planning · Search · Fetch · Rerank · Probe · Synthesize · Verify
      </p>

      <div className="mt-12">
        <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-subtle-foreground mb-1 pb-2 border-b border-border">
          Suggested
        </div>
        {suggestions.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="group flex w-full items-center justify-between border-b border-border py-4 text-left transition-colors hover:bg-surface-hover px-1"
          >
            <span className="italic font-sans text-[15px] text-muted-foreground group-hover:text-foreground">
              &ldquo;{s}&rdquo;
            </span>
            <span className="font-mono text-muted-foreground group-hover:text-accent ml-4 shrink-0">
              →
            </span>
          </button>
        ))}
        <button
          type="button"
          onClick={() => setShuffleTick((prev) => prev + 1)}
          aria-label="Shuffle suggested queries"
          className="mt-3 font-sans text-[12px] text-muted-foreground hover:text-foreground transition-colors"
        >
          ↻ Shuffle
        </button>
      </div>
    </div>
  );
}

function ChatTurn({
  row,
  onCancel,
  onOpenTrace,
  onRegenerate,
  onFollowUpClick,
  onApprovePlan,
  onApprovalCancel,
  onPickInterpretation,
}: {
  row: ChatRow;
  onCancel?: () => void;
  onOpenTrace?: () => void;
  onRegenerate?: () => void;
  onFollowUpClick?: (q: string) => void;
  onApprovePlan?: (editedSubQueries: string[] | null) => Promise<void>;
  onApprovalCancel?: () => void;
  onPickInterpretation?: (refinedQuery: string) => void;
}) {
  const answerText =
    row.final?.answer ??
    row.liveText ??
    (row.status === "cancelled" ? "_[Cancelled by user]_" : "");

  // Phase 1.875: surface unverified numeric tokens (from numeric_audit in
  // run_metadata) so RichMarkdown can wrap them with a ⚠ unverified badge.
  const unverifiedTokens = (() => {
    const meta = row.final?.run_metadata as
      | { numeric_audit?: { audit?: Array<{ token: string; grounded: boolean }> } }
      | undefined;
    const audit = meta?.numeric_audit?.audit;
    if (!audit || audit.length === 0) return undefined;
    const s = new Set<string>();
    for (const a of audit) {
      if (!a.grounded && a.token) s.add(a.token);
    }
    return s.size > 0 ? s : undefined;
  })();

  // B5: URL-keyed quote map for citation hover popovers. The orchestrator
  // emits cite_quote_map keyed by doc_id; we re-key by URL because the
  // rewritten markdown contains only [Title — domain](URL).
  const citeQuoteByUrl = (() => {
    const meta = row.final?.run_metadata;
    const quotes = meta?.cite_quote_map;
    const docMap = row.final?.doc_map;
    if (!quotes || !docMap) return undefined;
    const out: Record<string, string> = {};
    for (const [docId, quote] of Object.entries(quotes)) {
      const entry = docMap[docId];
      if (entry && entry[1] && quote) out[entry[1]] = quote;
    }
    return Object.keys(out).length > 0 ? out : undefined;
  })();

  const copyAnswer = async () => {
    if (!row.final?.answer) return;
    try {
      await navigator.clipboard.writeText(row.final.answer);
      toast.success("Copied");
    } catch {
      /* no-op */
    }
  };

  return (
    <div className="flex flex-col gap-5">
      {/* User message */}
      <div className="ml-12">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground mb-2">
          You
        </div>
        <div className="border-l-2 border-border pl-4 text-[15px] leading-relaxed font-sans">
          {row.query}
        </div>
      </div>

      {!row.historyTurn && (
        <div className={cn(row.status === "done" && "opacity-90")}>
          <StreamProgress
            events={row.events}
            status={row.status}
            error={row.error}
          />
        </div>
      )}

      {/* Assistant message */}
      <div>
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground mb-2">
          Agent
        </div>
        <div className="flex-1 min-w-0">
          {row.approvalPending && onApprovePlan && onApprovalCancel && (
            <PlanApprovalPanel
              turnId={row.approvalPending.turnId}
              plannerOutput={row.approvalPending.plannerOutput}
              subQueries={row.approvalPending.subQueries}
              onApprove={onApprovePlan}
              onCancel={onApprovalCancel}
            />
          )}
          {row.clarification && onPickInterpretation && (
            <ClarificationPanel
              question={row.clarification.clarifying_question}
              originalQuery={row.clarification.original_query}
              interpretations={row.clarification.possible_interpretations}
              onPickInterpretation={onPickInterpretation}
              onDismiss={() => {
                /* dismissal is local-only — the turn continues; no backend signal needed */
              }}
            />
          )}
          {row.plan && (
            <PlanCard plan={row.plan} phaseProgress={row.phaseProgress} />
          )}
          {row.uncertainty && onFollowUpClick && (
            <UncertaintyBadge
              kind={row.uncertainty.kind}
              followUps={row.uncertainty.follow_ups}
              reason={row.uncertainty.reason}
              evidenceGaps={row.evidenceGaps}
              onFollowUpClick={onFollowUpClick}
            />
          )}
          {answerText ? (
            <RichMarkdown
              unverifiedNumericTokens={unverifiedTokens}
              citeQuoteByUrl={citeQuoteByUrl}
            >
              {answerText}
            </RichMarkdown>
          ) : (
            <span className="text-sm text-muted-foreground font-mono">
              {row.status === "streaming"
                ? "working…"
                : "awaiting response."}
            </span>
          )}

          {row.final && (
            <div className="mt-6 pt-5 border-t border-border space-y-4">
              <div className="grid grid-cols-2 gap-6">
                <MetricBar
                  label="Citation integrity"
                  value={row.final.citation_integrity_score}
                />
                <MetricBar
                  label="Claim precision"
                  value={row.final.claim_precision_score}
                />
              </div>
              <div className="flex items-center justify-between pt-1">
                <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
                  {row.final.urls.length} sources ·{" "}
                  {row.final.prompt_tokens + row.final.completion_tokens} tokens
                  · {formatMs(row.final.latency_ms)}
                </span>
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={copyAnswer}
                    className="font-mono text-[11px] uppercase tracking-[0.12em]"
                  >
                    Copy
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onOpenTrace}
                    className="font-mono text-[11px] uppercase tracking-[0.12em]"
                  >
                    Trace
                  </Button>
                </div>
              </div>
            </div>
          )}

          {!row.final && row.historyTurn && (
            <div className="mt-6 pt-5 border-t border-border space-y-4">
              <div className="grid grid-cols-2 gap-6">
                <MetricBar
                  label="Citation integrity"
                  value={row.historyTurn.citation_integrity_score}
                />
                <MetricBar
                  label="Claim precision"
                  value={row.historyTurn.claim_precision_score}
                />
              </div>
              <div className="flex items-center justify-between pt-1">
                <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
                  {(row.historyTurn.urls_opened?.length ?? 0)} sources ·{" "}
                  {(row.historyTurn.prompt_tokens ?? 0) +
                    (row.historyTurn.completion_tokens ?? 0)}{" "}
                  tokens · {formatMs(row.historyTurn.latency_ms)}
                </span>
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={async () => {
                      try {
                        await navigator.clipboard.writeText(
                          row.historyTurn?.response ?? "",
                        );
                        toast.success("Copied");
                      } catch {
                        /* no-op */
                      }
                    }}
                    className="font-mono text-[11px] uppercase tracking-[0.12em]"
                  >
                    Copy
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onOpenTrace}
                    className="font-mono text-[11px] uppercase tracking-[0.12em]"
                  >
                    Trace
                  </Button>
                </div>
              </div>
            </div>
          )}

          {onCancel && (
            <div className="mt-4 pt-3 border-t border-border flex justify-end">
              <Button
                variant="ghost"
                size="sm"
                onClick={onCancel}
                className="font-mono text-[11px] uppercase tracking-[0.12em] text-destructive hover:text-destructive"
              >
                Cancel
              </Button>
            </div>
          )}

          {onRegenerate && (
            <div className="mt-4 pt-3 border-t border-border flex items-center justify-between gap-3">
              <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground">
                {row.status === "cancelled"
                  ? "This turn was cancelled."
                  : "This turn errored."}
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={onRegenerate}
                className="font-mono text-[11px] uppercase tracking-[0.12em] text-accent hover:text-accent"
              >
                ↻ Regenerate
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
