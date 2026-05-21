"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { SessionsRail } from "@/components/chat/sessions-rail";
import { ChatInput } from "@/components/chat/chat-input";
import { StreamProgress } from "@/components/chat/stream-progress";
import { MetricBar } from "@/components/chat/metric-bar";
import { RichMarkdown } from "@/lib/markdown";
import { useSseResearch } from "@/lib/use-sse-research";
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
import type { DoneEventData, ExecutionEvent, Turn } from "@/lib/types";
import type { SseStatus } from "@/lib/use-sse-research";

interface ChatRow {
  query: string;
  events: ExecutionEvent[];
  status: SseStatus;
  liveText?: string;
  final?: DoneEventData;
  error?: string | null;
  /** Set when row was rehydrated from /sessions history (not from a live SSE stream). */
  historyTurn?: Turn;
}

const SUGGESTED = [
  "What is India's current repo rate, and how has it changed in the last 12 months?",
  "Summarize the latest research on Mixture-of-Experts LLMs in 2026.",
  "Compare GPT-5 and Claude Opus 4.7 on coding benchmarks.",
  "What's the status of the EU AI Act enforcement?",
];

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
  const [refreshKey, setRefreshKey] = useState(0);
  const scrollRef = useRef<HTMLDivElement | null>(null);

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
      };
      return [...prev.slice(0, -1), updated];
    });
  }, [sse.events, sse.status, sse.currentText, sse.finalData, sse.error]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [rows, sse.currentText]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (sse.status === "done") setRefreshKey((k) => k + 1);
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
        // eslint-disable-next-line react-hooks/set-state-in-effect
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

  const handleSubmit = (query: string) => {
    if (!sessionId) return;
    setRows((prev) => [
      ...prev,
      { query, events: [], status: "streaming", liveText: "" },
    ]);
    void sse.start(query, sessionId);
  };

  const handleCancel = () => {
    sse.cancel();
    toast.info("Cancelling…");
  };

  const handleNewSession = () => {
    if (sse.status === "streaming") sse.cancel();
    sse.reset();
    const id = newSessionId();
    setRows([]);
    setSessionId(id);
    if (typeof window !== "undefined") {
      window.localStorage.setItem("dra:lastSessionId", id);
    }
  };

  const openTrace = (row: ChatRow) => {
    if (row.final) {
      setTraceData(doneToTraceData(row.query, row.final));
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
      <div className="hidden lg:block">
        <SessionsRail
          currentSessionId={sessionId}
          onSelect={(id) => {
            if (id === sessionId) return;
            sse.reset();
            setRows([]);
            setSessionId(id);
            if (typeof window !== "undefined") {
              window.localStorage.setItem("dra:lastSessionId", id);
            }
          }}
          onNew={handleNewSession}
          refreshKey={refreshKey}
        />
      </div>

      <div className="flex-1 flex flex-col min-w-0">
        <div className="h-14 px-8 border-b border-border flex items-center justify-between bg-background/80 backdrop-blur-sm sticky top-0 z-10">
          <div className="flex items-baseline gap-4 min-w-0">
            <h1 className="font-sans text-[15px] font-medium tracking-tight">
              Deep Research Agent
            </h1>
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              {sessionId ? `session · ${sessionId.slice(0, 8)}` : ""}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleNewSession}
              className="font-mono text-[11px] uppercase tracking-[0.12em]"
            >
              New session
            </Button>
          </div>
        </div>

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
                  onCancel={
                    i === rows.length - 1 && row.status === "streaming"
                      ? handleCancel
                      : undefined
                  }
                  onOpenTrace={() => openTrace(row)}
                />
              ))}
            </div>
          )}
        </div>

        <div className="border-t border-border bg-background px-6 md:px-12 py-5">
          <div className="max-w-3xl mx-auto">
            <ChatInput
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
  return (
    <div className="max-w-[640px] mx-auto py-20 md:py-28">
      <h1 className="font-display italic text-5xl md:text-6xl leading-[1.05] tracking-tight">
        What would you like to{" "}
        <span className="text-accent">research</span> today?
      </h1>
      <p className="mt-5 font-sans text-base text-muted-foreground max-w-lg">
        Web-grounded answers with full provenance — planning, search, fetch,
        rerank, and synthesis, every step recorded.
      </p>

      <div className="mt-12">
        <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-subtle-foreground mb-1 pb-2 border-b border-border">
          Suggested
        </div>
        {SUGGESTED.map((s) => (
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
      </div>
    </div>
  );
}

function ChatTurn({
  row,
  onCancel,
  onOpenTrace,
}: {
  row: ChatRow;
  onCancel?: () => void;
  onOpenTrace?: () => void;
}) {
  const answerText =
    row.final?.answer ??
    row.liveText ??
    (row.status === "cancelled" ? "_[Cancelled by user]_" : "");

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
          {answerText ? (
            <RichMarkdown>{answerText}</RichMarkdown>
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
                  {formatMs(row.historyTurn.latency_ms)}
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
        </div>
      </div>
    </div>
  );
}
