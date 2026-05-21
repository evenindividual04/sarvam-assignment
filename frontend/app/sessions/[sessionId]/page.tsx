"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { getSessionHistory, getTurnDetail } from "@/lib/api";
import type { Turn, TurnDetail } from "@/lib/types";
import { RichMarkdown } from "@/lib/markdown";
import { Button } from "@/components/ui/button";
import { MetricBar } from "@/components/chat/metric-bar";
import {
  TraceInspector,
  turnToTraceData,
  type TraceInspectorData,
} from "@/components/trace/trace-inspector";
import { formatDateTime, formatMs } from "@/lib/format";

interface PageProps {
  params: Promise<{ sessionId: string }>;
}

export default function SessionDetailPage({ params }: PageProps) {
  const { sessionId } = use(params);
  const decodedSessionId = decodeURIComponent(sessionId);

  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const [traceOpen, setTraceOpen] = useState(false);
  const [traceData, setTraceData] = useState<TraceInspectorData | null>(null);

  useEffect(() => {
    let alive = true;
    getSessionHistory(decodedSessionId)
      .then((ts) => alive && setTurns(ts))
      .catch((e) => alive && setErr(e instanceof Error ? e.message : "error"))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [decodedSessionId]);

  const openTrace = async (turn: Turn) => {
    setTraceData(turnToTraceData(turn));
    setTraceOpen(true);
    try {
      const detail: TurnDetail = await getTurnDetail(
        decodedSessionId,
        turn.turn_id,
      );
      setTraceData(turnToTraceData(detail));
    } catch {
      /* keep partial */
    }
  };

  return (
    <div className="px-8 md:px-12 py-12 max-w-[1024px] mx-auto w-full">
      <Link
        href="/sessions"
        className="inline-block font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground hover:text-foreground transition-colors mb-6"
      >
        ← All sessions
      </Link>

      <h1 className="text-2xl font-sans font-medium tracking-tight">Session</h1>
      <div className="mt-1 flex items-baseline gap-3 font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
        <span>{decodedSessionId}</span>
        <span className="text-border-strong">·</span>
        <span className="tabular-nums">
          {turns.length} turn{turns.length === 1 ? "" : "s"}
        </span>
      </div>

      <div className="mt-10">
        {loading && (
          <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
            loading…
          </div>
        )}
        {err && (
          <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
            {err.includes("404")
              ? "session not found."
              : "backend unreachable."}
          </div>
        )}
      </div>

      <div className="mt-2 flex flex-col gap-4">
        {turns.map((t, i) => (
          <article
            key={t.turn_id}
            className="border border-border rounded-[8px] bg-surface p-6"
          >
            <header className="flex items-baseline justify-between font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-4">
              <span>
                Turn{" "}
                <span className="text-foreground tabular-nums">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="mx-2 text-border-strong">·</span>
                <span className="normal-case tracking-normal">
                  {formatDateTime(t.created_at)}
                </span>
              </span>
              <span className="tabular-nums">{formatMs(t.latency_ms)}</span>
            </header>

            <div className="mb-5">
              <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground mb-2">
                Question
              </div>
              <div className="text-[15px] leading-relaxed font-sans border-l-2 border-border pl-4">
                {t.query}
              </div>
            </div>

            <div className="mb-5">
              <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground mb-2">
                Answer
              </div>
              <RichMarkdown>{t.response}</RichMarkdown>
            </div>

            {(t.citation_integrity_score !== undefined ||
              t.claim_precision_score !== undefined) && (
              <div className="grid grid-cols-2 gap-6 pt-5 border-t border-border">
                <MetricBar
                  label="Citation integrity"
                  value={t.citation_integrity_score}
                />
                <MetricBar
                  label="Claim precision"
                  value={t.claim_precision_score}
                />
              </div>
            )}

            {t.urls_opened && t.urls_opened.length > 0 && (
              <div className="mt-5 pt-5 border-t border-border">
                <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground mb-3">
                  Sources · {t.urls_opened.length}
                </div>
                <div className="flex flex-wrap gap-x-4 gap-y-1.5">
                  {t.urls_opened.slice(0, 10).map((u) => {
                    let domain = u;
                    try {
                      domain = new URL(u).hostname.replace(/^www\./, "");
                    } catch {
                      /* keep raw */
                    }
                    return (
                      <a
                        key={u}
                        href={u}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-mono text-[11px] text-muted-foreground hover:text-accent transition-colors"
                      >
                        {domain}
                      </a>
                    );
                  })}
                  {t.urls_opened.length > 10 && (
                    <span className="font-mono text-[11px] text-subtle-foreground">
                      +{t.urls_opened.length - 10} more
                    </span>
                  )}
                </div>
              </div>
            )}

            <div className="mt-5 pt-3 flex justify-end">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => openTrace(t)}
                className="font-mono text-[11px] uppercase tracking-[0.12em]"
              >
                Open trace →
              </Button>
            </div>
          </article>
        ))}
      </div>

      <TraceInspector
        open={traceOpen}
        onOpenChange={setTraceOpen}
        data={traceData}
      />
    </div>
  );
}
