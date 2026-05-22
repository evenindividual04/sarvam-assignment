"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight, MessageSquareText } from "lucide-react";
import { listSessions } from "@/lib/api";
import type { SessionListItem } from "@/lib/types";
import { formatRelativeTime, truncate } from "@/lib/format";
import { ErrorPanel } from "@/components/shell/error-panel";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

export default function SessionsPage() {
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setErr(null);
    listSessions()
      .then((s) => setSessions(s))
      .catch((e) => setErr(e instanceof Error ? e.message : "error"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="px-6 md:px-12 py-12 max-w-5xl mx-auto w-full">
      <h1 className="text-2xl font-sans font-medium tracking-tight">Sessions</h1>
      <p className="mt-1 font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
        All past research sessions · most recent first
      </p>

      <div className="mt-10">
        {loading && (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-10 w-full" />
            ))}
          </div>
        )}
        {err && !loading && <ErrorPanel detail={err} onRetry={load} />}
        {!loading && !err && sessions.length === 0 && <SessionsEmptyState />}

        {sessions.length > 0 && (
          <div className="border-t border-border">
            <div className="grid grid-cols-[1fr_120px_180px_24px] gap-6 py-3 border-b border-border font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
              <div>Session</div>
              <div className="text-right tabular-nums">Turns</div>
              <div className="text-right">Last activity</div>
              <div />
            </div>
            {sessions.map((s) => (
              <Link
                key={s.session_id}
                href={`/sessions/${encodeURIComponent(s.session_id)}`}
                className="grid grid-cols-[1fr_120px_180px_24px] gap-6 py-3 border-b border-border hover:bg-surface-hover/50 transition-colors items-center"
              >
                <div className="font-mono text-[12px] text-foreground truncate">
                  {truncate(s.session_id, 36)}
                </div>
                <div className="text-right font-mono tabular-nums-lining text-[12px] text-muted-foreground">
                  {s.turn_count}
                </div>
                <div className="text-right font-mono text-[11px] text-muted-foreground">
                  {formatRelativeTime(s.updated_at)}
                </div>
                <div className="text-right font-mono text-muted-foreground">
                  →
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function SessionsEmptyState() {
  return (
    <div className="mt-6 border border-border rounded-[8px] bg-surface px-8 py-12 flex flex-col items-center text-center">
      <div className="size-12 rounded-full border border-border flex items-center justify-center text-muted-foreground mb-5">
        <MessageSquareText size={20} aria-hidden />
      </div>
      <h2 className="font-sans text-[18px] font-medium tracking-tight text-foreground">
        No research sessions yet
      </h2>
      <p className="mt-2 font-sans text-[13px] text-muted-foreground max-w-prose leading-normal">
        Every question you ask becomes a session here, with full provenance:
        the sub-queries the planner generated, the URLs fetched, and the exact
        context shown to the model.
      </p>
      <Link href="/" className="mt-6">
        <Button size="sm" className="font-mono text-[11px] uppercase tracking-[0.12em]">
          Start your first research
          <ArrowRight size={14} className="ml-2" aria-hidden />
        </Button>
      </Link>

      <div className="mt-10 w-full max-w-md border-t border-border pt-6">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground mb-4 text-left">
          How sessions work
        </div>
        <ol className="space-y-3 text-left">
          {[
            "Ask a question — in any of the supported languages.",
            "The agent plans, searches the web, verifies claims, and probes for source conflicts.",
            "Every turn is saved here with the trace inspector for deep audit.",
          ].map((step, i) => (
            <li
              key={i}
              className="flex items-start gap-3 font-sans text-[13px] text-muted-foreground leading-normal"
            >
              <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-accent w-4 shrink-0 mt-1">
                {i + 1}
              </span>
              <span>{step}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
