"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { listSessions } from "@/lib/api";
import type { SessionListItem } from "@/lib/types";
import { formatRelativeTime, truncate } from "@/lib/format";
import { ErrorPanel } from "@/components/shell/error-panel";
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
    <div className="px-8 md:px-12 py-12 max-w-[1280px] mx-auto w-full">
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
        {!loading && !err && sessions.length === 0 && (
          <div className="font-mono text-[12px] text-subtle-foreground">
            No sessions yet. Submit a query from the Chat tab to start one.
          </div>
        )}

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
                <div className="text-right font-mono tabular-nums text-[12px] text-muted-foreground">
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
