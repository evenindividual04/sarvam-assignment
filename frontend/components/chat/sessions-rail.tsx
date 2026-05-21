"use client";

import { useEffect, useState } from "react";
import { listSessions } from "@/lib/api";
import type { SessionListItem } from "@/lib/types";
import { formatRelativeTime, truncate } from "@/lib/format";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

interface SessionsRailProps {
  currentSessionId: string;
  onSelect: (sessionId: string) => void;
  onNew: () => void;
  refreshKey?: number;
}

export function SessionsRail({
  currentSessionId,
  onSelect,
  onNew,
  refreshKey,
}: SessionsRailProps) {
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    listSessions()
      .then((s) => {
        if (alive) {
          setSessions(s);
          setErr(null);
        }
      })
      .catch((e) => alive && setErr(e instanceof Error ? e.message : "error"))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [refreshKey]);

  // Refresh the list when the chat tab regains focus — covers the case where
  // the user submitted a turn elsewhere or refreshed the backend.
  useEffect(() => {
    const onFocus = () => {
      listSessions()
        .then(setSessions)
        .catch(() => {
          /* keep previous list */
        });
    };
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, []);

  return (
    <div className="w-[260px] shrink-0 border-r border-border bg-background flex flex-col h-full">
      <div className="px-5 h-14 border-b border-border flex items-center justify-between">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          Sessions
        </div>
        <button
          onClick={onNew}
          className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground hover:text-accent transition-colors"
        >
          + New
        </button>
      </div>
      <ScrollArea className="flex-1">
        <div className="py-2">
          {loading && (
            <div className="font-mono text-[11px] text-subtle-foreground px-5 py-4">
              loading…
            </div>
          )}
          {err && (
            <div className="font-mono text-[11px] text-subtle-foreground px-5 py-4">
              backend unreachable.
            </div>
          )}
          {!loading && !err && sessions.length === 0 && (
            <div className="font-mono text-[11px] text-subtle-foreground px-5 py-4 leading-relaxed">
              no sessions yet.
              <br />
              send a query to start.
            </div>
          )}
          {sessions.map((s) => {
            const active = s.session_id === currentSessionId;
            return (
              <button
                key={s.session_id}
                onClick={() => onSelect(s.session_id)}
                className={cn(
                  "group w-full text-left px-5 py-3 border-l-2 transition-colors flex flex-col gap-1",
                  active
                    ? "border-accent bg-surface-hover"
                    : "border-transparent hover:bg-surface-hover/60 hover:border-border-strong",
                )}
              >
                <div className="font-mono text-[11px] truncate text-foreground">
                  {truncate(s.session_id, 22)}
                </div>
                <div className="flex items-center gap-2 font-mono text-[10px] text-subtle-foreground tabular-nums">
                  <span>{s.turn_count}T</span>
                  <span className="text-border-strong">·</span>
                  <span>{formatRelativeTime(s.updated_at)}</span>
                </div>
              </button>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
