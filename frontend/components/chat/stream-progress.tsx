"use client";

import type { ExecutionEvent, StreamStep } from "@/lib/types";
import type { SseStatus } from "@/lib/use-sse-research";
import { cn } from "@/lib/utils";

const PIPELINE: Array<{ step: StreamStep; label: string }> = [
  { step: "planning", label: "Planning" },
  { step: "searching", label: "Searching the web" },
  { step: "fetching", label: "Fetching sources" },
  { step: "selecting", label: "Selecting relevant context" },
  { step: "probing", label: "Probing for contradictions" },
  { step: "generating", label: "Generating answer with citations" },
  { step: "verifying", label: "Verifying claims" },
];

type RowStatus = "queued" | "active" | "done" | "skipped" | "cancelled";

interface RowState {
  status: RowStatus;
  note?: string;
  ms?: number;
}

function deriveStatuses(
  events: ExecutionEvent[],
  sseStatus: SseStatus,
): RowState[] {
  const result: RowState[] = PIPELINE.map(() => ({ status: "queued" }));
  const stepIndex = new Map(PIPELINE.map((p, i) => [p.step, i]));
  let lastSeen = -1;

  // First-seen timestamps per step (for elapsed display)
  const firstSeen = new Map<number, number>();

  for (const ev of events) {
    const idx = stepIndex.get(ev.step);
    if (idx === undefined) continue;
    lastSeen = Math.max(lastSeen, idx);
    if (!firstSeen.has(idx)) {
      const ts = (ev as { ts?: number; timestamp?: number }).ts ??
        (ev as { timestamp?: number }).timestamp;
      if (typeof ts === "number") firstSeen.set(idx, ts);
    }

    if (ev.step === "generating") {
      if (result[idx].status === "queued") result[idx].status = "active";
    } else {
      if (result[idx].status !== "skipped") result[idx].status = "active";
    }
  }

  for (let i = 0; i < PIPELINE.length; i++) {
    if (i < lastSeen && result[i].status === "active") {
      result[i].status = "done";
    }
  }

  if (sseStatus === "done") {
    for (let i = 0; i <= lastSeen; i++) {
      if (result[i].status === "active" || result[i].status === "queued") {
        result[i].status = "done";
      }
    }
    // Pull per-stage timings from the terminal done event
    const doneEvent = events.find((e) => e.step === "done");
    if (doneEvent) {
      const d = doneEvent.data as Record<string, unknown> | undefined;
      const map: Partial<Record<StreamStep, string>> = {
        planning: "planning_ms",
        searching: "search_ms",
        fetching: "fetch_ms",
        selecting: "select_ms",
        probing: "probe_ms",
        generating: "synthesize_ms",
        verifying: "verify_ms",
      };
      PIPELINE.forEach((p, i) => {
        const key = map[p.step];
        const v = key && d ? (d[key] as number | undefined) : undefined;
        if (typeof v === "number") result[i].ms = v;
      });
    }
  }

  if (sseStatus === "cancelled" || sseStatus === "error") {
    let hitActive = false;
    for (let i = 0; i < PIPELINE.length; i++) {
      if (result[i].status === "active") {
        result[i].status = "cancelled";
        hitActive = true;
      } else if (hitActive && result[i].status === "queued") {
        result[i].status = "cancelled";
      }
    }
  }

  for (const ev of events) {
    const d = ev.data as { skipped?: boolean; reason?: string } | undefined;
    if (d && typeof d === "object" && d.skipped) {
      const idx = stepIndex.get(ev.step);
      if (idx !== undefined) {
        result[idx].status = "skipped";
        result[idx].note = d.reason;
      }
    }
    if (ev.step === "done") {
      const meta = (ev.data as { run_metadata?: Record<string, unknown> })
        ?.run_metadata;
      const st = (meta?.state_trace as string[] | undefined) ?? [];
      for (const marker of st) {
        if (typeof marker !== "string") continue;
        const m = marker.match(/^([A-Z_]+)_SKIPPED(?::(.+))?$/);
        if (!m) continue;
        const phase = m[1].toLowerCase();
        const reason = m[2];
        const stepName = (
          {
            conflict_check: "probing",
            probing: "probing",
            verifying_claims: "verifying",
            verifying: "verifying",
            fetching: "fetching",
            searching: "searching",
            selecting: "selecting",
          } as Record<string, StreamStep>
        )[phase];
        if (stepName) {
          const idx = stepIndex.get(stepName);
          if (idx !== undefined) {
            result[idx].status = "skipped";
            result[idx].note = reason;
          }
        }
      }
    }
  }

  return result;
}

const DOT_CLASS: Record<RowStatus, string> = {
  queued: "bg-zinc-700",
  active: "bg-accent animate-pulse",
  done: "bg-zinc-500",
  skipped: "bg-zinc-800 ring-1 ring-amber-600/40 ring-offset-0",
  cancelled: "bg-zinc-800 ring-1 ring-red-600/40 ring-offset-0",
};

function formatTiming(ms: number | undefined): string {
  if (typeof ms !== "number" || Number.isNaN(ms)) return "";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

interface StreamProgressProps {
  events: ExecutionEvent[];
  status: SseStatus;
  error?: string | null;
}

export function StreamProgress({ events, status, error }: StreamProgressProps) {
  const rows = deriveStatuses(events, status);

  return (
    <div className="border border-border rounded-[8px] bg-surface px-5 py-4">
      <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground mb-4 pb-2 border-b border-border">
        Pipeline
      </div>

      <div className="relative pl-4">
        <div
          aria-hidden
          className="absolute left-[5px] top-1 bottom-1 w-px bg-border"
        />
        <ul className="flex flex-col gap-2.5">
          {PIPELINE.map((p, i) => {
            const r = rows[i];
            return (
              <li
                key={p.step}
                className="grid grid-cols-[auto_1fr_auto] items-center gap-3"
              >
                <span
                  className={cn(
                    "size-[10px] rounded-full -ml-[14px] z-10 relative",
                    DOT_CLASS[r.status],
                    "ring-2 ring-surface",
                  )}
                />
                <div className="min-w-0">
                  <div
                    className={cn(
                      "font-mono text-[11px] uppercase tracking-[0.12em] truncate",
                      r.status === "queued" && "text-subtle-foreground",
                      r.status === "active" && "text-foreground",
                      r.status === "done" && "text-muted-foreground",
                      r.status === "skipped" && "text-amber-400/80",
                      r.status === "cancelled" && "text-red-400/80",
                    )}
                  >
                    {p.label}
                  </div>
                  {r.note && (
                    <div className="font-mono italic text-[10px] text-subtle-foreground mt-0.5">
                      skipped: {r.note}
                    </div>
                  )}
                </div>
                <span className="font-mono tabular-nums text-[11px] text-muted-foreground shrink-0">
                  {r.status === "active" ? "···" : formatTiming(r.ms)}
                </span>
              </li>
            );
          })}
        </ul>
      </div>

      {error && status === "error" && (
        <div className="mt-4 px-3 py-2 border border-red-900/60 bg-red-950/30 text-red-300 font-mono text-[11px] rounded-[3px]">
          {error}
        </div>
      )}
    </div>
  );
}
