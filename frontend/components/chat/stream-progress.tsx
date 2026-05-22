"use client";

import { useEffect, useMemo, useState } from "react";
import { getStreamLabels } from "@/lib/api";
import type { ExecutionEvent, StreamStep } from "@/lib/types";
import type { SseStatus } from "@/lib/use-sse-research";
import { cn } from "@/lib/utils";
import { SourceChip, type SourceChipStatus } from "./source-chip";

// Phase 1.25: default labels (fallback when /stream/labels hasn't loaded yet).
// Backend is the single source of truth — this only ensures first-paint works
// offline / before the fetch resolves.
const DEFAULT_PIPELINE: Array<{ step: StreamStep; label: string }> = [
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
  ms?: number; // final timing once `done` event lands
}

interface StreamProgressProps {
  events: ExecutionEvent[];
  status: SseStatus;
  error?: string | null;
}

/**
 * Vertical pipeline progress with live per-step elapsed time.
 *
 * Concurrent-mode correctness: render is kept pure. All non-deterministic
 * inputs (wall-clock time, first-seen timestamps) are held in `useState` and
 * mutated only inside `useEffect`. Render reads those values as plain props
 * so React 18+ can safely tear down / replay renders without observable drift.
 *
 * - `firstSeen` is a state-held `Map<step, ms>` populated by an effect when a
 *   new step first appears in the event stream. Immutable updates (a fresh
 *   Map per change) keep referential equality honest for downstream memos.
 * - `now` is state-held, updated by a 250ms `setInterval` (only while
 *   streaming) so the *active* step's elapsed counter ticks live. Cleaned up
 *   on unmount and when streaming finishes — no idle CPU.
 * - Done steps show their final timing from the terminal `done` event's
 *   per-stage `*_ms` fields; queued/cancelled steps show no time at all.
 */
export function StreamProgress({ events, status, error }: StreamProgressProps) {
  const [now, setNow] = useState<number>(() => Date.now());
  const [firstSeen, setFirstSeen] = useState<Map<StreamStep, number>>(
    () => new Map(),
  );

  // Phase 1.25: fetch the canonical labels once. Falls back to DEFAULT_PIPELINE
  // until the network call resolves.
  const [pipeline, setPipeline] = useState(DEFAULT_PIPELINE);
  useEffect(() => {
    let cancelled = false;
    getStreamLabels()
      .then((res) => {
        if (cancelled) return;
        const order = res.order && res.order.length > 0 ? res.order : DEFAULT_PIPELINE.map((p) => p.step);
        setPipeline(
          order
            .filter((s) => DEFAULT_PIPELINE.some((p) => p.step === s))
            .map((s) => ({
              step: s as StreamStep,
              label: res.labels[s] ?? s,
            })),
        );
      })
      .catch(() => {
        // Keep the default; no UI degradation.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Phase 1.25: aggregate typed events into a per-phase view model. The legacy
  // step-based pipeline rendering stays as-is — new info is layered on top.
  const phaseExtras = useMemo(() => aggregateTypedEvents(events), [events]);

  // Capture first-seen timestamps via an effect (NOT during render) so the
  // render function stays pure for React 18+ concurrent mode. The Map is
  // replaced (never mutated) when a new step arrives, so React sees a fresh
  // reference and dependent consumers re-render deterministically. The
  // setState-in-effect pattern is intentional here: wall-clock time is an
  // external input, captured at observation rather than at render.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setFirstSeen((prev) => {
      let next: Map<StreamStep, number> | null = null;
      const stamp = Date.now();
      for (const ev of events) {
        if (!prev.has(ev.step) && !(next?.has(ev.step) ?? false)) {
          if (next === null) next = new Map(prev);
          next.set(ev.step, stamp);
        }
      }
      return next ?? prev;
    });
  }, [events]);

  // Drive live re-renders only while streaming. Stops once status leaves
  // "streaming" so finished pipelines don't burn idle CPU.
  useEffect(() => {
    if (status !== "streaming") return;
    const id = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(id);
  }, [status]);

  const rows = deriveStatuses(events, status, pipeline);

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
          {pipeline.map((p, i) => {
            const r = rows[i];
            const timing = computeTiming(r, p.step, firstSeen, now);
            const extra = phaseExtras.phases[p.step];
            return (
              <li
                key={p.step}
                className="grid grid-cols-[auto_1fr_auto] items-start gap-3"
              >
                <span
                  className={cn(
                    "size-[10px] mt-1 rounded-full -ml-[14px] z-10 relative ring-2 ring-surface",
                    DOT_CLASS[r.status],
                  )}
                />
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <div
                      className={cn(
                        "font-mono text-[11px] uppercase tracking-[0.12em] truncate",
                        r.status === "queued" && "text-subtle-foreground",
                        r.status === "active" && "text-foreground",
                        r.status === "done" && "text-muted-foreground",
                        r.status === "skipped" && "text-amber-500",
                        r.status === "cancelled" && "text-red-500",
                      )}
                    >
                      {p.label}
                    </div>
                    {/* Phase 1.25: inline progress pill (e.g. "5/9 · 2 failed") */}
                    {extra?.progress && (
                      <span className="font-mono text-[10px] px-1.5 py-0.5 rounded-[3px] border border-border bg-surface text-subtle-foreground">
                        {extra.progress.current}/{extra.progress.total}
                        {extra.progress.failed
                          ? ` · ${extra.progress.failed} failed`
                          : ""}
                      </span>
                    )}
                    {/* Phase 1.25: phase_finished latency + count badge */}
                    {extra?.finishedBadge && (
                      <span className="font-mono text-[10px] px-1.5 py-0.5 rounded-[3px] border border-border bg-surface text-subtle-foreground">
                        {extra.finishedBadge}
                      </span>
                    )}
                  </div>
                  {r.note && (
                    <div className="font-mono italic text-[10px] text-subtle-foreground mt-0.5">
                      skipped: {r.note}
                    </div>
                  )}
                  {/* Phase 1.25: per-source rail rendered beneath the active phase */}
                  {extra?.chips && extra.chips.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1.5">
                      {extra.chips.slice(0, 12).map((c) => (
                        <SourceChip
                          key={c.url}
                          url={c.url}
                          title={c.title}
                          domain={c.domain}
                          status={c.status}
                        />
                      ))}
                      {extra.chips.length > 12 && (
                        <span className="font-mono text-[10px] text-subtle-foreground self-center">
                          +{extra.chips.length - 12} more
                        </span>
                      )}
                    </div>
                  )}
                </div>
                <span
                  className={cn(
                    "font-mono tabular-nums text-[11px] shrink-0",
                    r.status === "active"
                      ? "text-accent"
                      : "text-muted-foreground",
                  )}
                >
                  {timing}
                </span>
              </li>
            );
          })}
        </ul>
        {/* Phase 1.25: terminal cost/token pill */}
        {phaseExtras.runFinished && (
          <div className="mt-3 pt-3 border-t border-border font-mono text-[10px] text-muted-foreground">
            Generated in {(phaseExtras.runFinished.total_latency_ms / 1000).toFixed(2)}s ·{" "}
            {phaseExtras.runFinished.total_tokens.toLocaleString()} tokens · $
            {phaseExtras.runFinished.cost_usd.toFixed(4)}
          </div>
        )}
      </div>

      {error && status === "error" && (
        <div className="mt-4 px-3 py-2 border border-destructive/40 bg-destructive/10 text-destructive font-mono text-[11px] rounded-[3px]">
          {error}
        </div>
      )}
    </div>
  );
}

const DOT_CLASS: Record<RowStatus, string> = {
  // Theme-aware via CSS variables (var(--surface-emphasis) etc.) so they
  // remain visible on both cream and near-black backgrounds.
  queued: "bg-[var(--surface-emphasis)]",
  active: "bg-accent animate-pulse",
  done: "bg-muted-foreground/60",
  skipped: "bg-surface ring-1 ring-amber-500/50",
  cancelled: "bg-surface ring-1 ring-destructive/60",
};

function computeTiming(
  row: RowState,
  step: StreamStep,
  firstSeen: Map<StreamStep, number>,
  nowMs: number,
): string {
  if (row.status === "queued" || row.status === "cancelled") return "";
  if (row.status === "skipped") return "—";
  // Prefer the authoritative `*_ms` from the terminal done event when present.
  if (typeof row.ms === "number") return formatTiming(row.ms);
  // Otherwise compute live elapsed from the client-side first-seen timestamp.
  const start = firstSeen.get(step);
  if (typeof start !== "number") return row.status === "active" ? "···" : "";
  return formatTiming(nowMs - start);
}

function formatTiming(ms: number | undefined): string {
  if (typeof ms !== "number" || Number.isNaN(ms) || ms < 0) return "";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function deriveStatuses(
  events: ExecutionEvent[],
  sseStatus: SseStatus,
  pipeline: Array<{ step: StreamStep; label: string }> = DEFAULT_PIPELINE,
): RowState[] {
  const result: RowState[] = pipeline.map(() => ({ status: "queued" }));
  const stepIndex = new Map(pipeline.map((p, i) => [p.step, i]));
  let lastSeen = -1;

  for (const ev of events) {
    const idx = stepIndex.get(ev.step);
    if (idx === undefined) continue;
    lastSeen = Math.max(lastSeen, idx);

    if (ev.step === "generating") {
      if (result[idx].status === "queued") result[idx].status = "active";
    } else {
      if (result[idx].status !== "skipped") result[idx].status = "active";
    }
  }

  for (let i = 0; i < pipeline.length; i++) {
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
      pipeline.forEach((p, i) => {
        const key = map[p.step];
        const v = key && d ? (d[key] as number | undefined) : undefined;
        if (typeof v === "number") result[i].ms = v;
      });
    }
  }

  if (sseStatus === "cancelled" || sseStatus === "error") {
    let hitActive = false;
    for (let i = 0; i < pipeline.length; i++) {
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

// Phase 1.25: typed-event aggregation. Walks the ExecutionEvent stream once,
// bucketing source chips per phase and surfacing the cumulative run_finished
// pill. The reducer is purely additive — legacy step-based rendering stays
// untouched.
interface PhaseExtras {
  chips?: Array<{
    url: string;
    title: string;
    domain: string;
    status: SourceChipStatus;
  }>;
  progress?: { current: number; total: number; failed: number };
  finishedBadge?: string;
}

interface RunFinishedSummary {
  total_latency_ms: number;
  total_tokens: number;
  cost_usd: number;
}

interface AggregateResult {
  phases: Record<string, PhaseExtras | undefined>;
  runFinished?: RunFinishedSummary;
}

function aggregateTypedEvents(events: ExecutionEvent[]): AggregateResult {
  const out: AggregateResult = { phases: {} };
  const ensure = (phase: string): PhaseExtras => {
    const existing = out.phases[phase];
    if (existing) return existing;
    const fresh: PhaseExtras = { chips: [] };
    out.phases[phase] = fresh;
    return fresh;
  };

  // Track per-URL chip index across phases so we can update fetch status on
  // chips originally created during the searching phase.
  const chipByUrl = new Map<string, { phase: string; idx: number }>();

  for (const ev of events) {
    const t = (ev as ExecutionEvent & { type?: string }).type;
    if (!t) continue;
    const d = ev.data as Record<string, unknown> | undefined;
    if (!d) continue;

    if (t === "source_found") {
      const url = String(d.url || "");
      if (!url) continue;
      const bucket = ensure("searching");
      const chip = {
        url,
        title: String(d.title || ""),
        domain: String(d.domain || ""),
        status: "pending" as SourceChipStatus,
      };
      const idx = (bucket.chips ??= []).length;
      bucket.chips.push(chip);
      chipByUrl.set(url, { phase: "searching", idx });
    } else if (t === "source_fetched") {
      const url = String(d.url || "");
      const loc = chipByUrl.get(url);
      if (!loc) continue;
      const bucket = out.phases[loc.phase];
      const chip = bucket?.chips?.[loc.idx];
      if (chip && bucket?.chips) {
        // Immutable update — mutating a derived object inside useMemo
        // is fragile under StrictMode and React's reconciliation. Replace
        // the chip in place with a new object instead.
        bucket.chips[loc.idx] = {
          ...chip,
          status: d.status === "ok" ? "ok" : "error",
        };
      }
    } else if (t === "phase_progress") {
      const name = String(d.name || "");
      if (!name) continue;
      const bucket = ensure(name);
      bucket.progress = {
        current: Number(d.current ?? 0),
        total: Number(d.total ?? 0),
        failed: Number(d.failed ?? 0),
      };
    } else if (t === "phase_finished") {
      const name = String(d.name || "");
      if (!name) continue;
      const bucket = ensure(name);
      const ms = Number(d.duration_ms ?? 0);
      const sec = (ms / 1000).toFixed(2);
      let count: number | null = null;
      if (typeof d.n_results === "number") count = d.n_results;
      else if (typeof d.fetched === "number") count = d.fetched;
      else if (typeof d.n_selected === "number") count = d.n_selected;
      bucket.finishedBadge =
        count !== null ? `${sec}s · ${count}` : `${sec}s`;
    } else if (t === "run_finished") {
      const usage = (d.usage as { total_tokens?: number } | undefined) ?? {};
      out.runFinished = {
        total_latency_ms: Number(d.total_latency_ms ?? 0),
        total_tokens: Number(usage.total_tokens ?? 0),
        cost_usd: Number(d.cost_usd ?? 0),
      };
    }
  }
  return out;
}
