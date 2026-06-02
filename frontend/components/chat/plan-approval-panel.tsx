"use client";

/**
 * Phase 2 — Plan Approval Gate panel.
 *
 * Rendered inline above the streaming pipeline when the backend emits a
 * `plan_approval` SSE event. The user can:
 *   - approve the plan as-is (POST /research/approve with sub_queries=null),
 *   - edit / delete / add sub_queries (capped at 6 by the server schema), or
 *   - cancel the turn (POST /research/cancel).
 *
 * The server enforces a 300s default timeout (APPROVAL_TIMEOUT_S env override);
 * we surface a read-only countdown badge so the user knows the window.
 */

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { PlannerOutput } from "@/lib/types";

interface PlanApprovalPanelProps {
  turnId: string;
  plannerOutput: PlannerOutput;
  subQueries: string[];
  onApprove: (editedSubQueries: string[] | null) => void;
  onCancel: () => void;
  /** Default 300s; injected so tests can override. */
  timeoutSeconds?: number;
  /** Wall-clock time the `plan_approval` SSE event arrived. Required for
   *  the countdown to survive parent re-renders / row remounts — without
   *  it the timer resets to 300s every time React decides to unmount and
   *  re-add the panel (e.g. on row-list re-rendering during streaming). */
  arrivedAt?: number;
}

const MAX_SUB_QUERIES = 6;
const MAX_QUERY_CHARS = 200;

export function PlanApprovalPanel({
  turnId,
  plannerOutput,
  subQueries,
  onApprove,
  onCancel,
  timeoutSeconds = 300,
  arrivedAt,
}: PlanApprovalPanelProps) {
  const [edited, setEdited] = useState<string[]>(() => [...subQueries]);
  const [submitting, setSubmitting] = useState(false);
  const [remaining, setRemaining] = useState(timeoutSeconds);

  useEffect(() => {
    setEdited([...subQueries]);
  }, [subQueries]);

  useEffect(() => {
    if (submitting) return;
    // Anchor the countdown on the actual moment the approval event arrived
    // (via `arrivedAt` prop). Falling back to effect-run-time only
    // matters during synthetic tests that don't supply the prop.
    const anchor = arrivedAt ?? Date.now();
    const compute = () =>
      Math.max(0, timeoutSeconds - Math.floor((Date.now() - anchor) / 1000));
    
    setRemaining(compute());
    const id = window.setInterval(() => {
      setRemaining(compute());
    }, 1000);
    return () => window.clearInterval(id);
  }, [arrivedAt, timeoutSeconds, submitting]);

  const expired = remaining <= 0;

  const dirty = useMemo(() => {
    if (edited.length !== subQueries.length) return true;
    return edited.some((q, i) => q !== subQueries[i]);
  }, [edited, subQueries]);

  const canSubmit =
    !expired &&
    edited.length > 0 &&
    edited.every((q) => q.trim().length > 0);

  const handleApprove = () => {
    if (submitting || expired || !canSubmit) return;
    setSubmitting(true);
    const payload = dirty ? edited.map((q) => q.trim()) : null;
    onApprove(payload);
  };

  const handleCancel = () => {
    if (submitting) return;
    setSubmitting(true);
    onCancel();
  };

  const updateQuery = (idx: number, value: string) => {
    setEdited((prev) => prev.map((q, i) => (i === idx ? value : q)));
  };

  const deleteQuery = (idx: number) => {
    setEdited((prev) => prev.filter((_, i) => i !== idx));
  };

  const addQuery = () => {
    if (edited.length >= MAX_SUB_QUERIES) return;
    setEdited((prev) => [...prev, ""]);
  };

  return (
    <div
      className={cn(
        "border border-border bg-surface rounded-[8px] p-5 mb-4",
        "shadow-sm",
      )}
      data-turn-id={turnId}
      role="region"
      aria-label="Plan approval"
    >
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground">
            Plan awaiting approval
          </div>
          <p className="mt-1 text-[13px] text-muted-foreground">
            Review the planner&rsquo;s sub-queries below. Edit or remove them, then
            approve to continue. Cancel aborts the turn.
          </p>
        </div>
        <span
          className={cn(
            "font-mono text-[10px] tabular-nums",
            expired ? "text-destructive" : "text-subtle-foreground",
          )}
          aria-live="polite"
        >
          {expired ? "timed out" : `${remaining}s`}
        </span>
      </div>

      {plannerOutput.strategy && (
        <p className="text-[12px] text-muted-foreground italic mb-3 border-l-2 border-border pl-3">
          {plannerOutput.strategy}
        </p>
      )}

      <ul className="flex flex-col gap-2 mb-3">
        {edited.map((q, idx) => (
          <li key={idx} className="flex items-center gap-2">
            <span className="font-mono text-[10px] text-subtle-foreground w-5 shrink-0 text-right">
              {idx + 1}.
            </span>
            <input
              type="text"
              value={q}
              maxLength={MAX_QUERY_CHARS}
              disabled={submitting}
              onChange={(e) => updateQuery(idx, e.target.value)}
              className={cn(
                "flex-1 bg-background border border-border rounded-[6px] px-2 py-1.5",
                "text-[13px] font-sans",
                "focus:outline-none focus:border-border-accent",
                "disabled:opacity-60",
              )}
              aria-label={`Sub-query ${idx + 1}`}
            />
            <button
              type="button"
              onClick={() => deleteQuery(idx)}
              disabled={submitting || edited.length <= 1}
              className={cn(
                "font-mono text-[10px] uppercase tracking-[0.12em]",
                "text-subtle-foreground hover:text-destructive",
                "disabled:opacity-30 disabled:cursor-not-allowed",
                "px-2 py-1",
              )}
              aria-label={`Delete sub-query ${idx + 1}`}
            >
              ×
            </button>
          </li>
        ))}
      </ul>

      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={addQuery}
          disabled={submitting || edited.length >= MAX_SUB_QUERIES}
          className={cn(
            "font-mono text-[11px] uppercase tracking-[0.12em]",
            "text-muted-foreground hover:text-foreground",
            "disabled:opacity-30 disabled:cursor-not-allowed",
          )}
        >
          + Add sub-query
        </button>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCancel}
            disabled={submitting}
            className="font-mono text-[11px] uppercase tracking-[0.12em] text-destructive hover:text-destructive"
          >
            Cancel
          </Button>
          <Button
            size="sm"
            onClick={handleApprove}
            disabled={submitting || !canSubmit}
            className="font-mono text-[11px] uppercase tracking-[0.12em]"
            title={
              expired
                ? "Approval window has expired — re-submit the query to try again."
                : undefined
            }
          >
            {expired
              ? "Window expired"
              : dirty
                ? "Approve edited plan"
                : "Approve plan"}
          </Button>
        </div>
      </div>
    </div>
  );
}
