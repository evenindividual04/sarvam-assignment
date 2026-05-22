"use client";

/**
 * Vagueness-gated clarification panel.
 *
 * Rendered inline above the streaming pipeline when the backend emits a
 * `clarification_offered` SSE event. Non-blocking: the turn continues; the
 * panel is informational and offers click-to-resubmit shortcuts to one of
 * the suggested interpretations the planner derived from the success
 * criteria.
 *
 * Anti-copying note: this is NOT a multi-step "Question 2 of 3" wizard
 * with chip-style modal flow. It's a single, dismissable typographic
 * suggestion — at most one of these fires per turn, and only when the
 * vagueness heuristic (entity count + length + ambiguity flag + wh
 * breadth) crosses the 0.55 threshold AND the planner has concrete
 * success criteria to offer as interpretations.
 */

import { useState } from "react";
import { cn } from "@/lib/utils";

interface ClarificationPanelProps {
  question?: string;
  originalQuery: string;
  interpretations: string[];
  /** Re-submit the turn with the picked interpretation as the new query. */
  onPickInterpretation: (refinedQuery: string) => void;
  /** Hide the panel and continue with the current turn unchanged. */
  onDismiss: () => void;
}

export function ClarificationPanel({
  question,
  originalQuery,
  interpretations,
  onPickInterpretation,
  onDismiss,
}: ClarificationPanelProps) {
  const [dismissed, setDismissed] = useState(false);
  if (dismissed) return null;
  if (interpretations.length === 0) return null;

  const handleDismiss = () => {
    setDismissed(true);
    onDismiss();
  };

  return (
    <div
      className={cn(
        "border border-border bg-surface rounded-[8px] p-5 mb-4",
        "shadow-sm",
      )}
      role="region"
      aria-label="Clarifying question"
    >
      <div className="flex items-baseline justify-between mb-3">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground">
          Possible refinement
        </div>
        <button
          type="button"
          onClick={handleDismiss}
          className="font-mono text-[10px] uppercase tracking-[0.12em] text-subtle-foreground hover:text-foreground"
          aria-label="Dismiss clarification"
        >
          Dismiss
        </button>
      </div>

      {question && (
        <p className="text-[13px] text-foreground leading-relaxed mb-1">
          {question}
        </p>
      )}
      <p className="text-[12px] text-muted-foreground italic mb-4 border-l-2 border-border pl-3">
        Your query: {originalQuery}
      </p>

      <ul className="flex flex-col gap-2">
        {interpretations.map((interp, i) => (
          <li key={`${interp}-${i}`}>
            <button
              type="button"
              onClick={() => onPickInterpretation(interp)}
              className={cn(
                "w-full text-left border border-border rounded-[6px] px-3 py-2",
                "text-[13px] font-sans text-foreground leading-snug",
                "hover:border-border-accent hover:bg-surface-hover transition-colors",
                "focus:outline-none focus:border-border-accent",
              )}
            >
              {interp}
            </button>
          </li>
        ))}
      </ul>

      <p className="mt-3 font-mono text-[10px] uppercase tracking-[0.12em] text-subtle-foreground">
        Clicking a refinement cancels the current turn and starts a new one with the selected query.
      </p>
    </div>
  );
}
