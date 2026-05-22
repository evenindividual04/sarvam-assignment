"use client";

import { cn } from "@/lib/utils";
import type { EvidenceGap, UncertaintyKind } from "@/lib/types";

interface UncertaintyBadgeProps {
  kind: UncertaintyKind;
  followUps: string[];
  reason?: string;
  onFollowUpClick: (query: string) => void;
  // Phase 1.875: per-sub-query evidence gaps (some angles yielded no usable
  // evidence). Rendered as a subsection below the main follow-ups.
  evidenceGaps?: EvidenceGap[];
}

interface KindStyle {
  label: string;
  badge: string;
  dot: string;
}

const KIND_STYLES: Record<UncertaintyKind, KindStyle> = {
  weak: {
    label: "Low Confidence",
    badge: "border-amber-500/40 bg-amber-500/10 text-amber-600",
    dot: "bg-amber-500",
  },
  missing: {
    label: "No Evidence Found",
    badge: "border-red-500/40 bg-red-500/10 text-red-600",
    dot: "bg-red-500",
  },
  conflict: {
    label: "Sources Disagree",
    badge: "border-purple-500/40 bg-purple-500/10 text-purple-600",
    dot: "bg-purple-500",
  },
};

export function UncertaintyBadge({
  kind,
  followUps,
  reason,
  onFollowUpClick,
  evidenceGaps,
}: UncertaintyBadgeProps): React.ReactElement {
  const style = KIND_STYLES[kind];
  const chips = followUps.filter((f) => f && f.trim());
  const gaps = (evidenceGaps ?? []).filter((g) => g.query && g.query.trim());

  return (
    <div className="mb-4 rounded-[8px] border border-border bg-surface/40 p-4">
      <div className="flex items-center gap-2">
        <span
          className={cn(
            "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5",
            "font-mono text-[10px] uppercase tracking-[0.14em]",
            style.badge,
          )}
        >
          <span
            className={cn("inline-block size-[6px] rounded-full", style.dot)}
          />
          {style.label}
        </span>
        {reason && (
          <span className="font-mono text-[11px] text-muted-foreground line-clamp-1">
            {reason}
          </span>
        )}
      </div>

      {chips.length > 0 && (
        <div className="mt-3">
          <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground mb-2">
            Try one of these instead
          </div>
          <div className="flex flex-wrap gap-2">
            {chips.map((q, i) => (
              <button
                key={`${i}-${q}`}
                type="button"
                onClick={() => onFollowUpClick(q)}
                className={cn(
                  "rounded-[6px] border border-border bg-surface px-3 py-1.5",
                  "font-sans text-[12px] text-foreground text-left",
                  "hover:border-border-accent hover:bg-surface-hover transition-colors",
                  "max-w-full truncate",
                )}
                title={q}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {gaps.length > 0 && (
        <div className="mt-3 pt-3 border-t border-border">
          <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground mb-2">
            Evidence gaps
          </div>
          <div className="font-sans text-[11px] text-muted-foreground mb-2">
            Some angles yielded no usable evidence:
          </div>
          <div className="flex flex-wrap gap-2">
            {gaps.map((g, i) => (
              <button
                key={`gap-${i}-${g.query}`}
                type="button"
                onClick={() => onFollowUpClick(g.query)}
                className={cn(
                  "rounded-[6px] border border-amber-500/30 bg-amber-500/5 px-3 py-1.5",
                  "font-sans text-[12px] text-foreground text-left",
                  "hover:border-amber-500/60 hover:bg-amber-500/10 transition-colors",
                  "max-w-full truncate",
                )}
                title={`${g.query} (${g.intent}: ${g.reason})`}
              >
                <span className="font-mono text-[10px] uppercase tracking-[0.10em] text-amber-600 mr-1.5">
                  {g.reason === "no_results" ? "no results" : "filtered"}
                </span>
                {g.query}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
