"use client";

import { cn } from "@/lib/utils";
import type {
  PlanQueryPhase,
  PlannerOutput,
  PlannerTypedQuery,
} from "@/lib/types";

interface PlanCardProps {
  plan: PlannerOutput;
  /**
   * Phase 1.875: per-sub-query phase progress. Keys are TypedQuery.text.
   * Missing key → "pending". Updated by the parent as `source_found` /
   * `source_fetched` / `phase_finished:fetching` events stream in.
   */
  phaseProgress?: Record<string, PlanQueryPhase>;
}

type PillTone = "accent" | "amber" | "emerald" | "rose" | "muted" | "purple";

function Pill({
  label,
  tone = "muted",
  title,
}: {
  label: string;
  tone?: PillTone;
  title?: string;
}) {
  const toneClass: Record<PillTone, string> = {
    accent: "border-accent/40 bg-accent/10 text-accent",
    amber: "border-amber-500/40 bg-amber-500/10 text-amber-600",
    emerald: "border-emerald-500/40 bg-emerald-500/10 text-emerald-600",
    rose: "border-red-500/40 bg-red-500/10 text-red-600",
    muted: "border-border bg-surface text-muted-foreground",
    purple: "border-purple-500/40 bg-purple-500/10 text-purple-600",
  };
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5",
        "font-mono text-[10px] uppercase tracking-[0.14em]",
        toneClass[tone],
      )}
    >
      {label}
    </span>
  );
}

function PhaseIndicator({ phase }: { phase: PlanQueryPhase }) {
  if (phase === "done") {
    return (
      <span
        aria-label="done"
        className="inline-flex size-4 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-600 font-mono text-[10px]"
      >
        ✓
      </span>
    );
  }
  if (phase === "searching" || phase === "fetching") {
    return (
      <span
        aria-label={phase}
        className="inline-block size-3 animate-pulse rounded-full bg-accent"
      />
    );
  }
  return (
    <span
      aria-label="pending"
      className="inline-block size-3 rounded-full border border-border"
    />
  );
}

function intentTone(intent: string): PillTone {
  switch (intent) {
    case "primary":
      return "accent";
    case "comparison":
      return "muted";
    case "recency_check":
      return "amber";
    case "contradiction_probe":
      return "purple";
    case "definition":
      return "emerald";
    default:
      return "muted";
  }
}

function difficultyTone(d: string | undefined): PillTone {
  if (d === "hard") return "rose";
  if (d === "easy") return "emerald";
  return "muted";
}

function timeSensitivityTone(t: string | undefined): PillTone {
  if (t === "live") return "rose";
  if (t === "recent") return "amber";
  return "muted";
}

function confidenceTone(c: string | undefined): PillTone {
  if (c === "high") return "emerald";
  if (c === "low") return "amber";
  return "muted";
}

export function PlanCard({
  plan,
  phaseProgress,
}: PlanCardProps): React.ReactElement {
  const queries: PlannerTypedQuery[] = plan.queries ?? [];
  const criteria = (plan.success_criteria ?? []).filter(
    (c) => c && c.trim(),
  );

  // Detect fallback-path plans so the UI signals that this isn't the
  // normal planner output. The orchestrator stamps strategy with
  // "Direct retrieval fallback" (+ optional reason) when the planner
  // LLM call timed out or raised. Fallback plans are 1 query and
  // confidence=low — still actionable, but reviewers shouldn't mistake
  // them for the rich multi-query decomposition the planner normally
  // produces.
  const isFallback =
    typeof plan.strategy === "string" &&
    plan.strategy.toLowerCase().startsWith("direct retrieval fallback");

  return (
    <div className="mb-5 rounded-[8px] border border-border bg-surface/40 p-4">
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground">
          Plan
        </span>
        {isFallback && (
          <Pill
            label="Fallback"
            tone="amber"
            title="The planner LLM failed; the agent fell back to a direct-retrieval plan using the user's original query verbatim."
          />
        )}
        {plan.confidence && (
          <Pill
            label={`Confidence: ${plan.confidence}`}
            tone={confidenceTone(plan.confidence)}
            title="Planner self-rated confidence"
          />
        )}
        {plan.difficulty && (
          <Pill
            label={`Difficulty: ${plan.difficulty}`}
            tone={difficultyTone(plan.difficulty)}
            title="Difficulty estimate from planner"
          />
        )}
        {plan.time_sensitivity && (
          <Pill
            label={`Time: ${plan.time_sensitivity}`}
            tone={timeSensitivityTone(plan.time_sensitivity)}
            title="Time sensitivity"
          />
        )}
        {plan.ambiguity_flag && (
          <Pill label="Ambiguous" tone="amber" title="Planner flagged ambiguity" />
        )}
      </div>

      {plan.strategy && (
        <p className="mb-3 text-[13px] leading-relaxed text-foreground/90 font-sans">
          {plan.strategy}
        </p>
      )}

      {queries.length > 0 && (
        <ul className="flex flex-col gap-2 mb-3">
          {queries.map((q, i) => {
            const phase: PlanQueryPhase =
              phaseProgress?.[q.text] ?? "pending";
            return (
              <li
                key={`${i}-${q.text}`}
                className="flex items-start gap-2 text-[13px]"
              >
                <span className="mt-1 shrink-0">
                  <PhaseIndicator phase={phase} />
                </span>
                <div className="flex-1 min-w-0">
                  <div className="flex flex-wrap items-baseline gap-2">
                    <span className="font-sans text-foreground/90 break-words">
                      {q.text}
                    </span>
                    <Pill
                      label={q.intent}
                      tone={intentTone(q.intent)}
                    />
                  </div>
                  {q.rationale && (
                    <div className="mt-0.5 font-sans text-[11px] text-muted-foreground line-clamp-2">
                      {q.rationale}
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}

      {criteria.length > 0 && (
        <div className="border-t border-border pt-3">
          <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground mb-1.5">
            Success criteria
          </div>
          <ul className="ml-4 list-disc text-[12px] text-foreground/90 font-sans space-y-0.5">
            {criteria.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
