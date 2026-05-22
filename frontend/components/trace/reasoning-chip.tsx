// B3: retrieval-grounded reasoning chip.
//
// Renders two payload shapes emitted by the orchestrator per hop:
//   1. `phase: "intent"`      — planner-derived rationales (TypedQuery.rationale)
//   2. `phase: "observation"` — title/domain/score from selected chunks
//
// IMPORTANT (anti-CoT discipline): nothing rendered here is model-generated
// prose. `intent` strings come from the planner's structured JSON output,
// `observation` rows come from retrieved chunk metadata. We never render
// synthesizer chain-of-thought or scripted "Thought:" / "Action:" labels.

import type { StreamEvent } from "@/lib/types";

type ReasoningEvent = Extract<StreamEvent, { type: "reasoning" }>;

interface ReasoningChipProps {
  event?: ReasoningEvent;
  events?: ReasoningEvent[];
}

export function ReasoningChip({ event, events }: ReasoningChipProps) {
  if (events && events.length > 0) {
    return (
      <div className="space-y-2">
        {events.map((e, i) => (
          <ReasoningChip key={`r-${e.hop}-${e.phase}-${i}`} event={e} />
        ))}
      </div>
    );
  }
  if (!event) return null;
  if (event.phase === "intent") {
    const queries = event.queries ?? [];
    if (queries.length === 0) return null;
    return (
      <div className="rounded-md border border-neutral-200 bg-neutral-50 p-3 text-sm dark:border-neutral-800 dark:bg-neutral-900">
        <div className="mb-1.5 flex items-center gap-2">
          <span className="text-xs font-medium uppercase tracking-wide text-neutral-500">
            Reasoning · hop {event.hop} · intent
          </span>
        </div>
        <ul className="space-y-1.5">
          {queries.map((q, i) => (
            <li key={`${q.text}-${i}`} className="text-neutral-700 dark:text-neutral-300">
              <span className="font-mono text-xs text-neutral-500">[{q.intent}]</span>{" "}
              <span className="font-medium">{q.text}</span>
              {q.rationale ? (
                <div className="mt-0.5 pl-4 text-xs text-neutral-600 dark:text-neutral-400">
                  {q.rationale}
                </div>
              ) : null}
            </li>
          ))}
        </ul>
      </div>
    );
  }

  const observation = event.observation ?? [];
  if (observation.length === 0) return null;
  return (
    <div className="rounded-md border border-neutral-200 bg-neutral-50 p-3 text-sm dark:border-neutral-800 dark:bg-neutral-900">
      <div className="mb-1.5 flex items-center gap-2">
        <span className="text-xs font-medium uppercase tracking-wide text-neutral-500">
          Reasoning · hop {event.hop} · observation
        </span>
      </div>
      <ul className="space-y-1">
        {observation.map((o, i) => (
          <li
            key={`${o.url || o.title}-${i}`}
            className="flex items-baseline justify-between gap-2 text-neutral-700 dark:text-neutral-300"
          >
            <span className="truncate">
              <span className="font-medium">{o.title || "(untitled)"}</span>{" "}
              <span className="text-xs text-neutral-500">— {o.domain}</span>
            </span>
            <span className="font-mono text-xs tabular-nums text-neutral-500">
              {typeof o.score === "number" ? o.score.toFixed(3) : "—"}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
