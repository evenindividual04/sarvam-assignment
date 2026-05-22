"use client";

import type { ConflictKind, ContradictionProbeRow } from "@/lib/types";
import { MetricBar } from "@/components/chat/metric-bar";

interface ProbePanelProps {
  probe: ContradictionProbeRow | null | undefined;
}

const SKIP_EXPLAIN: Record<string, string> = {
  timeout: "The contradiction-probe LLM call did not return in time.",
  parse_fail: "The probe response could not be parsed as valid JSON.",
  lt_2_chunks:
    "Probing was skipped because there were fewer than two distinct sources to compare.",
  breaker_open:
    "The circuit breaker for this provider was open. Probing was skipped to avoid cascading failures.",
};

// DRAGged-into-Conflict taxonomy (Cattan et al., Google, arXiv:2506.08500).
// Three kinds with explicit reader-facing copy so the inspector teaches the
// taxonomy without naming the paper inline.
const KIND_LABEL: Record<ConflictKind, string> = {
  self: "internal contradiction",
  pair: "sources disagree",
  conditional: "agree under qualifier",
  none: "no conflict",
};

const KIND_HINT: Record<ConflictKind, string> = {
  self: "A single source contradicts itself.",
  pair: "Two sources disagree on a fact; the answer cites both.",
  conditional:
    "Sources agree once a qualifier (time, region, sub-domain) is applied.",
  none: "",
};

export function ProbePanel({ probe }: ProbePanelProps) {
  if (!probe) {
    return (
      <div className="border border-dashed border-border rounded-[8px] p-10 text-center font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
        No contradiction probe recorded.
      </div>
    );
  }

  if (probe.skip_reason) {
    return (
      <div className="border border-border rounded-[8px] bg-surface p-6">
        <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-amber-400 mb-2">
          Probe skipped · {probe.skip_reason}
        </div>
        <p className="text-[13px] text-muted-foreground leading-relaxed">
          {SKIP_EXPLAIN[probe.skip_reason] ?? "The probe stage was skipped."}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="border border-border rounded-[8px] bg-surface p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              Probe result
            </span>
            <span
              className={
                probe.has_conflict
                  ? "font-mono text-[10px] uppercase tracking-[0.10em] px-1.5 py-0.5 rounded-[3px] border bg-red-950/40 text-red-300 border-red-900/40"
                  : "font-mono text-[10px] uppercase tracking-[0.10em] px-1.5 py-0.5 rounded-[3px] border bg-emerald-950/40 text-emerald-300 border-emerald-900/40"
              }
            >
              {probe.has_conflict ? "Conflict found" : "No conflict"}
            </span>
            {probe.dominant_kind && probe.dominant_kind !== "none" && (
              <span
                className="font-mono text-[10px] uppercase tracking-[0.10em] px-1.5 py-0.5 rounded-[3px] border border-border text-foreground"
                title={KIND_HINT[probe.dominant_kind]}
              >
                {KIND_LABEL[probe.dominant_kind]}
              </span>
            )}
          </div>
        </div>
        <MetricBar label="Probe confidence" value={probe.confidence} />
        {probe.dominant_kind && probe.dominant_kind !== "none" && (
          <p className="mt-3 text-[12px] text-subtle-foreground leading-snug">
            {KIND_HINT[probe.dominant_kind]}
          </p>
        )}
      </div>

      {(probe.contradictions ?? []).map((c, i) => (
        <div
          key={i}
          className="border border-border rounded-[8px] overflow-hidden bg-surface"
        >
          {probe.is_temporal_evolution && (
            <div className="bg-amber-950/30 border-b border-amber-900/30 text-amber-300 font-mono text-[10px] uppercase tracking-[0.14em] px-4 py-1.5">
              Temporal evolution · not a contradiction
            </div>
          )}
          {c.topic && (
            <div className="px-5 py-2 border-b border-border font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
              {c.topic}
            </div>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-border">
            <div className="p-5">
              <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-2">
                Position A
                {(c.position_a_doc_ids ?? []).length > 0 && (
                  <span className="ml-2 text-subtle-foreground normal-case tracking-normal">
                    · {(c.position_a_doc_ids ?? []).join(", ")}
                  </span>
                )}
              </div>
              <p className="text-[14px] leading-relaxed text-foreground">
                {c.position_a}
              </p>
            </div>
            <div className="p-5">
              <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-2">
                Position B
                {(c.position_b_doc_ids ?? []).length > 0 && (
                  <span className="ml-2 text-subtle-foreground normal-case tracking-normal">
                    · {(c.position_b_doc_ids ?? []).join(", ")}
                  </span>
                )}
              </div>
              <p className="text-[14px] leading-relaxed text-foreground">
                {c.position_b}
              </p>
            </div>
          </div>
          <div className="px-5 py-2 border-t border-border flex items-baseline justify-center gap-3 font-mono text-[10px] uppercase tracking-[0.18em]">
            <span className="text-accent">
              {c.kind ? KIND_LABEL[c.kind] : "sources disagree"}
            </span>
            {c.qualifier && (
              <span className="text-subtle-foreground tracking-[0.10em] normal-case font-normal text-[11px]">
                under qualifier: <em>{c.qualifier}</em>
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
