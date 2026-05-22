// Hop-loop terminator chip — surfaces *why* the multi-hop loop stopped.
//
// Anti-copying note: competitor uses a hardcoded `MAX_HOPS=3`. We surface
// adaptive reasons (EVIDENCE_SUFFICIENT, MARGINAL_GAIN_LOW, CRITERIA_SATISFIED, …)
// so the reader sees the agent stopped *for a reason*, not because a counter
// ran out. Visual style: single typographic line, no badge, no color coding.

const REASON_LABEL: Record<string, string> = {
  MAX_HOPS_REACHED: "max hops reached",
  EVIDENCE_SUFFICIENT: "evidence sufficient",
  MARGINAL_GAIN_LOW: "marginal gain low",
  NO_NEW_QUERIES: "no new queries",
  BUDGET_EXHAUSTED: "budget exhausted",
  CRITERIA_SATISFIED: "criteria satisfied",
};

interface TerminatorChipProps {
  reason?: string | null;
  hop?: number;
  detail?: string | null;
}

export function TerminatorChip({ reason, hop, detail }: TerminatorChipProps) {
  if (!reason) return null;
  const label = REASON_LABEL[reason] ?? reason.toLowerCase().replace(/_/g, " ");

  return (
    <div className="flex items-baseline gap-3 text-[12px]">
      <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground">
        Stopped
      </span>
      <span className="font-mono text-foreground">{label}</span>
      {typeof hop === "number" && (
        <span className="font-mono text-[10px] tabular-nums text-subtle-foreground">
          @ hop {hop}
        </span>
      )}
      {detail && (
        <span className="text-subtle-foreground truncate">{detail}</span>
      )}
    </div>
  );
}
