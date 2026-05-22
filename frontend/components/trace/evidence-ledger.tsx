// Forensic evidence ledger — renders per-hop hop_evidence events.
//
// Anti-copying note: this is NOT a "Found so far / Still needed" recap card.
// Each `grounded` row is a mechanical assertion: a specific entity/number/
// criterion token resolved to a real `doc_id` with a verbatim quote excerpt.
// Each `open` row is a planner success-criterion that did NOT get grounded
// this hop — derived by set difference, never by LLM recap.
//
// Visual style follows the trace-inspector's typographic audit-log feel:
// monospace headers, hairline rows, single accent. No traffic-light colors.

interface GroundedToken {
  token: string;
  kind: "entity" | "number" | "criterion";
  doc_id?: string;
  url?: string;
  quote?: string;
}

interface OpenCriterion {
  criterion: string;
  reason: "no_evidence" | "partial" | "conflicting";
}

export interface HopEvidence {
  hop: number;
  grounded: GroundedToken[];
  open: OpenCriterion[];
}

const KIND_LABEL: Record<GroundedToken["kind"], string> = {
  entity: "ent",
  number: "num",
  criterion: "crit",
};

const OPEN_REASON_LABEL: Record<OpenCriterion["reason"], string> = {
  no_evidence: "no evidence",
  partial: "partial match",
  conflicting: "conflicting",
};

interface EvidenceLedgerProps {
  hops: HopEvidence[];
}

export function EvidenceLedger({ hops }: EvidenceLedgerProps) {
  if (!hops || hops.length === 0) {
    return (
      <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
        No evidence ledger recorded.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {hops.map((hop) => (
        <HopBlock key={`hop-${hop.hop}`} hop={hop} />
      ))}
    </div>
  );
}

function HopBlock({ hop }: { hop: HopEvidence }) {
  const groundedCount = hop.grounded.length;
  const openCount = hop.open.length;

  return (
    <section>
      <div className="flex items-baseline justify-between border-b border-border pb-1.5 mb-3">
        <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-foreground">
          Hop {hop.hop}
        </span>
        <span className="font-mono text-[10px] tracking-[0.10em] text-subtle-foreground tabular-nums">
          {groundedCount} grounded · {openCount} open
        </span>
      </div>

      {groundedCount > 0 && (
        <div className="mb-4">
          <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-subtle-foreground mb-2">
            Grounded
          </div>
          <ul className="divide-y divide-border/60">
            {hop.grounded.map((row, i) => (
              <li
                key={`g-${hop.hop}-${i}`}
                className="py-2 flex items-start gap-3"
              >
                <span className="font-mono text-[10px] uppercase tracking-[0.10em] text-subtle-foreground shrink-0 w-9 pt-0.5">
                  {KIND_LABEL[row.kind]}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="font-mono text-[12px] text-foreground">
                    {row.token}
                  </div>
                  {row.quote && (
                    <div className="text-[12px] text-subtle-foreground italic leading-snug mt-0.5">
                      “{row.quote}”
                    </div>
                  )}
                </div>
                {row.doc_id && (
                  <span className="font-mono text-[10px] uppercase tracking-[0.10em] text-accent shrink-0 pt-0.5">
                    {row.doc_id}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {openCount > 0 && (
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-subtle-foreground mb-2">
            Open
          </div>
          <ul className="divide-y divide-border/60">
            {hop.open.map((row, i) => (
              <li key={`o-${hop.hop}-${i}`} className="py-2 flex items-start gap-3">
                <span className="font-mono text-[10px] uppercase tracking-[0.10em] text-subtle-foreground shrink-0 w-20 pt-0.5">
                  {OPEN_REASON_LABEL[row.reason]}
                </span>
                <div className="flex-1 text-[12px] text-foreground leading-snug">
                  {row.criterion}
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {groundedCount === 0 && openCount === 0 && (
        <div className="font-mono text-[10px] uppercase tracking-[0.10em] text-subtle-foreground">
          Hop produced no ledger entries.
        </div>
      )}
    </section>
  );
}
