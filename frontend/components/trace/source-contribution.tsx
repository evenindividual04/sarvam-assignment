// Token-share source contribution table.
//
// Anti-copying note: competitor uses `chunks_from_url / total_chunks` —
// a chunk-count proxy. Ours uses `tokens_from_url / total_context_tokens`
// (tiktoken cl100k_base), reflecting what the synthesizer actually saw.
// Visual style: monochrome bar + tabular numerals, no badge colors.

export interface SourceContributionItem {
  url: string;
  domain: string;
  title: string;
  tokens: number;
  share: number; // 0..1
  citations: number;
}

interface SourceContributionProps {
  contributions: SourceContributionItem[];
  totalTokens: number;
  /** Optional URL → role from the source_role event, rendered as a tiny label. */
  roleByUrl?: Readonly<Record<string, { role: string; confidence: number }>>;
}

const ROLE_SHORT: Record<string, string> = {
  primary_source: "primary",
  secondary_analysis: "analysis",
  statistical: "stats",
  news_event: "news",
  official: "official",
  encyclopedic: "ref",
  contradicting: "conflict",
  unclassified: "—",
};

export function SourceContribution({
  contributions,
  totalTokens,
  roleByUrl,
}: SourceContributionProps) {
  if (!contributions || contributions.length === 0) {
    return (
      <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
        No contribution data recorded.
      </div>
    );
  }

  const sorted = [...contributions].sort((a, b) => b.share - a.share);

  return (
    <div>
      <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground mb-3 flex items-baseline justify-between">
        <span>Token-share contribution</span>
        <span className="tabular-nums">
          {totalTokens.toLocaleString()} ctx tokens
        </span>
      </div>
      <ul className="divide-y divide-border/60">
        {sorted.map((row) => {
          const pct = Math.round(row.share * 100);
          const role = roleByUrl?.[row.url];
          return (
            <li key={row.url} className="py-2.5">
              <div className="flex items-baseline justify-between gap-3 mb-1">
                <a
                  href={row.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-mono text-[11px] text-accent hover:underline truncate"
                  title={row.title}
                >
                  {row.domain}
                </a>
                <span className="font-mono text-[11px] tabular-nums text-foreground shrink-0">
                  {pct}% · {row.tokens.toLocaleString()}t · {row.citations}{" "}
                  {row.citations === 1 ? "cite" : "cites"}
                </span>
              </div>
              <div
                className="h-[2px] bg-accent/70"
                style={{ width: `${Math.max(2, pct)}%` }}
                aria-hidden
              />
              <div className="flex items-baseline justify-between mt-1">
                <div
                  className="text-[12px] text-subtle-foreground truncate flex-1"
                  title={row.title}
                >
                  {row.title}
                </div>
                {role && (
                  <span
                    className="font-mono text-[10px] uppercase tracking-[0.10em] text-subtle-foreground shrink-0 pl-2"
                    title={`Role confidence: ${role.confidence.toFixed(2)}`}
                  >
                    {ROLE_SHORT[role.role] ?? role.role}
                  </span>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
