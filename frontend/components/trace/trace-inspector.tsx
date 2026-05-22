"use client";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { CodeBlock } from "./code-block";
import { LatencyBar } from "./latency-bar";
import { EvidenceLedger, type HopEvidence } from "./evidence-ledger";
import { ReasoningChip } from "./reasoning-chip";
import {
  SourceContribution,
  type SourceContributionItem,
} from "./source-contribution";
import { TerminatorChip } from "./terminator-chip";
import { MetricBar } from "@/components/chat/metric-bar";
import { ClaimsTable } from "@/components/eval/claims-table";
import { ProbePanel } from "@/components/eval/probe-panel";
import {
  formatMs,
  formatScore,
  formatTurnAsBibtex,
  formatTurnAsMarkdown,
  trustTierColor,
  trustTierLabel,
} from "@/lib/format";
import { costFor, formatCost } from "@/lib/cost";
import type {
  ClaimAuditRow,
  ContextSnippetRow,
  ContradictionProbeRow,
  DocMap,
  RunMetadata,
  StreamEvent,
  Turn,
} from "@/lib/types";

type ReasoningEvent = Extract<StreamEvent, { type: "reasoning" }>;

export interface TraceInspectorData {
  turn_id?: string;
  query?: string;
  planning_strategy?: string;
  selection_strategy?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  latency_ms?: number;
  planning_ms?: number;
  search_ms?: number;
  fetch_ms?: number;
  select_ms?: number;
  probe_ms?: number;
  synthesize_ms?: number;
  citation_integrity_score?: number;
  claim_precision_score?: number;
  urls?: string[];
  doc_map?: DocMap;
  context_xml?: string;
  claim_audit?: ClaimAuditRow[];
  contradiction_probes?: ContradictionProbeRow | null;
  // V3.8: which retrieval path actually ran for this turn (after capability
  // fallback, if any). Sourced from `run_metadata_json.retrieval_mode`.
  retrieval_mode_effective?: string;
  retrieval_mode_requested?: string;
  retrieval_mode_reason?: string | null;
  // V3.11: per-snippet trust signals from `turn_context`. When present,
  // takes precedence over `doc_map` in the Sources tab so tier dots can
  // be rendered alongside each domain.
  context_snippets?: ContextSnippetRow[];
  // Phase 4: extra fields needed for client-side Markdown / BibTeX export.
  // Optional so existing callers keep working; populated by `turnToTraceData`
  // and `doneToTraceData` when the source provides them.
  session_id?: string;
  created_at?: string;
  search_queries?: string[];
  response?: string;
  // Phase 1.5: surfaced from `run_metadata_json` so the trace inspector can
  // show which uncertainty branch fired and which follow-ups were proposed.
  uncertainty_kind?: "none" | "weak" | "missing" | "conflict" | null;
  follow_up_queries?: string[];
  evidence_gaps_reason?: string | null;
  // Phase 1.75: per-turn budget distribution + summarization fallbacks fired
  // during context assembly. Both are sourced from `run_metadata_json`.
  budget_distribution?: {
    system?: number;
    history?: number;
    web_context?: number;
    output_reserved?: number;
  };
  context_fallbacks?: string[];
  // Phase 1.875: agent-flow polish telemetry.
  numeric_grounding_ratio?: number | null;
  criteria_coverage?: boolean[];
  terminator_fired?: string | null;
  evidence_gaps?: Array<{ query: string; intent: string; reason: string }>;
  // Forensic-differentiation payloads (live-only for now; historic turns will
  // populate these from `run_metadata_json` once the backend persists them).
  hop_evidence?: HopEvidence[];
  source_contributions?: SourceContributionItem[];
  total_context_tokens?: number;
  source_roles?: Record<string, { role: string; confidence: number }>;
  terminator_reason?: string | null;
  terminator_hop?: number | null;
  terminator_detail?: string | null;
  // B3: live retrieval-grounded reasoning events for the current run.
  reasoning_events?: ReasoningEvent[];
}

interface TraceInspectorProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  data: TraceInspectorData | null;
}

const TAB_TRIGGER =
  "font-mono text-[11px] uppercase tracking-[0.14em] py-2.5 px-1 border-b-2 border-transparent text-muted-foreground hover:text-foreground data-[state=active]:border-accent data-[state=active]:text-foreground transition-colors bg-transparent rounded-none";

export function TraceInspector({
  open,
  onOpenChange,
  data,
}: TraceInspectorProps) {
  const turnShort = data?.turn_id?.slice(0, 8) ?? "—";
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-full sm:max-w-[560px] flex flex-col p-0 gap-0 bg-background border-l border-border"
      >
        <SheetHeader className="px-6 py-4 border-b border-border space-y-2">
          <SheetTitle className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
            Trace · <span className="text-foreground">{turnShort}</span>
          </SheetTitle>
          <SheetDescription className="font-sans text-[13px] text-muted-foreground line-clamp-1">
            {data?.query ?? "Run details for the most recent turn."}
          </SheetDescription>
          {data && (
            <div className="flex gap-2 pt-1">
              <ExportButton
                label="Export Markdown"
                onClick={() => exportTurn(data, "markdown")}
                disabled={!data.turn_id}
              />
              <ExportButton
                label="Export .bib"
                onClick={() => exportTurn(data, "bibtex")}
                disabled={!data.turn_id}
              />
            </div>
          )}
        </SheetHeader>

        {!data ? (
          <div className="p-10 font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground text-center">
            No turn selected.
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto">
            <Tabs defaultValue="overview" className="px-6 py-4">
              <TabsList className="bg-transparent p-0 h-auto border-b border-border w-full justify-start gap-6 rounded-none">
                <TabsTrigger value="overview" className={TAB_TRIGGER}>
                  Overview
                </TabsTrigger>
                <TabsTrigger value="sources" className={TAB_TRIGGER}>
                  Sources ({data.urls?.length ?? 0})
                </TabsTrigger>
                <TabsTrigger value="evidence" className={TAB_TRIGGER}>
                  Evidence ({data.hop_evidence?.length ?? 0})
                </TabsTrigger>
                <TabsTrigger value="context" className={TAB_TRIGGER}>
                  Context
                </TabsTrigger>
                <TabsTrigger value="claims" className={TAB_TRIGGER}>
                  Claims ({data.claim_audit?.length ?? 0})
                </TabsTrigger>
                <TabsTrigger value="probe" className={TAB_TRIGGER}>
                  Probe
                </TabsTrigger>
                <TabsTrigger value="uncertainty" className={TAB_TRIGGER}>
                  Uncertainty
                </TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="pt-6 space-y-6">
                <section>
                  <SectionLabel>Run</SectionLabel>
                  <div className="grid grid-cols-2 gap-x-6 gap-y-3">
                    <Field label="Planning" value={data.planning_strategy ?? "—"} />
                    <Field label="Selection" value={data.selection_strategy ?? "—"} />
                    <Field
                      label="Retrieval"
                      value={
                        data.retrieval_mode_effective
                          ? data.retrieval_mode_requested &&
                            data.retrieval_mode_requested !== data.retrieval_mode_effective
                            ? `${data.retrieval_mode_effective} (req: ${data.retrieval_mode_requested})`
                            : data.retrieval_mode_effective
                          : "—"
                      }
                      hint={data.retrieval_mode_reason ?? undefined}
                    />
                    <Field
                      label="Total"
                      value={formatMs(data.latency_ms)}
                      mono
                    />
                    <Field
                      label="Tokens"
                      value={`${data.prompt_tokens ?? 0} / ${data.completion_tokens ?? 0}`}
                      mono
                      hint="prompt / completion"
                    />
                    <Field
                      label="Cost"
                      value={`${formatCost(costFor(undefined, data.prompt_tokens, data.completion_tokens))} · ${data.prompt_tokens ?? 0} in · ${data.completion_tokens ?? 0} out`}
                      mono
                    />
                  </div>
                </section>

                <div className="border-t border-border" />

                <section>
                  <SectionLabel>Per-stage latency</SectionLabel>
                  <LatencyBar
                    planning_ms={data.planning_ms}
                    search_ms={data.search_ms}
                    fetch_ms={data.fetch_ms}
                    select_ms={data.select_ms}
                    probe_ms={data.probe_ms}
                    synthesize_ms={data.synthesize_ms}
                  />
                </section>

                <div className="border-t border-border" />

                <section className="space-y-3">
                  <SectionLabel>Quality</SectionLabel>
                  <MetricBar
                    label="Citation integrity"
                    value={data.citation_integrity_score}
                  />
                  <MetricBar
                    label="Claim precision"
                    value={data.claim_precision_score}
                  />
                  {data.numeric_grounding_ratio !== undefined &&
                    data.numeric_grounding_ratio !== null && (
                      <MetricBar
                        label="Numeric grounding"
                        value={data.numeric_grounding_ratio}
                      />
                    )}
                </section>

                {(data.terminator_fired ||
                  (data.criteria_coverage && data.criteria_coverage.length > 0) ||
                  (data.evidence_gaps && data.evidence_gaps.length > 0)) && (
                  <>
                    <div className="border-t border-border" />
                    <section className="space-y-3">
                      <SectionLabel>Agent flow</SectionLabel>
                      {data.terminator_fired && (
                        <Field
                          label="Terminator"
                          value={data.terminator_fired}
                          mono
                          hint="Why the multi-hop loop stopped"
                        />
                      )}
                      {data.criteria_coverage && data.criteria_coverage.length > 0 && (
                        <Field
                          label="Criteria coverage"
                          value={`${data.criteria_coverage.filter(Boolean).length} / ${data.criteria_coverage.length}`}
                          mono
                          hint="Success criteria satisfied (heuristic)"
                        />
                      )}
                      {data.evidence_gaps && data.evidence_gaps.length > 0 && (
                        <div>
                          <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground mb-1">
                            Evidence gaps
                          </div>
                          <ul className="space-y-1">
                            {data.evidence_gaps.map((g, i) => (
                              <li
                                key={`gap-${i}`}
                                className="font-mono text-[11px] text-foreground"
                              >
                                <span className="text-amber-600 mr-2">
                                  [{g.reason}]
                                </span>
                                <span className="text-subtle-foreground mr-1">
                                  {g.intent}:
                                </span>
                                {g.query}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </section>
                  </>
                )}
              </TabsContent>

              <TabsContent value="sources" className="pt-6">
                {data.context_snippets && data.context_snippets.length > 0 ? (
                  <>
                    {/* V3.11: each row shows a tier dot reflecting V2.3 source-
                       trust prior. Tier comes from the backend, computed via
                       `utils.source_trust.trust_for(domain)`. */}
                    <ul className="divide-y divide-border">
                      {data.context_snippets.map((s) => (
                        <li key={s.doc_id} className="py-3 flex items-start gap-3">
                          <span className="font-mono text-[10px] uppercase tracking-[0.10em] text-subtle-foreground w-12 shrink-0 pt-1">
                            {s.doc_id}
                          </span>
                          <span
                            className={`inline-block size-[8px] rounded-full shrink-0 mt-[7px] ${trustTierColor(s.trust_tier)}`}
                            title={`Trust: ${trustTierLabel(s.trust_tier)} (score ${formatScore(s.trust_score)})`}
                          />
                          <div className="flex-1 min-w-0">
                            <a
                              href={s.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="font-mono text-[11px] text-accent hover:underline truncate block"
                            >
                              {s.domain}
                            </a>
                            <div
                              className="text-[13px] text-foreground truncate"
                              title={s.title}
                            >
                              {s.title}
                            </div>
                          </div>
                        </li>
                      ))}
                    </ul>
                    <TrustLegend />
                  </>
                ) : data.doc_map && Object.keys(data.doc_map).length > 0 ? (
                  <ul className="divide-y divide-border">
                    {Object.entries(data.doc_map).map(([docId, tup]) => {
                      const [title, url, domain] = tup;
                      return (
                        <li key={docId} className="py-3 flex items-start gap-3">
                          <span className="font-mono text-[10px] uppercase tracking-[0.10em] text-subtle-foreground w-12 shrink-0 pt-0.5">
                            {docId}
                          </span>
                          <div className="flex-1 min-w-0">
                            <a
                              href={url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="font-mono text-[11px] text-accent hover:underline truncate block"
                            >
                              {domain}
                            </a>
                            <div
                              className="text-[13px] text-foreground truncate"
                              title={title}
                            >
                              {title}
                            </div>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                ) : data.urls && data.urls.length > 0 ? (
                  <ul className="divide-y divide-border">
                    {data.urls.map((u) => (
                      <li key={u} className="py-3">
                        <a
                          href={u}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="font-mono text-[12px] text-accent hover:underline truncate block"
                        >
                          {u}
                        </a>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
                    No sources recorded.
                  </div>
                )}
              </TabsContent>

              <TabsContent value="evidence" className="pt-6 space-y-6">
                {data.terminator_reason && (
                  <section className="pb-4 border-b border-border">
                    <TerminatorChip
                      reason={data.terminator_reason}
                      hop={data.terminator_hop ?? undefined}
                      detail={data.terminator_detail}
                    />
                  </section>
                )}
                <section>
                  <SectionLabel>Per-hop ledger</SectionLabel>
                  <EvidenceLedger hops={data.hop_evidence ?? []} />
                </section>
                {data.reasoning_events && data.reasoning_events.length > 0 && (
                  <>
                    <div className="border-t border-border" />
                    <section>
                      <SectionLabel>Reasoning</SectionLabel>
                      <ReasoningChip events={data.reasoning_events} />
                    </section>
                  </>
                )}
                <div className="border-t border-border" />
                <section>
                  <SectionLabel>Source contribution</SectionLabel>
                  <SourceContribution
                    contributions={data.source_contributions ?? []}
                    totalTokens={data.total_context_tokens ?? 0}
                    roleByUrl={data.source_roles}
                  />
                </section>
              </TabsContent>

              <TabsContent value="context" className="pt-6 space-y-5">
                <BudgetDistributionPanel
                  dist={data.budget_distribution}
                  fallbacks={data.context_fallbacks}
                />
                {data.context_xml ? (
                  <CodeBlock code={data.context_xml} language="xml" />
                ) : (
                  <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
                    No context XML recorded.
                  </div>
                )}
              </TabsContent>

              <TabsContent value="claims" className="pt-6">
                <ClaimsTable rows={data.claim_audit ?? []} />
              </TabsContent>

              <TabsContent value="probe" className="pt-6">
                <ProbePanel probe={data.contradiction_probes ?? null} />
              </TabsContent>

              <TabsContent value="uncertainty" className="pt-6 space-y-4">
                <section>
                  <SectionLabel>Signal</SectionLabel>
                  <Field
                    label="Kind"
                    value={data.uncertainty_kind ?? "none"}
                    mono
                  />
                  {data.evidence_gaps_reason && (
                    <div className="mt-3">
                      <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground mb-1">
                        Reason
                      </div>
                      <div className="text-[13px] text-foreground">
                        {data.evidence_gaps_reason}
                      </div>
                    </div>
                  )}
                </section>
                <div className="border-t border-border" />
                <section>
                  <SectionLabel>Suggested follow-ups</SectionLabel>
                  {data.follow_up_queries && data.follow_up_queries.length > 0 ? (
                    <ul className="space-y-1.5">
                      {data.follow_up_queries.map((q, i) => (
                        <li
                          key={`${i}-${q}`}
                          className="font-mono text-[12px] text-foreground"
                        >
                          <span className="text-subtle-foreground mr-2">
                            {i + 1}.
                          </span>
                          {q}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
                      No follow-ups recorded for this turn.
                    </div>
                  )}
                </section>
              </TabsContent>
            </Tabs>
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}

/**
 * Build a minimal `Turn`-shaped object from a `TraceInspectorData` and hand it
 * to the formatter. `formatTurn*` only reads typed fields, so missing optional
 * fields render as "Not recorded" / "None recorded".
 */
function traceDataToTurn(d: TraceInspectorData): Turn {
  return {
    turn_id: d.turn_id ?? "",
    session_id: d.session_id ?? "",
    query: d.query ?? "",
    response: d.response ?? "",
    created_at: d.created_at ?? "",
    search_queries: d.search_queries,
    urls_opened: d.urls,
    doc_map: d.doc_map,
    context_xml_sent: d.context_xml,
    prompt_tokens: d.prompt_tokens,
    completion_tokens: d.completion_tokens,
    latency_ms: d.latency_ms,
    citation_integrity_score: d.citation_integrity_score,
    claim_precision_score: d.claim_precision_score,
    selection_strategy: d.selection_strategy,
    planning_strategy: d.planning_strategy,
    planning_ms: d.planning_ms,
    search_ms: d.search_ms,
    fetch_ms: d.fetch_ms,
    select_ms: d.select_ms,
    probe_ms: d.probe_ms,
    synthesize_ms: d.synthesize_ms,
  };
}

/**
 * Client-side blob download. Matches the `URL.createObjectURL` + anchor-click
 * pattern; revokes the object URL on next tick so the download starts cleanly.
 */
function downloadBlob(filename: string, content: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function exportTurn(data: TraceInspectorData, kind: "markdown" | "bibtex"): void {
  const turn = traceDataToTurn(data);
  const shortId = (data.turn_id ?? "turn").slice(0, 8) || "turn";
  if (kind === "markdown") {
    downloadBlob(`turn-${shortId}.md`, formatTurnAsMarkdown(turn), "text/markdown;charset=utf-8");
  } else {
    downloadBlob(
      `turn-${shortId}.bib`,
      formatTurnAsBibtex(turn),
      "application/x-bibtex;charset=utf-8",
    );
  }
}

function ExportButton({
  label,
  onClick,
  disabled,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="font-mono text-[10px] uppercase tracking-[0.14em] px-2.5 py-1 border border-border text-muted-foreground hover:text-foreground hover:border-border-accent disabled:opacity-40 disabled:cursor-not-allowed transition-colors rounded-sm bg-transparent"
    >
      {label}
    </button>
  );
}

function TrustLegend() {
  const tiers: { tier: string; label: string }[] = [
    { tier: "tier_1_primary", label: "Primary (.gov / .edu / WHO)" },
    { tier: "tier_2_reference", label: "Reference (Nature, ArXiv, Wikipedia)" },
    { tier: "tier_3_journalism", label: "Journalism (Reuters, AP, BBC)" },
    { tier: "tier_4_mid", label: "Mid-tier (TechCrunch, Forbes)" },
    { tier: "tier_5_low", label: "Low (Medium, Substack, blogs)" },
    { tier: "unknown", label: "Unknown domain (default prior)" },
  ];
  return (
    <details className="mt-5 pt-4 border-t border-border">
      <summary className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground cursor-pointer hover:text-foreground transition-colors">
        Trust tier legend
      </summary>
      <ul className="mt-3 space-y-1.5">
        {tiers.map((t) => (
          <li key={t.tier} className="flex items-center gap-2">
            <span className={`inline-block size-[8px] rounded-full ${trustTierColor(t.tier)}`} />
            <span className="font-mono text-[11px] text-muted-foreground">
              {t.label}
            </span>
          </li>
        ))}
      </ul>
    </details>
  );
}

interface BudgetDistribution {
  system?: number;
  history?: number;
  web_context?: number;
  output_reserved?: number;
}

function BudgetDistributionPanel({
  dist,
  fallbacks,
}: {
  dist?: BudgetDistribution;
  fallbacks?: string[];
}) {
  const hasDist =
    dist &&
    (dist.system !== undefined ||
      dist.history !== undefined ||
      dist.web_context !== undefined ||
      dist.output_reserved !== undefined);
  const hasFallback = (fallbacks?.length ?? 0) > 0;
  if (!hasDist && !hasFallback) return null;

  const rows: Array<{ label: string; tokens: number }> = hasDist
    ? [
        { label: "System", tokens: dist?.system ?? 0 },
        { label: "History", tokens: dist?.history ?? 0 },
        { label: "Web context", tokens: dist?.web_context ?? 0 },
        { label: "Output reserved", tokens: dist?.output_reserved ?? 0 },
      ]
    : [];
  const total = rows.reduce((acc, r) => acc + r.tokens, 0) || 1;

  return (
    <section>
      <SectionLabel>Budget distribution</SectionLabel>
      {hasFallback && (
        <div className="mb-3 inline-flex items-center gap-2 rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1 font-mono text-[10px] uppercase tracking-[0.12em] text-amber-600 dark:text-amber-400">
          <span>⚠</span>
          <span>
            {(fallbacks ?? [])
              .map((f) => f.replace(/_/g, " "))
              .join(" · ")}
          </span>
        </div>
      )}
      {hasDist && (
        <table className="w-full font-mono tabular-nums text-[12px]">
          <tbody>
            {rows.map((r) => {
              const pct = total > 0 ? Math.round((r.tokens / total) * 100) : 0;
              return (
                <tr key={r.label} className="border-b border-border/50 last:border-0">
                  <td className="py-1.5 text-foreground">{r.label}</td>
                  <td className="py-1.5 text-right text-foreground">
                    {r.tokens.toLocaleString()}
                  </td>
                  <td className="py-1.5 pl-3 text-right text-subtle-foreground w-12">
                    {pct}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground mb-3">
      {children}
    </div>
  );
}

function Field({
  label,
  value,
  hint,
  mono,
}: {
  label: string;
  value: string;
  hint?: string;
  mono?: boolean;
}) {
  return (
    <div>
      <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground mb-1">
        {label}
      </div>
      <div
        className={
          mono
            ? "font-mono tabular-nums text-[13px] text-foreground"
            : "text-[13px] text-foreground truncate"
        }
      >
        {value}
        {hint && (
          <span className="ml-1.5 font-mono text-[10px] text-subtle-foreground">
            {hint}
          </span>
        )}
      </div>
    </div>
  );
}

// Helper for callers that have a `Turn` and want a value object suitable
// for `data` prop. Kept in this file for colocation.
export function turnToTraceData(t: {
  turn_id?: string;
  session_id?: string;
  created_at?: string;
  query?: string;
  response?: string;
  search_queries?: string[];
  context_xml_sent?: string;
  doc_map?: DocMap;
  urls_opened?: string[];
  prompt_tokens?: number;
  completion_tokens?: number;
  latency_ms?: number;
  citation_integrity_score?: number;
  claim_precision_score?: number;
  selection_strategy?: string;
  planning_strategy?: string;
  planning_ms?: number;
  search_ms?: number;
  fetch_ms?: number;
  select_ms?: number;
  probe_ms?: number;
  synthesize_ms?: number;
  claim_audit?: ClaimAuditRow[];
  contradiction_probes?: ContradictionProbeRow | null;
  context_snippets?: ContextSnippetRow[];
  run_metadata_json?: RunMetadata;
}): TraceInspectorData {
  // V3.8: extract effective retrieval mode from run_metadata if present.
  const meta: RunMetadata | undefined = t.run_metadata_json;
  const rm = meta?.retrieval_mode;
  return {
    turn_id: t.turn_id,
    session_id: t.session_id,
    created_at: t.created_at,
    query: t.query,
    response: t.response,
    search_queries: t.search_queries,
    context_xml: t.context_xml_sent,
    doc_map: t.doc_map,
    urls: t.urls_opened,
    prompt_tokens: t.prompt_tokens,
    completion_tokens: t.completion_tokens,
    latency_ms: t.latency_ms,
    citation_integrity_score: t.citation_integrity_score,
    claim_precision_score: t.claim_precision_score,
    selection_strategy: t.selection_strategy,
    planning_strategy: t.planning_strategy,
    planning_ms: t.planning_ms,
    search_ms: t.search_ms,
    fetch_ms: t.fetch_ms,
    select_ms: t.select_ms,
    probe_ms: t.probe_ms,
    synthesize_ms: t.synthesize_ms,
    claim_audit: t.claim_audit,
    contradiction_probes: t.contradiction_probes,
    context_snippets: t.context_snippets,
    retrieval_mode_effective: rm?.effective,
    retrieval_mode_requested: rm?.requested,
    retrieval_mode_reason: rm?.reason ?? null,
    uncertainty_kind: meta?.uncertainty_kind ?? null,
    follow_up_queries: meta?.follow_up_queries ?? [],
    evidence_gaps_reason: meta?.evidence_gaps_reason ?? null,
    budget_distribution: meta?.budget_distribution,
    context_fallbacks: meta?.context_fallbacks ?? [],
    numeric_grounding_ratio: meta?.numeric_grounding_ratio ?? null,
    criteria_coverage: meta?.criteria_coverage ?? [],
    terminator_fired: meta?.terminator_fired ?? null,
    evidence_gaps: meta?.evidence_gaps ?? [],
    // Forensic-differentiation fields, sourced from run_metadata_json when the
    // backend persists them. Safe empty defaults otherwise.
    hop_evidence: (meta?.hop_evidence as HopEvidence[] | undefined) ?? [],
    source_contributions:
      (meta?.source_contributions as SourceContributionItem[] | undefined) ?? [],
    total_context_tokens: meta?.total_context_tokens ?? 0,
    source_roles: meta?.source_roles ?? {},
    terminator_reason: meta?.terminator_fired ?? null,
    terminator_hop: meta?.terminator_hop ?? null,
    terminator_detail: meta?.terminator_detail ?? null,
  };
}

// Live forensic state captured from SSE events during the current turn.
// Wired by `page.tsx` from the `useSseResearch` hook and merged into the
// trace data when the user opens the inspector for a just-finished turn.
export interface LiveForensicState {
  hopEvidence?: HopEvidence[];
  sourceContribution?: {
    contributions: SourceContributionItem[];
    total_tokens: number;
  } | null;
  sourceRoles?: Record<string, { role: string; confidence: number }>;
  terminator?: {
    reason: string;
    hop?: number;
    detail?: string | null;
  } | null;
  reasoningEvents?: ReasoningEvent[];
}

export function doneToTraceData(
  query: string,
  d: {
    turn_id: string;
    answer?: string;
    context_xml: string;
    doc_map: DocMap;
    urls: string[];
    prompt_tokens: number;
    completion_tokens: number;
    latency_ms: number;
    citation_integrity_score: number;
    claim_precision_score: number;
    selection_strategy: string;
    planning_strategy: string;
    planning_ms: number;
    search_ms: number;
    fetch_ms: number;
    select_ms: number;
    probe_ms: number;
    synthesize_ms: number;
    run_metadata?: RunMetadata;
  },
  sessionId?: string,
  live?: LiveForensicState,
): TraceInspectorData {
  const rm = d.run_metadata?.retrieval_mode;
  const meta = d.run_metadata;
  return {
    turn_id: d.turn_id,
    session_id: sessionId,
    created_at: new Date().toISOString(),
    response: d.answer,
    query,
    context_xml: d.context_xml,
    doc_map: d.doc_map,
    urls: d.urls,
    prompt_tokens: d.prompt_tokens,
    completion_tokens: d.completion_tokens,
    latency_ms: d.latency_ms,
    citation_integrity_score: d.citation_integrity_score,
    claim_precision_score: d.claim_precision_score,
    selection_strategy: d.selection_strategy,
    planning_strategy: d.planning_strategy,
    planning_ms: d.planning_ms,
    search_ms: d.search_ms,
    fetch_ms: d.fetch_ms,
    select_ms: d.select_ms,
    probe_ms: d.probe_ms,
    synthesize_ms: d.synthesize_ms,
    retrieval_mode_effective: rm?.effective,
    retrieval_mode_requested: rm?.requested,
    retrieval_mode_reason: rm?.reason ?? null,
    uncertainty_kind: meta?.uncertainty_kind ?? null,
    follow_up_queries: meta?.follow_up_queries ?? [],
    evidence_gaps_reason: meta?.evidence_gaps_reason ?? null,
    budget_distribution: meta?.budget_distribution,
    context_fallbacks: meta?.context_fallbacks ?? [],
    numeric_grounding_ratio: meta?.numeric_grounding_ratio ?? null,
    criteria_coverage: meta?.criteria_coverage ?? [],
    terminator_fired: meta?.terminator_fired ?? null,
    evidence_gaps: meta?.evidence_gaps ?? [],
    // Forensic-differentiation: live state from SSE for the just-finished turn.
    // Historic turns fall through to empty defaults until the backend persists
    // these fields in run_metadata_json.
    hop_evidence: live?.hopEvidence ?? [],
    source_contributions: live?.sourceContribution?.contributions ?? [],
    total_context_tokens: live?.sourceContribution?.total_tokens ?? 0,
    source_roles: live?.sourceRoles ?? {},
    terminator_reason:
      live?.terminator?.reason ?? meta?.terminator_fired ?? null,
    terminator_hop: live?.terminator?.hop ?? null,
    terminator_detail: live?.terminator?.detail ?? null,
    reasoning_events: live?.reasoningEvents ?? [],
  };
}

export const _formatScore = formatScore;
