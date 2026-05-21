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
import { MetricBar } from "@/components/chat/metric-bar";
import { formatMs, formatScore } from "@/lib/format";
import type { DocMap } from "@/lib/types";

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
        <SheetHeader className="px-6 py-4 border-b border-border space-y-1">
          <SheetTitle className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
            Trace · <span className="text-foreground">{turnShort}</span>
          </SheetTitle>
          <SheetDescription className="font-sans text-[13px] text-muted-foreground line-clamp-1">
            {data?.query ?? "Run details for the most recent turn."}
          </SheetDescription>
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
                <TabsTrigger value="context" className={TAB_TRIGGER}>
                  Context
                </TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="pt-6 space-y-6">
                <section>
                  <SectionLabel>Run</SectionLabel>
                  <div className="grid grid-cols-2 gap-x-6 gap-y-3">
                    <Field label="Planning" value={data.planning_strategy ?? "—"} />
                    <Field label="Selection" value={data.selection_strategy ?? "—"} />
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
                </section>
              </TabsContent>

              <TabsContent value="sources" className="pt-6">
                {data.doc_map && Object.keys(data.doc_map).length > 0 ? (
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

              <TabsContent value="context" className="pt-6">
                {data.context_xml ? (
                  <CodeBlock code={data.context_xml} language="xml" />
                ) : (
                  <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
                    No context XML recorded.
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </div>
        )}
      </SheetContent>
    </Sheet>
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
  query?: string;
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
}): TraceInspectorData {
  return {
    turn_id: t.turn_id,
    query: t.query,
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
  };
}

export function doneToTraceData(
  query: string,
  d: {
    turn_id: string;
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
  },
): TraceInspectorData {
  return {
    turn_id: d.turn_id,
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
  };
}

export const _formatScore = formatScore;
