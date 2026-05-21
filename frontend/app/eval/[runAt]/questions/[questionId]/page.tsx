"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { getQuestionDetail } from "@/lib/api";
import type { EvalQuestionDetail } from "@/lib/types";
import {
  Tabs,
  TabsList,
  TabsTrigger,
  TabsContent,
} from "@/components/ui/tabs";
import { CodeBlock } from "@/components/trace/code-block";
import { ClaimsTable } from "@/components/eval/claims-table";
import { ProbePanel } from "@/components/eval/probe-panel";
import { MetricBar } from "@/components/chat/metric-bar";
import { RichMarkdown } from "@/lib/markdown";
import {
  failureClassColor,
  formatMs,
  formatScore,
  truncate,
} from "@/lib/format";

interface PageProps {
  params: Promise<{ runAt: string; questionId: string }>;
}

const TAB_TRIGGER =
  "font-mono text-[11px] uppercase tracking-[0.14em] py-3 px-1 border-b-2 border-transparent text-muted-foreground hover:text-foreground data-[state=active]:border-accent data-[state=active]:text-foreground transition-colors bg-transparent rounded-none";

export default function QuestionDetailPage({ params }: PageProps) {
  const { runAt, questionId } = use(params);
  const decodedRunAt = decodeURIComponent(runAt);
  const decodedQuestionId = decodeURIComponent(questionId);

  const [detail, setDetail] = useState<EvalQuestionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    getQuestionDetail(decodedRunAt, decodedQuestionId)
      .then((d) => alive && setDetail(d))
      .catch((e) => alive && setErr(e instanceof Error ? e.message : "error"))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [decodedRunAt, decodedQuestionId]);

  return (
    <div className="px-8 md:px-12 py-12 max-w-[1024px] mx-auto w-full">
      {/* Breadcrumb */}
      <nav className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground mb-10">
        <Link
          href="/eval"
          className="hover:text-foreground transition-colors"
        >
          Eval
        </Link>
        <span className="mx-2">/</span>
        <Link
          href={`/eval/${encodeURIComponent(decodedRunAt)}`}
          className="hover:text-foreground transition-colors"
        >
          {truncate(decodedRunAt, 20)}
        </Link>
        <span className="mx-2">/</span>
        <span className="text-foreground">{decodedQuestionId}</span>
      </nav>

      {loading && (
        <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
          loading…
        </div>
      )}
      {err && !detail && (
        <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
          {err.includes("404")
            ? "question not found in this run."
            : "backend unreachable."}
        </div>
      )}

      {detail && (
        <>
          {/*
           * SECOND EDITORIAL MOMENT — the question rendered as a magazine
           * pull-quote. Instrument Serif italic appears here and ONLY here
           * (other than the chat empty-state hero).
           */}
          <div className="border-y border-border py-10 my-2">
            <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground mb-4">
              Question {decodedQuestionId}
            </div>
            <h1 className="font-display italic text-3xl md:text-4xl leading-tight tracking-tight text-foreground">
              <span className="text-subtle-foreground mr-1">“</span>
              {detail.question}
              <span className="text-subtle-foreground ml-1">”</span>
            </h1>
          </div>

          {/* Meta row */}
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-10 font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
            {detail.category && (
              <span className="text-foreground">{detail.category}</span>
            )}
            <span className="text-border-strong">·</span>
            {detail.failure_class && (
              <span
                className={`inline-block px-1.5 py-0.5 rounded-[3px] border ${failureClassColor(detail.failure_class)}`}
              >
                {detail.failure_class}
              </span>
            )}
            <span className="text-border-strong">·</span>
            {detail.latency_ms !== undefined && (
              <span className="tabular-nums">{formatMs(detail.latency_ms)}</span>
            )}
          </div>

          {/* Tabs — text-based underline */}
          <Tabs defaultValue="answer">
            <TabsList className="bg-transparent p-0 h-auto border-b border-border w-full justify-start gap-8 rounded-none">
              <TabsTrigger value="answer" className={TAB_TRIGGER}>
                Answer
              </TabsTrigger>
              <TabsTrigger value="context" className={TAB_TRIGGER}>
                Context
              </TabsTrigger>
              <TabsTrigger value="docmap" className={TAB_TRIGGER}>
                Doc Map
              </TabsTrigger>
              <TabsTrigger value="judge" className={TAB_TRIGGER}>
                Judge
              </TabsTrigger>
              <TabsTrigger value="claims" className={TAB_TRIGGER}>
                Claims
              </TabsTrigger>
              <TabsTrigger value="probe" className={TAB_TRIGGER}>
                Probe
              </TabsTrigger>
            </TabsList>

            <TabsContent value="answer" className="pt-8 pb-12 space-y-8">
              <div className="grid grid-cols-2 gap-8">
                <MetricBar
                  label="Citation integrity"
                  value={detail.citation_integrity}
                />
                <MetricBar
                  label="Claim precision"
                  value={detail.claim_precision}
                />
              </div>
              <div>
                <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground mb-4">
                  Agent answer
                </div>
                <RichMarkdown>{detail.agent_answer || "_No answer._"}</RichMarkdown>
              </div>
            </TabsContent>

            <TabsContent value="context" className="pt-8 pb-12">
              {detail.context_xml_sent ? (
                <CodeBlock
                  code={detail.context_xml_sent}
                  language="xml"
                  maxHeight="65vh"
                />
              ) : (
                <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
                  No context XML recorded.
                </div>
              )}
            </TabsContent>

            <TabsContent value="docmap" className="pt-8 pb-12">
              {detail.doc_map && Object.keys(detail.doc_map).length > 0 ? (
                <DocMapTable docMap={detail.doc_map} />
              ) : (
                <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
                  No doc map recorded.
                </div>
              )}
            </TabsContent>

            <TabsContent value="judge" className="pt-8 pb-12 space-y-8">
              <div className="space-y-4">
                <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground mb-2">
                  Judge scores
                </div>
                <MetricBar label="Faithfulness" value={detail.faithfulness} />
                <MetricBar
                  label="Answer relevance"
                  value={detail.answer_relevance}
                />
                <MetricBar
                  label="Context precision"
                  value={detail.context_precision}
                />
                <MetricBar
                  label="Citation integrity"
                  value={detail.citation_integrity}
                />
                <MetricBar
                  label="Claim precision"
                  value={detail.claim_precision}
                />
                {detail.conflict_adherence !== undefined && (
                  <MetricBar
                    label="Conflict adherence"
                    value={detail.conflict_adherence}
                  />
                )}
              </div>

              <div className="border-t border-border pt-6 grid grid-cols-2 sm:grid-cols-3 gap-x-8 gap-y-4">
                <Stat label="Faith" value={formatScore(detail.faithfulness)} />
                <Stat label="Relv" value={formatScore(detail.answer_relevance)} />
                <Stat
                  label="CtxPrec"
                  value={formatScore(detail.context_precision)}
                />
                <Stat label="Cite" value={formatScore(detail.citation_integrity)} />
                <Stat label="Claim" value={formatScore(detail.claim_precision)} />
                <Stat
                  label="Conflict"
                  value={formatScore(detail.conflict_adherence)}
                />
              </div>

              {detail.judge_reasoning && (
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground mb-3">
                    Judge reasoning
                  </div>
                  <div className="border border-border bg-surface rounded-[6px] p-5 text-[14px] leading-relaxed whitespace-pre-wrap text-foreground">
                    {detail.judge_reasoning}
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="claims" className="pt-8 pb-12">
              <ClaimsTable rows={detail.claim_audit ?? []} />
            </TabsContent>

            <TabsContent value="probe" className="pt-8 pb-12">
              <ProbePanel probe={detail.contradiction_probes ?? null} />
            </TabsContent>
          </Tabs>

          <div className="mt-10 pt-6 border-t border-border">
            <Link
              href={`/eval/${encodeURIComponent(decodedRunAt)}`}
              className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground hover:text-foreground transition-colors"
            >
              ← Back to run
            </Link>
          </div>
        </>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground">
        {label}
      </div>
      <div className="font-mono tabular-nums text-[15px] text-foreground mt-1">
        {value}
      </div>
    </div>
  );
}

const TIER_DOT: Record<string, string> = {
  tier_1: "bg-teal-500",
  tier_2: "bg-emerald-500",
  tier_3: "bg-sky-500",
  tier_4: "bg-amber-500",
  tier_5: "bg-orange-500",
};

function tierLabel(domain: string): { tier: string; cls: string } {
  // Heuristic — without backend tier signal we infer from domain.
  // Editorial: keep the dot present for visual rhythm, but mark unknown when uncertain.
  const d = domain.toLowerCase();
  if (
    /\b(gov|edu|nih|who|imf|worldbank|rbi|sec|europa\.eu)\b/.test(d) ||
    d.endsWith(".gov") ||
    d.endsWith(".edu")
  )
    return { tier: "tier_1", cls: TIER_DOT.tier_1 };
  if (
    /\b(arxiv|nature|science|acm|ieee|reuters|bloomberg|ft\.com|wsj|economist)\b/.test(
      d,
    )
  )
    return { tier: "tier_2", cls: TIER_DOT.tier_2 };
  if (
    /\b(techcrunch|wired|theverge|nytimes|bbc|cnn|guardian|washingtonpost)\b/.test(
      d,
    )
  )
    return { tier: "tier_3", cls: TIER_DOT.tier_3 };
  if (/\b(medium|substack|hackernews|news\.ycombinator)\b/.test(d))
    return { tier: "tier_4", cls: TIER_DOT.tier_4 };
  return { tier: "unknown", cls: "bg-zinc-700" };
}

function DocMapTable({
  docMap,
}: {
  docMap: Record<string, [string, string, string]>;
}) {
  return (
    <div className="border-t border-border">
      <div className="grid grid-cols-[60px_1fr_180px_24px] gap-4 py-3 border-b border-border font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
        <div>Doc</div>
        <div>Title</div>
        <div>Source</div>
        <div />
      </div>
      {Object.entries(docMap).map(([docId, tup]) => {
        const [title, url, domain] = tup;
        const t = tierLabel(domain);
        return (
          <a
            key={docId}
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="grid grid-cols-[60px_1fr_180px_24px] gap-4 py-3 border-b border-border hover:bg-surface-hover/50 transition-colors items-center"
          >
            <div className="font-mono text-[11px] text-muted-foreground">
              {docId}
            </div>
            <div
              className="text-[13px] text-foreground truncate"
              title={title}
            >
              {title}
            </div>
            <div className="flex items-center gap-2 min-w-0">
              <span className={`size-[6px] rounded-full shrink-0 ${t.cls}`} />
              <span className="font-mono text-[10px] uppercase tracking-[0.10em] text-muted-foreground shrink-0">
                {t.tier.replace("_", " ")}
              </span>
              <span className="font-mono text-[11px] text-muted-foreground truncate">
                {domain}
              </span>
            </div>
            <div className="text-right font-mono text-muted-foreground">↗</div>
          </a>
        );
      })}
    </div>
  );
}
