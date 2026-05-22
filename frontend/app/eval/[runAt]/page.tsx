"use client";

import { use, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { getRunSummary, getRunQuestions } from "@/lib/api";
import type { EvalQuestion, EvalSummary } from "@/lib/types";
import { failureClassColor, formatDateTime, formatScore } from "@/lib/format";
import { costFor, formatCost } from "@/lib/cost";
import { ErrorPanel } from "@/components/shell/error-panel";
import { Skeleton } from "@/components/ui/skeleton";

interface PageProps {
  params: Promise<{ runAt: string }>;
}

export default function RunSummaryPage({ params }: PageProps) {
  const { runAt } = use(params);
  const decodedRunAt = decodeURIComponent(runAt);

  const [summary, setSummary] = useState<EvalSummary | null>(null);
  const [questions, setQuestions] = useState<EvalQuestion[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setErr(null);
    Promise.all([getRunSummary(decodedRunAt), getRunQuestions(decodedRunAt)])
      .then(([s, q]) => {
        setSummary(s);
        setQuestions(q);
      })
      .catch((e) => setErr(e instanceof Error ? e.message : "error"))
      .finally(() => setLoading(false));
  }, [decodedRunAt]);

  useEffect(() => {
    load();
  }, [load]);

  const failureChart = summary
    ? Object.entries(summary.failure_class_distribution).map(
        ([failure_class, count]) => ({ failure_class, count }),
      )
    : [];

  // pass_rate can be null on pre-V2.4 runs (no claim_precision tier yet) —
  // treat null/undefined/NaN as 0 so the hero number renders cleanly rather
  // than throwing TypeError: Cannot read properties of null (reading toFixed).
  const passPctRaw = summary?.pass_rate;
  const passPct =
    passPctRaw === null || passPctRaw === undefined || Number.isNaN(passPctRaw)
      ? 0
      : passPctRaw <= 1
        ? passPctRaw * 100
        : passPctRaw;

  return (
    <div className="px-8 md:px-12 py-12 max-w-[1280px] mx-auto w-full">
      <Link
        href="/eval"
        className="inline-block font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground hover:text-foreground transition-colors mb-6"
      >
        ← All runs
      </Link>

      <h1 className="text-2xl font-sans font-medium tracking-tight">
        Run · {formatDateTime(decodedRunAt)}
      </h1>
      <p className="mt-1 font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
        {decodedRunAt}
      </p>

      {loading && !summary && (
        <div className="mt-10 space-y-3">
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-48 w-full" />
        </div>
      )}
      {err && !loading && (
        err.includes("404") ? (
          <div className="mt-10 font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
            run not found.
          </div>
        ) : (
          <div className="mt-10">
            <ErrorPanel detail={err} onRetry={load} />
          </div>
        )
      )}

      {summary && (
        <>
          {/* Hero — pass-rate big mono number */}
          <div className="mt-12 mb-12 border-t border-b border-border py-10">
            <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
              Pass rate
            </div>
            <div className="mt-2 font-mono tabular-nums-lining text-6xl md:text-7xl text-foreground leading-none">
              {passPct.toFixed(0)}
              <span className="text-muted-foreground">%</span>
            </div>
            <div className="mt-3 font-mono text-[11px] text-muted-foreground tabular-nums-lining">
              {questions.length || Object.values(summary.failure_class_distribution).reduce((a, b) => a + b, 0)} questions evaluated
            </div>
          </div>

          {/* Metric strip */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-x-8 gap-y-6 mb-12">
            <Metric label="Faithfulness" value={summary.avg_faithfulness} />
            <Metric label="Relevance" value={summary.avg_relevance} />
            <Metric label="Ctx precision" value={summary.avg_context_precision} />
            <Metric label="Citation" value={summary.avg_citation_integrity} />
            <Metric label="Claim prec" value={summary.avg_claim_precision} />
          </div>

          {/* Tier A (Phase 1+): per-turn quality metrics promoted from run_metadata */}
          <PerTurnQualityCard summary={summary} />

          {/* Cross-language consistency (en/hi) */}
          <CrossLanguageCard summary={summary} />

          {/* Calibration — confidence vs faithfulness (V3.5) */}
          <CalibrationStrip summary={summary} />

          {/* Failure-class legend — collapsed by default so it doesn't crowd
              the charts. Expanding it explains what HALLUCINATION_FACT vs
              KNOWLEDGE_BLEED actually mean for reviewers outside the codebase. */}
          <FailureClassLegend />

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12">
            <ChartCard title="Failure class distribution">
              <ResponsiveContainer>
                <BarChart data={failureChart}>
                  <CartesianGrid
                    strokeDasharray="2 2"
                    stroke="rgba(128,128,128,0.20)"
                  />
                  <XAxis
                    dataKey="failure_class"
                    stroke="#6b6b73"
                    fontSize={10}
                    tickLine={false}
                    axisLine={false}
                    angle={-12}
                    textAnchor="end"
                    height={50}
                    tick={{ fontFamily: "var(--font-mono)" }}
                  />
                  <YAxis
                    stroke="#6b6b73"
                    fontSize={10}
                    tickLine={false}
                    axisLine={false}
                    allowDecimals={false}
                    tick={{ fontFamily: "var(--font-mono)" }}
                  />
                  <Tooltip
                    cursor={{ fill: "rgba(255, 255, 255, 0.04)" }}
                    contentStyle={tooltipStyle}
                    labelStyle={tooltipLabelStyle}
                  />
                  <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                    {failureChart.map((d) => (
                      <Cell
                        key={d.failure_class}
                        fill={
                          d.failure_class === "PASS"
                            ? "rgba(13, 148, 136, 0.85)"
                            : "rgba(13, 148, 136, 0.30)"
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Per-category averages">
              <ResponsiveContainer>
                <BarChart data={summary.by_category}>
                  <CartesianGrid
                    strokeDasharray="2 2"
                    stroke="rgba(128,128,128,0.20)"
                  />
                  <XAxis
                    dataKey="category"
                    stroke="#6b6b73"
                    fontSize={10}
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontFamily: "var(--font-mono)" }}
                  />
                  <YAxis
                    stroke="#6b6b73"
                    fontSize={10}
                    tickLine={false}
                    axisLine={false}
                    domain={[0, 1]}
                    tick={{ fontFamily: "var(--font-mono)" }}
                  />
                  <Tooltip
                    cursor={{ fill: "rgba(255, 255, 255, 0.04)" }}
                    contentStyle={tooltipStyle}
                    labelStyle={tooltipLabelStyle}
                  />
                  <Bar
                    dataKey="faithfulness"
                    fill="rgba(13, 148, 136, 0.95)"
                    radius={[2, 2, 0, 0]}
                  />
                  <Bar
                    dataKey="answer_relevance"
                    fill="rgba(13, 148, 136, 0.70)"
                    radius={[2, 2, 0, 0]}
                  />
                  <Bar
                    dataKey="citation_integrity"
                    fill="rgba(13, 148, 136, 0.50)"
                    radius={[2, 2, 0, 0]}
                  />
                  <Bar
                    dataKey="claim_precision"
                    fill="rgba(13, 148, 136, 0.30)"
                    radius={[2, 2, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          {/* Questions table */}
          <div>
            <div className="flex items-baseline justify-between mb-3">
              <h2 className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                Questions
              </h2>
              <span className="font-mono text-[11px] tabular-nums text-subtle-foreground">
                {questions.length}
              </span>
            </div>
            <div className="border-t border-border">
              <div className="grid grid-cols-[60px_1fr_100px_36px_60px_60px_60px_60px_70px_120px_24px] gap-4 py-3 border-b border-border font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
                <div>ID</div>
                <div>Question</div>
                <div>Category</div>
                <div>Lang</div>
                <div className="text-right">Faith</div>
                <div className="text-right">Relv</div>
                <div className="text-right">Cite</div>
                <div className="text-right">Claim</div>
                <div className="text-right">Cost</div>
                <div>Failure</div>
                <div />
              </div>
              {questions.map((q) => (
                <Link
                  key={q.question_id}
                  href={`/eval/${encodeURIComponent(decodedRunAt)}/questions/${encodeURIComponent(q.question_id)}`}
                  className="grid grid-cols-[60px_1fr_100px_36px_60px_60px_60px_60px_70px_120px_24px] gap-4 py-3 border-b border-border hover:bg-surface-hover/50 transition-colors items-center"
                >
                  <div className="font-mono text-[11px] text-muted-foreground">
                    {q.question_id}
                  </div>
                  <div className="text-[13px] truncate">{q.question}</div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.10em] text-muted-foreground">
                    {q.category}
                  </div>
                  <div>
                    <span className="inline-block font-mono text-[10px] uppercase tracking-[0.10em] px-1.5 py-0.5 rounded-[3px] border border-border text-muted-foreground">
                      {(q.language ?? "en").toUpperCase()}
                    </span>
                  </div>
                  <div className="text-right font-mono tabular-nums text-[12px] text-foreground">
                    {formatScore(q.faithfulness)}
                  </div>
                  <div className="text-right font-mono tabular-nums text-[12px] text-foreground">
                    {formatScore(q.answer_relevance)}
                  </div>
                  <div className="text-right font-mono tabular-nums text-[12px] text-foreground">
                    {formatScore(q.citation_integrity)}
                  </div>
                  <div className="text-right font-mono tabular-nums text-[12px] text-foreground">
                    {formatScore(q.claim_precision)}
                  </div>
                  <div className="text-right font-mono tabular-nums text-[11px] text-muted-foreground">
                    {formatCost(costFor(undefined, q.prompt_tokens, q.completion_tokens))}
                  </div>
                  <div>
                    <span
                      className={`inline-block font-mono text-[10px] uppercase tracking-[0.10em] px-1.5 py-0.5 rounded-[3px] border ${failureClassColor(q.failure_class)}`}
                    >
                      {q.failure_class ?? "—"}
                    </span>
                  </div>
                  <div className="text-right font-mono text-muted-foreground">
                    →
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

const tooltipStyle: React.CSSProperties = {
  background: "var(--popover)",
  color: "var(--popover-foreground)",
  border: "1px solid var(--border-strong)",
  borderRadius: 6,
  fontSize: 11,
  fontFamily: "var(--font-mono)",
  padding: "6px 8px",
};

const tooltipLabelStyle: React.CSSProperties = {
  color: "var(--muted-foreground)",
  fontSize: 10,
  textTransform: "uppercase",
  letterSpacing: "0.12em",
};

function ChartCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="border border-border rounded-[8px] bg-surface p-5">
      <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground mb-4">
        {title}
      </div>
      <div className="h-56">{children}</div>
    </div>
  );
}

function CalibrationStrip({ summary }: { summary: EvalSummary }) {
  const buckets = summary.calibration?.buckets ?? [];
  const correlation =
    summary.calibration?.correlation ??
    summary.run_summary?.calibration_correlation ??
    null;
  const ordered = ["low", "medium", "high"].map((c) =>
    buckets.find((b) => b.confidence === c),
  );
  const enough = ordered.every((b) => b && b.n >= 3);
  if (!enough) {
    return (
      <div className="mb-12 border-t border-b border-border py-6">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          Confidence calibration
        </div>
        <div className="mt-2 font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
          insufficient data for calibration.
        </div>
      </div>
    );
  }
  const data = ordered.map((b) => ({
    confidence: b!.confidence.toUpperCase(),
    faithfulness: b!.mean_faithfulness ?? 0,
    n: b!.n,
  }));
  return (
    <div className="mb-12 border-t border-b border-border py-6">
      <div className="flex items-baseline justify-between mb-3">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          Confidence calibration
        </div>
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground tabular-nums">
          CORR: {correlation === null ? "—" : correlation.toFixed(2)}
        </div>
      </div>
      <div className="h-20">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid
              strokeDasharray="2 2"
              stroke="rgba(128,128,128,0.20)"
              vertical={false}
            />
            <XAxis
              dataKey="confidence"
              stroke="#6b6b73"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              tick={{ fontFamily: "var(--font-mono)" }}
            />
            <YAxis
              stroke="#6b6b73"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              domain={[0, 1]}
              tick={{ fontFamily: "var(--font-mono)" }}
              width={28}
            />
            <Tooltip
              cursor={{ stroke: "rgba(255,255,255,0.08)" }}
              contentStyle={tooltipStyle}
              labelStyle={tooltipLabelStyle}
            />
            <Line
              type="monotone"
              dataKey="faithfulness"
              stroke="rgba(13, 148, 136, 0.6)"
              strokeWidth={1.5}
              dot={{ fill: "rgba(13, 148, 136, 0.6)", r: 3 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/**
 * Tier A (Phase 1+): renders the three per-turn quality ratios — quote
 * grounding, numeric grounding, and criteria coverage. Hidden when all three
 * are null so older runs (pre-migration) don't show an empty card.
 */
function PerTurnQualityCard({ summary }: { summary: EvalSummary }) {
  const ptq = summary.per_turn_quality;
  if (!ptq) return null;
  const items: { label: string; value: number | null | undefined }[] = [
    { label: "Quote grounding", value: ptq.mean_quote_grounding_ratio },
    { label: "Numeric grounding", value: ptq.mean_numeric_grounding_ratio },
    { label: "Criteria coverage", value: ptq.mean_criteria_coverage_ratio },
  ];
  if (items.every((it) => it.value === null || it.value === undefined)) return null;
  return (
    <div className="mb-12 border-t border-b border-border py-6">
      <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground mb-4">
        Per-turn quality (Phase 1+)
      </div>
      <div className="grid grid-cols-3 gap-8">
        {items.map((it) => (
          <div key={it.label}>
            <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-2">
              {it.label}
            </div>
            <div className="font-mono tabular-nums text-2xl text-foreground leading-none">
              {it.value === null || it.value === undefined
                ? "—"
                : it.value.toFixed(2)}
            </div>
            <div className="mt-2 h-[2px] w-full bg-border">
              <div
                className="h-full bg-accent"
                style={{
                  width: `${
                    it.value === null || it.value === undefined
                      ? 0
                      : Math.max(0, Math.min(100, it.value * 100))
                  }%`,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Cross-language consistency strip. Each row is an en/hi question pair on the
 * same concept; Jaccard score measures key-claim overlap. Hidden when no
 * cross-language pairs were evaluated (English-only runs).
 */
function CrossLanguageCard({ summary }: { summary: EvalSummary }) {
  const cl = summary.cross_language;
  if (!cl || !cl.rows || cl.rows.length === 0) return null;
  const meanJaccard = cl.mean_jaccard ?? 0;
  const flaggedCount = cl.rows.filter((r) => r.flagged_inconsistent).length;
  return (
    <div className="mb-12 border-t border-b border-border py-6">
      <div className="flex items-baseline justify-between mb-4">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          Cross-language consistency (en ↔ hi)
        </div>
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground tabular-nums">
          MEAN JACCARD: {meanJaccard.toFixed(2)} · FLAGGED: {flaggedCount}/{cl.rows.length}
        </div>
      </div>
      <div className="border-t border-border">
        <div className="grid grid-cols-[1fr_80px_80px_80px_60px] gap-4 py-2 border-b border-border font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
          <div>Concept</div>
          <div className="text-right">EN qid</div>
          <div className="text-right">HI qid</div>
          <div className="text-right">Jaccard</div>
          <div className="text-right">Flag</div>
        </div>
        {cl.rows.map((r) => (
          <div
            key={r.concept_id}
            className="grid grid-cols-[1fr_80px_80px_80px_60px] gap-4 py-2 border-b border-border items-center font-mono text-[11px]"
          >
            <div className="text-foreground">{r.concept_id}</div>
            <div className="text-right text-muted-foreground tabular-nums">{r.en_question_id}</div>
            <div className="text-right text-muted-foreground tabular-nums">{r.hi_question_id}</div>
            <div className="text-right text-foreground tabular-nums">
              {r.jaccard_score === null || r.jaccard_score === undefined
                ? "—"
                : r.jaccard_score.toFixed(2)}
            </div>
            <div className="text-right">
              {r.flagged_inconsistent ? (
                <span className="text-[#dc2626]">⚠</span>
              ) : (
                <span className="text-subtle-foreground">·</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number | null | undefined }) {
  // Same null-aware treatment as MetricBar: pre-V2.4 runs return null for
  // metrics that didn't exist on that run. Render "—" instead of crashing.
  const missing =
    value === null || value === undefined || Number.isNaN(value);
  const v = missing ? undefined : (value as number);
  const norm = v === undefined ? 0 : v <= 1 ? v : v / 100;
  return (
    <div>
      <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-2">
        {label}
      </div>
      <div className="font-mono tabular-nums text-3xl text-foreground leading-none">
        {missing ? "—" : norm.toFixed(2)}
      </div>
      <div className="mt-2 h-[2px] w-full bg-border">
        <div
          className="h-full bg-accent"
          style={{ width: `${Math.max(0, Math.min(100, norm * 100))}%` }}
        />
      </div>
    </div>
  );
}

/**
 * Six-row legend explaining what each failure class catches. Collapsed by
 * default so the dashboard isn't visually crowded; reviewers outside the
 * codebase open it to learn what HALLUCINATION_FACT etc. actually mean.
 *
 * Classes are emitted by `eval/judge.py:classify_failure` based on the
 * judges' per-metric votes for a given question.
 */
function FailureClassLegend() {
  const items: { fc: string; explanation: string }[] = [
    { fc: "PASS", explanation: "Judges agreed the answer is grounded, relevant, and well-cited." },
    { fc: "HALLUCINATION_FACT", explanation: "A claim asserted in the answer is not present in the retrieved context." },
    { fc: "HALLUCINATION_ATTRIBUTION", explanation: "A citation points to a doc that doesn’t support the claim it’s attached to." },
    { fc: "KNOWLEDGE_BLEED", explanation: "The answer leaned on the model’s training-data prior instead of the retrieved context." },
    { fc: "RETRIEVAL_FAILURE", explanation: "Retrieval didn’t surface the relevant information; synthesis was set up to fail." },
    { fc: "CONFLICT_MISS", explanation: "Sources disagreed but the agent picked a side without surfacing the disagreement." },
    { fc: "COHERENCE_FAIL", explanation: "A multi-turn follow-up lost the prior turn’s context." },
  ];
  return (
    <details className="mb-6 border border-border rounded-[6px] bg-surface px-5 py-3">
      <summary className="cursor-pointer font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground hover:text-foreground transition-colors">
        Failure class legend — what each class means
      </summary>
      <ul className="mt-4 space-y-2.5">
        {items.map(({ fc, explanation }) => (
          <li key={fc} className="flex items-start gap-3">
            <span className={`inline-block px-1.5 py-0.5 rounded-[3px] border font-mono text-[10px] uppercase tracking-[0.10em] shrink-0 ${failureClassColor(fc)}`}>
              {fc}
            </span>
            <span className="text-[12px] text-muted-foreground leading-normal max-w-prose">
              {explanation}
            </span>
          </li>
        ))}
      </ul>
    </details>
  );
}
