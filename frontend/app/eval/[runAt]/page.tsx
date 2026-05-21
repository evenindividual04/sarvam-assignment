"use client";

import { use, useEffect, useState } from "react";
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

  useEffect(() => {
    let alive = true;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    Promise.all([getRunSummary(decodedRunAt), getRunQuestions(decodedRunAt)])
      .then(([s, q]) => {
        if (!alive) return;
        setSummary(s);
        setQuestions(q);
      })
      .catch((e) => alive && setErr(e instanceof Error ? e.message : "error"))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [decodedRunAt]);

  const failureChart = summary
    ? Object.entries(summary.failure_class_distribution).map(
        ([failure_class, count]) => ({ failure_class, count }),
      )
    : [];

  const passPct = summary
    ? summary.pass_rate <= 1
      ? summary.pass_rate * 100
      : summary.pass_rate
    : 0;

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

      {loading && (
        <div className="mt-10 font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
          loading…
        </div>
      )}
      {err && (
        <div className="mt-10 font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
          {err.includes("404") ? "run not found." : "backend unreachable."}
        </div>
      )}

      {summary && (
        <>
          {/* Hero — pass-rate big mono number */}
          <div className="mt-12 mb-12 border-t border-b border-border py-10">
            <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
              Pass rate
            </div>
            <div className="mt-2 font-mono tabular-nums text-6xl md:text-7xl text-foreground leading-none">
              {passPct.toFixed(0)}
              <span className="text-muted-foreground">%</span>
            </div>
            <div className="mt-3 font-mono text-[11px] text-muted-foreground tabular-nums">
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

          {/* Calibration — confidence vs faithfulness (V3.5) */}
          <CalibrationStrip summary={summary} />

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12">
            <ChartCard title="Failure class distribution">
              <ResponsiveContainer>
                <BarChart data={failureChart}>
                  <CartesianGrid
                    strokeDasharray="2 2"
                    stroke="rgba(255,255,255,0.06)"
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
                    stroke="rgba(255,255,255,0.06)"
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
  background: "#111114",
  border: "1px solid rgba(255, 255, 255, 0.12)",
  borderRadius: 6,
  fontSize: 11,
  fontFamily: "var(--font-mono)",
  padding: "6px 8px",
};

const tooltipLabelStyle: React.CSSProperties = {
  color: "#9b9ba3",
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
              stroke="rgba(255,255,255,0.06)"
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

function Metric({ label, value }: { label: string; value: number | undefined }) {
  const v = value === undefined || Number.isNaN(value) ? undefined : value;
  const norm = v === undefined ? 0 : v <= 1 ? v : v / 100;
  return (
    <div>
      <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-2">
        {label}
      </div>
      <div className="font-mono tabular-nums text-3xl text-foreground leading-none">
        {v === undefined ? "—" : norm.toFixed(2)}
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
