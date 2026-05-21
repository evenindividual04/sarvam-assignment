"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { listEvalRuns } from "@/lib/api";
import type { EvalRun } from "@/lib/types";
import { MetricBar } from "@/components/chat/metric-bar";
import { formatDateTime } from "@/lib/format";

export default function EvalListPage() {
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setErr(null);
    listEvalRuns()
      .then((r) => setRuns(r))
      .catch((e) => setErr(e instanceof Error ? e.message : "error"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, [load]);

  return (
    <div className="px-8 md:px-12 py-12 max-w-[1280px] mx-auto w-full">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-sans font-medium tracking-tight">
            Evaluation Runs
          </h1>
          <p className="mt-1 font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
            Aggregated scores across all completed runs
          </p>
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground hover:text-accent transition-colors disabled:opacity-40"
        >
          {loading ? "loading…" : "↻ refresh"}
        </button>
      </div>

      <div className="mt-10">
        {err && !loading && (
          <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
            backend unreachable.
          </div>
        )}
        {!loading && !err && runs.length === 0 && (
          <div className="font-mono text-[12px] text-subtle-foreground">
            No eval runs recorded. Run{" "}
            <code className="text-muted-foreground">
              python eval/eval_runner.py
            </code>
            .
          </div>
        )}

        {runs.length > 0 && (
          <div className="border-t border-border">
            <div className="grid grid-cols-[1.6fr_60px_80px_1fr_1fr_1fr_1fr_1fr_24px] gap-4 py-3 border-b border-border font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground items-end">
              <div>Run</div>
              <div className="text-right tabular-nums">N</div>
              <div className="text-right tabular-nums">Pass</div>
              <div>Faith</div>
              <div>Relv</div>
              <div>CtxP</div>
              <div>Cite</div>
              <div>ClmP</div>
              <div />
            </div>
            {runs.map((r) => {
              const passPct =
                (r.pass_rate <= 1 ? r.pass_rate * 100 : r.pass_rate);
              return (
                <Link
                  key={r.run_at}
                  href={`/eval/${encodeURIComponent(r.run_at)}`}
                  className="grid grid-cols-[1.6fr_60px_80px_1fr_1fr_1fr_1fr_1fr_24px] gap-4 py-4 border-b border-border hover:bg-surface-hover/50 transition-colors items-center"
                >
                  <div>
                    <div className="text-[13px] text-foreground">
                      {formatDateTime(r.run_at)}
                    </div>
                    <div className="font-mono text-[10px] text-subtle-foreground mt-0.5">
                      {r.run_at}
                    </div>
                  </div>
                  <div className="text-right font-mono tabular-nums text-[12px] text-muted-foreground">
                    {r.n_questions}
                  </div>
                  <div className="text-right font-mono tabular-nums text-[13px] text-foreground">
                    {passPct.toFixed(0)}%
                  </div>
                  <MetricBar label="" value={r.avg_faithfulness} compact />
                  <MetricBar label="" value={r.avg_relevance} compact />
                  <MetricBar label="" value={r.avg_context_precision} compact />
                  <MetricBar label="" value={r.avg_citation_integrity} compact />
                  <MetricBar label="" value={r.avg_claim_precision} compact />
                  <div className="text-right font-mono text-muted-foreground">
                    →
                  </div>
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
