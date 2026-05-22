"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { listEvalRuns } from "@/lib/api";
import { BACKEND } from "@/lib/api";
import type { EvalRun } from "@/lib/types";
import { MetricBar } from "@/components/chat/metric-bar";
import { Button } from "@/components/ui/button";
import { ErrorPanel } from "@/components/shell/error-panel";
import { Skeleton } from "@/components/ui/skeleton";
import { formatDateTime } from "@/lib/format";
import { toast } from "sonner";

type SmokeState =
  | { status: "idle" }
  | { status: "running"; elapsed: number; completed: number; total: number; label?: string }
  | { status: "done"; runAt: string }
  | { status: "error"; message: string };

export default function EvalListPage() {
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [smoke, setSmoke] = useState<SmokeState>({ status: "idle" });
  const abortRef = useRef<AbortController | null>(null);

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

  // Subscribe to `/eval/live` — backend pushes `runs_changed` whenever a new
  // eval row lands. The dashboard re-fetches the list without a page reload.
  // EventSource auto-reconnects on transient disconnects; on unmount we close
  // the connection.
  useEffect(() => {
    const es = new EventSource(`${BACKEND}/eval/live`);
    let retryCount = 0;
    const MAX_RETRIES = 5;
    es.onopen = () => {
      // Successful (re)connect — reset failure budget.
      retryCount = 0;
    };
    es.onmessage = (ev) => {
      try {
        const parsed = JSON.parse(ev.data) as { step?: string };
        if (parsed.step === "runs_changed") load();
      } catch {
        // ignore non-JSON (keepalive comments don't fire onmessage anyway)
      }
    };
    es.onerror = () => {
      // EventSource auto-retries on transient drops (CONNECTING state). On a
      // permanent failure (CLOSED) we cap retries so a 404/401 doesn't flood
      // the network tab indefinitely.
      if (es.readyState === EventSource.CLOSED) {
        retryCount += 1;
        if (retryCount > MAX_RETRIES) {
          es.close();
        }
      }
    };
    return () => es.close();
  }, [load]);

  const runSmoke = useCallback(async () => {
    if (smoke.status === "running") return;
    setSmoke({ status: "running", elapsed: 0, completed: 0, total: 8 });
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      const res = await fetch(`${BACKEND}/eval/smoke`, {
        method: "POST",
        signal: ac.signal,
      });
      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => res.statusText);
        throw new Error(text || res.statusText);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";
        for (const part of parts) {
          const line = part.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          let ev: { step: string; label?: string; data?: unknown };
          try {
            ev = JSON.parse(line.slice(6));
          } catch {
            continue;
          }
          if (ev.step === "start") {
            const d = ev.data as { n_questions: number };
            setSmoke({
              status: "running",
              elapsed: 0,
              completed: 0,
              total: d.n_questions,
              label: "Starting",
            });
          } else if (ev.step === "progress") {
            const d = ev.data as { completed: number; total: number; elapsed_s: number };
            setSmoke({
              status: "running",
              elapsed: d.elapsed_s,
              completed: d.completed,
              total: d.total,
              label: ev.label,
            });
          } else if (ev.step === "done") {
            const d = ev.data as { run_at: string };
            setSmoke({ status: "done", runAt: d.run_at });
            toast.success("Smoke eval complete");
            load();
          } else if (ev.step === "error") {
            setSmoke({
              status: "error",
              message: typeof ev.data === "string" ? ev.data : ev.label || "Smoke eval failed",
            });
            toast.error(typeof ev.data === "string" ? ev.data : "Smoke eval failed");
          }
        }
      }
    } catch (e: unknown) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      setSmoke({
        status: "error",
        message: e instanceof Error ? e.message : "Smoke eval failed",
      });
    }
  }, [smoke.status, load]);

  useEffect(() => () => abortRef.current?.abort(), []);

  return (
    <div className="px-6 md:px-12 py-12 max-w-5xl mx-auto w-full">
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
        {loading && runs.length === 0 && (
          <div className="space-y-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        )}
        {err && !loading && <ErrorPanel detail={err} onRetry={load} />}
        {!loading && !err && runs.length === 0 && (
          <SmokePanel smoke={smoke} onRun={runSmoke} variant="empty" />
        )}
        {runs.length > 0 && smoke.status !== "idle" && (
          <SmokePanel smoke={smoke} onRun={runSmoke} variant="inline" />
        )}
        {runs.length > 0 && smoke.status === "idle" && (
          <div className="mb-6 flex items-center justify-between font-mono text-[11px] text-muted-foreground">
            <span className="uppercase tracking-[0.14em]">
              {runs.length} run{runs.length === 1 ? "" : "s"} recorded
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={runSmoke}
              className="font-mono text-[10px] uppercase tracking-[0.12em]"
            >
              ▶ Run smoke eval (8 questions, ~2–3 min)
            </Button>
          </div>
        )}

        {runs.length >= 2 && <LastRunDelta runs={runs} />}
        {runs.length > 0 && (
          <div className="border-t border-border">
            <div className="grid grid-cols-[1.6fr_60px_80px_1fr_1fr_1fr_1fr_1fr_24px] gap-4 py-3 border-b border-border font-mono text-[11px] uppercase tracking-[0.08em] text-foreground/80 items-end">
              <div>Run</div>
              <div className="text-right tabular-nums" title="Number of questions evaluated in this run">N</div>
              <div className="text-right tabular-nums" title="Pass rate — fraction of questions whose failure_class = PASS">Pass</div>
              <div className="cursor-help underline decoration-dotted decoration-foreground/30 underline-offset-2" title="Faithfulness — does every claim in the answer trace back to the retrieved context? (0–1)">Faith</div>
              <div className="cursor-help underline decoration-dotted decoration-foreground/30 underline-offset-2" title="Answer Relevance — does the answer actually address the question? (0–1)">Relv</div>
              <div className="cursor-help underline decoration-dotted decoration-foreground/30 underline-offset-2" title="Context Precision — did the retrieval layer fetch information needed to answer? (0–1)">CtxP</div>
              <div className="cursor-help underline decoration-dotted decoration-foreground/30 underline-offset-2" title="Citation Integrity — do [doc_N] markers point to URLs that actually contain the cited claim? (0–1)">Cite</div>
              <div className="cursor-help underline decoration-dotted decoration-foreground/30 underline-offset-2" title="Claim Precision — fraction of individual claims in the answer that are supported by a citation (0–1)">ClmP</div>
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
                    <div className="text-[13px] text-foreground flex items-center gap-2">
                      {formatDateTime(r.run_at)}
                      <span className="font-mono text-[9px] uppercase tracking-[0.14em] text-muted-foreground border border-border px-1.5 py-0.5">
                        {(r.retrieval_mode ?? "bm25").toUpperCase()}
                      </span>
                    </div>
                    <div className="font-mono text-[10px] text-subtle-foreground mt-0.5">
                      {r.run_at}
                    </div>
                  </div>
                  <div className="text-right font-mono tabular-nums text-[12px] text-muted-foreground">
                    {r.n_questions}
                  </div>
                  <div className="text-right font-mono tabular-nums-lining text-[13px] text-foreground">
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

/**
 * Smoke runner panel. Two visual modes:
 *  - `empty`: the whole-page empty-state (no runs in the DB yet).
 *  - `inline`: a thin status strip shown above the runs table while a smoke
 *    eval is in progress or its result is being announced.
 *
 * The full 53-question eval is intentionally CLI-only (`python eval/eval_runner.py`)
 * for reproducibility — the smoke variant is the in-browser path so a
 * reviewer never sees an empty Eval page and never has to leave the URL.
 */
function SmokePanel({
  smoke,
  onRun,
  variant,
}: {
  smoke: SmokeState;
  onRun: () => void;
  variant: "empty" | "inline";
}) {
  if (variant === "empty") {
    return (
      <div className="border border-border rounded-[8px] bg-surface p-8 space-y-6">
        <div>
          <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
            No evaluation runs yet
          </div>
          <h2 className="mt-1 font-sans text-[18px] font-medium tracking-tight text-foreground">
            Run your first eval
          </h2>
          <p className="mt-2 font-sans text-[13px] leading-normal text-muted-foreground max-w-prose">
            The full 53-question dataset is meant to be run from CLI for
            reproducibility (
            <code className="font-mono text-[12px] text-foreground bg-surface-hover border border-border rounded-[3px] px-1 py-px">
              python eval/eval_runner.py
            </code>
            ). The 8-question smoke eval runs in-browser in ~2–3 minutes and
            covers all six categories plus a multi-turn pair.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <SmokeStatusInline smoke={smoke} onRun={onRun} />
          <a
            href={`${BACKEND}/eval/results/SAMPLE_REPORT.md`}
            target="_blank"
            rel="noreferrer"
            className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground hover:text-accent transition-colors"
          >
            View sample report →
          </a>
        </div>

        <div className="border-t border-border pt-5">
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground mb-3">
            What the eval measures
          </div>
          <ul className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-2 font-sans text-[12px] text-muted-foreground">
            {[
              ["Faithfulness", "Claims grounded in cited sources"],
              ["Citation integrity", "URLs match the cited claim"],
              ["Context precision", "Retrieval surfaced what was needed"],
              ["Answer relevance", "Response addresses the question"],
              ["Claim precision", "No unsupported assertions"],
              ["Conflict adherence", "Disagreements are surfaced, not hidden"],
              ["Session coherence", "Multi-turn references are resolved"],
              ["Numeric audit", "Numbers tie back to source text"],
            ].map(([label, desc]) => (
              <li key={label} className="flex items-baseline gap-2">
                <span className="text-foreground font-sans">{label}</span>
                <span className="text-subtle-foreground">— {desc}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    );
  }
  return (
    <div className="mb-6 border border-border rounded-[6px] p-4 bg-surface">
      <SmokeStatusInline smoke={smoke} onRun={onRun} />
    </div>
  );
}

function SmokeStatusInline({
  smoke,
  onRun,
}: {
  smoke: SmokeState;
  onRun: () => void;
}) {
  if (smoke.status === "idle") {
    return (
      <Button
        onClick={onRun}
        className="font-mono text-[11px] uppercase tracking-[0.12em]"
      >
        ▶ Run smoke eval
      </Button>
    );
  }
  if (smoke.status === "running") {
    const pct = smoke.total > 0 ? Math.min(100, Math.round((smoke.completed / smoke.total) * 100)) : 0;
    return (
      <div className="space-y-3">
        <div className="flex items-baseline justify-between font-mono text-[11px] text-muted-foreground">
          <span className="uppercase tracking-[0.14em]">
            {smoke.label ?? "Running"} · {smoke.completed}/{smoke.total} questions
          </span>
          <span className="tabular-nums">{smoke.elapsed}s elapsed</span>
        </div>
        <div className="h-1 bg-border-strong rounded-full overflow-hidden">
          <div
            className="h-full bg-accent transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="font-sans text-[12px] text-muted-foreground leading-normal max-w-prose">
          The smoke eval runs sequentially against the live agent. Each
          question generates an answer, then 5–7 LLM-judge calls evaluate
          faithfulness, citation integrity, relevance, conflict adherence,
          and (for the multi-turn pair) session coherence.
        </p>
      </div>
    );
  }
  if (smoke.status === "done") {
    return (
      <div className="flex items-baseline justify-between font-mono text-[11px]">
        <span className="uppercase tracking-[0.14em] text-accent">
          ✓ Smoke eval complete
        </span>
        <Link
          href={`/eval/${encodeURIComponent(smoke.runAt)}`}
          className="text-muted-foreground hover:text-accent transition-colors"
        >
          View run →
        </Link>
      </div>
    );
  }
  return (
    <div className="space-y-2">
      <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-destructive">
        Smoke eval failed
      </div>
      <div className="font-mono text-[11px] text-muted-foreground leading-relaxed">
        {smoke.message}
      </div>
      <Button
        variant="ghost"
        size="sm"
        onClick={onRun}
        className="font-mono text-[10px] uppercase tracking-[0.12em]"
      >
        Retry
      </Button>
    </div>
  );
}

/**
 * Headline delta strip — shows how the most-recent run moved relative to the
 * previous one across the 5 headline metrics. Surfacing this on the list page
 * (rather than per-run) makes regression detection a glance, not a hunt.
 *
 * Runs are returned by the backend in reverse-chronological order, so
 * `runs[0]` is the latest and `runs[1]` is the previous baseline.
 */
function LastRunDelta({ runs }: { runs: EvalRun[] }) {
  const latest = runs[0];
  const prev = runs[1];
  const norm = (v: number | null | undefined): number => {
    if (v === null || v === undefined || Number.isNaN(v)) return 0;
    return v <= 1 ? v : v / 100;
  };
  const items: { label: string; latest: number; prev: number }[] = [
    {
      label: "Pass",
      latest: norm(latest.pass_rate),
      prev: norm(prev.pass_rate),
    },
    {
      label: "Faith",
      latest: norm(latest.avg_faithfulness),
      prev: norm(prev.avg_faithfulness),
    },
    {
      label: "Relv",
      latest: norm(latest.avg_relevance),
      prev: norm(prev.avg_relevance),
    },
    {
      label: "CtxP",
      latest: norm(latest.avg_context_precision),
      prev: norm(prev.avg_context_precision),
    },
    {
      label: "Cite",
      latest: norm(latest.avg_citation_integrity),
      prev: norm(prev.avg_citation_integrity),
    },
  ];
  return (
    <div className="mb-6 border border-border rounded-[6px] bg-surface px-5 py-4">
      <div className="flex items-baseline justify-between mb-3">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          Latest run vs previous
        </div>
        <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground">
          {formatDateTime(prev.run_at)} → {formatDateTime(latest.run_at)}
        </div>
      </div>
      <div className="grid grid-cols-5 gap-6">
        {items.map((it) => {
          const delta = it.latest - it.prev;
          const sign = delta > 0 ? "+" : "";
          const color =
            Math.abs(delta) < 0.005
              ? "text-muted-foreground"
              : delta > 0
                ? "text-accent"
                : "text-[#dc2626]";
          return (
            <div key={it.label}>
              <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-1.5">
                {it.label}
              </div>
              <div className="font-mono tabular-nums text-[18px] text-foreground leading-none">
                {it.latest.toFixed(2)}
              </div>
              <div
                className={`mt-1 font-mono tabular-nums text-[10px] ${color}`}
              >
                {sign}
                {delta.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
