"use client";

import { useMemo, useState } from "react";
import type { ClaimAuditRow } from "@/lib/types";
import { truncate } from "@/lib/format";
import { cn } from "@/lib/utils";

interface ClaimsTableProps {
  rows: ClaimAuditRow[];
}

const STATUS_ORDER: Record<string, number> = {
  unsupported: 0,
  ambiguous_resolved: 1,
  supported: 2,
};

const STATUS_BADGE: Record<string, string> = {
  supported: "bg-emerald-950/40 text-emerald-300 border-emerald-900/40",
  unsupported: "bg-red-950/40 text-red-300 border-red-900/40",
  ambiguous_resolved: "bg-amber-950/40 text-amber-300 border-amber-900/40",
};

const METHOD_BADGE: Record<string, string> = {
  deterministic: "bg-zinc-900 text-muted-foreground border-border",
  llm: "bg-blue-950/40 text-blue-300 border-blue-900/40",
  skip: "bg-zinc-900 text-subtle-foreground border-border",
};

export function ClaimsTable({ rows }: ClaimsTableProps) {
  const [sortKey, setSortKey] = useState<"status" | "score">("status");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  const sorted = useMemo(() => {
    const out = [...rows];
    out.sort((a, b) => {
      let cmp = 0;
      if (sortKey === "status") {
        cmp =
          (STATUS_ORDER[a.status] ?? 99) - (STATUS_ORDER[b.status] ?? 99);
      } else if (sortKey === "score") {
        cmp = (a.score ?? 0) - (b.score ?? 0);
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
    return out;
  }, [rows, sortKey, sortDir]);

  if (!rows || rows.length === 0) {
    return (
      <div className="border border-dashed border-border rounded-[8px] p-10 text-center font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
        No per-claim audit recorded.
        <div className="mt-1 normal-case tracking-normal">
          The verification stage may have been skipped.
        </div>
      </div>
    );
  }

  const onSort = (key: "status" | "score") => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  return (
    <div className="border-t border-border">
      <div className="grid grid-cols-[1.5fr_140px_100px_140px_120px] gap-4 py-3 border-b border-border font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
        <div>Claim</div>
        <div>Cited</div>
        <div>Method</div>
        <button
          onClick={() => onSort("score")}
          className="text-left hover:text-foreground transition-colors"
        >
          Score {sortKey === "score" ? (sortDir === "asc" ? "↑" : "↓") : ""}
        </button>
        <button
          onClick={() => onSort("status")}
          className="text-left hover:text-foreground transition-colors"
        >
          Status {sortKey === "status" ? (sortDir === "asc" ? "↑" : "↓") : ""}
        </button>
      </div>
      {sorted.map((r, i) => {
        const score = r.score ?? 0;
        const normalized = score <= 1 ? score : score / 100;
        const pct = Math.max(0, Math.min(100, normalized * 100));
        return (
          <div
            key={`${r.claim_id ?? i}`}
            className="grid grid-cols-[1.5fr_140px_100px_140px_120px] gap-4 py-4 border-b border-border items-start"
          >
            <div>
              <div className="text-[13px] leading-relaxed text-foreground line-clamp-3">
                {r.claim_text}
              </div>
              {r.reasoning && (
                <div className="mt-1.5 font-mono text-[10px] text-subtle-foreground line-clamp-2">
                  {truncate(r.reasoning, 200)}
                </div>
              )}
            </div>
            <div className="flex flex-wrap gap-1">
              {(r.cited_doc_ids ?? []).map((d) => (
                <span
                  key={d}
                  className="font-mono text-[10px] text-muted-foreground border border-border bg-surface px-1.5 py-0.5 rounded-[3px]"
                >
                  {d}
                </span>
              ))}
            </div>
            <div>
              <span
                className={cn(
                  "inline-block font-mono text-[10px] uppercase tracking-[0.10em] px-1.5 py-0.5 rounded-[3px] border",
                  METHOD_BADGE[r.method] ?? METHOD_BADGE.skip,
                )}
              >
                {r.method}
              </span>
            </div>
            <div className="flex items-center gap-2 pt-1">
              <div className="h-[3px] w-[60px] bg-border rounded-[1px] overflow-hidden">
                <div
                  className="h-full bg-accent"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="font-mono tabular-nums text-[11px] text-foreground">
                {normalized.toFixed(2)}
              </span>
            </div>
            <div>
              <span
                className={cn(
                  "inline-block font-mono text-[10px] uppercase tracking-[0.10em] px-1.5 py-0.5 rounded-[3px] border",
                  STATUS_BADGE[r.status] ?? "bg-zinc-900 text-muted-foreground border-border",
                )}
              >
                {r.status}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
