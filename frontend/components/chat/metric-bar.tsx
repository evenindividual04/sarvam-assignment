"use client";

import { cn } from "@/lib/utils";

interface MetricBarProps {
  label: string;
  value: number | null | undefined; // 0..1 expected; null/undefined renders as "—"
  tone?: "accent" | "emerald" | "rose" | "amber" | "muted";
  className?: string;
  compact?: boolean;
}

const fillClass: Record<NonNullable<MetricBarProps["tone"]>, string> = {
  accent: "bg-accent",
  emerald: "bg-emerald-500",
  rose: "bg-red-500",
  amber: "bg-amber-500",
  muted: "bg-muted-foreground",
};

export function MetricBar({
  label,
  value,
  tone,
  className,
  compact,
}: MetricBarProps) {
  // Backends sometimes report null for metrics that weren't computed on a
  // given run (e.g. claim_precision on pre-V2.4 rows, context_precision on
  // pre-V3 rows). Treat both null and undefined as "not measured".
  const missing =
    value === null || value === undefined || Number.isNaN(value);
  const v = missing ? 0 : (value as number);
  const normalized = v <= 1 ? v : v / 100;
  const pct = Math.max(0, Math.min(100, normalized * 100));
  const display = missing ? "—" : normalized.toFixed(2);

  const autoTone: NonNullable<MetricBarProps["tone"]> =
    tone ??
    (pct >= 75
      ? "accent"
      : pct >= 50
        ? "accent"
        : pct >= 25
          ? "amber"
          : "rose");

  return (
    <div className={cn("w-full", className)}>
      {label && (
        <div className="flex items-baseline justify-between mb-1.5">
          <span
            className={cn(
              "font-mono uppercase tracking-[0.12em] text-muted-foreground",
              compact ? "text-[10px]" : "text-[11px]",
            )}
          >
            {label}
          </span>
          <span
            className={cn(
              "font-mono tabular-nums text-foreground",
              compact ? "text-[11px]" : "text-[12px]",
            )}
          >
            {display}
          </span>
        </div>
      )}
      <div className="h-[4px] w-full bg-border rounded-[2px] overflow-hidden">
        <div
          className={cn("h-full transition-[width] duration-300", fillClass[autoTone])}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
