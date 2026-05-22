"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getProviderHealth, type ProviderUsageRow } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * Compact display of today's free-tier usage for the most-utilized provider.
 * Demonstrates that V3.9 quota tracking is wired end-to-end without taking
 * scarce sidebar real-estate from the nav.
 *
 * - Picks the provider with the highest `requests/request_limit` ratio
 *   (preferring providers that actually have a documented limit).
 * - Colors the dot teal / amber / red at 50% / 80% thresholds.
 * - Refreshes on `window.focus` so revisiting the tab after an eval run
 *   shows fresh numbers without a manual reload.
 */
export function QuotaPill() {
  const [hottest, setHottest] = useState<ProviderUsageRow | null>(null);
  const [loaded, setLoaded] = useState(false);

  const refresh = async () => {
    try {
      const snap = await getProviderHealth();
      const rows = snap.usage_today ?? [];
      const withLimit = rows.filter((r) => (r.request_limit ?? 0) > 0);
      const ranked = (withLimit.length ? withLimit : rows).slice().sort((a, b) => {
        const aPct = (a.requests || 0) / (a.request_limit || 1);
        const bPct = (b.requests || 0) / (b.request_limit || 1);
        return bPct - aPct;
      });
      setHottest(ranked[0] ?? null);
    } catch {
      setHottest(null);
    } finally {
      setLoaded(true);
    }
  };

  useEffect(() => {
    refresh();
    const onFocus = () => refresh();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, []);

  if (!loaded) {
    return (
      <div className="h-[18px] w-full rounded-[4px] animate-pulse bg-[var(--surface-hover)]" />
    );
  }
  if (!hottest) {
    return (
      <Link
        href="/status"
        className="block font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground hover:text-foreground transition-colors"
        title="No usage recorded today yet"
      >
        Quota — no usage today
      </Link>
    );
  }

  const limit = hottest.request_limit ?? 0;
  const pct = limit > 0 ? Math.min(1, hottest.requests / limit) : 0;
  const dotColor =
    pct >= 0.8 ? "bg-red-500" : pct >= 0.5 ? "bg-amber-500" : "bg-accent";
  const pctLabel = limit > 0 ? `${Math.round(pct * 100)}%` : "";

  return (
    <Link
      href="/status"
      className="flex items-center gap-2 group"
      title={`${hottest.provider}: ${hottest.requests}${limit ? `/${limit}` : ""} requests today (click for full per-provider snapshot)`}
    >
      <span className={cn("inline-block size-[6px] rounded-full shrink-0", dotColor)} />
      <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-foreground/80 group-hover:text-foreground transition-colors truncate">
        {hottest.provider} {hottest.requests}
        {limit > 0 && `/${limit}`}
        {pctLabel && <span className="ml-1 text-muted-foreground">· {pctLabel}</span>}
      </span>
    </Link>
  );
}
