"use client";

import { useCallback, useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { getProviderHealth, type ProviderHealth, type ProviderProbe } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { formatRelativeTime } from "@/lib/format";
import { cn } from "@/lib/utils";

const STATUS_LABELS: Record<ProviderProbe["status"], string> = {
  ok: "OK",
  degraded: "Degraded",
  down: "Down",
  missing_key: "Missing key",
  not_configured: "Not configured",
};

function StatusPill({ status }: { status: ProviderProbe["status"] }) {
  const color =
    status === "ok"
      ? "bg-accent/15 text-accent border-accent/30"
      : status === "degraded"
        ? "bg-amber-500/15 text-amber-400 border-amber-500/30"
        : status === "missing_key" || status === "not_configured"
          ? "bg-surface-emphasis text-muted-foreground/70 border-border"
          : "bg-red-500/15 text-red-400 border-red-500/30";
  return (
    <span
      className={cn(
        "inline-block rounded border px-2 py-[2px] font-mono text-[11px] uppercase tracking-wider",
        color,
      )}
    >
      {STATUS_LABELS[status]}
    </span>
  );
}

/**
 * Latency cell color thresholds — kept in sync with the legend below the
 * table. Returns a Tailwind class string so callers can splat it into cn().
 */
function latencyColor(latencyMs: number | null): string {
  if (latencyMs == null) return "text-muted-foreground";
  if (latencyMs < 500) return "text-emerald-300";
  if (latencyMs <= 1500) return "text-amber-300";
  return "text-red-400";
}

function OverallBanner({ health }: { health: ProviderHealth }) {
  // not_configured providers are intentional absences (e.g. optional Ollama
  // on remote deploys) — exclude them from failure counts.
  const degradedNames = health.providers
    .filter((p) => p.status === "degraded")
    .map((p) => p.name);
  const downNames = health.providers
    .filter((p) => p.status === "down")
    .map((p) => p.name);

  let message: string;
  if (health.overall === "ok") {
    message = "All configured providers reachable. Search, planning, and synthesis are nominal.";
  } else if (health.overall === "degraded") {
    const names = [...degradedNames, ...downNames].join(", ") || "one or more providers";
    message = `Degraded: ${names}. Fallbacks active for these roles.`;
  } else {
    message = `Critical provider unreachable: ${downNames.join(", ") || "unknown"}. Requests may fail.`;
  }

  const tone =
    health.overall === "ok"
      ? "border-accent/30 bg-accent/5 text-foreground"
      : health.overall === "degraded"
        ? "border-amber-500/30 bg-amber-500/5 text-foreground"
        : "border-red-500/30 bg-red-500/5 text-foreground";
  const overallLabel =
    health.overall === "ok"
      ? "operational"
      : health.overall;
  return (
    <div className={cn("rounded border p-4 text-sm", tone)}>
      <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
        Overall · {overallLabel}
      </div>
      <p className="mt-1 font-sans leading-relaxed">{message}</p>
    </div>
  );
}

export default function StatusPage() {
  const [health, setHealth] = useState<ProviderHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  // useCallback so the setInterval below doesn't close over a stale reference.
  const load = useCallback(async (force = false) => {
    try {
      setError(null);
      if (force) setRefreshing(true);
      const snap = await getProviderHealth(force);
      setHealth(snap);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load health");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    load(false);
    const id = setInterval(() => load(false), 60_000);
    return () => clearInterval(id);
  }, [load]);

  return (
    <div className="mx-auto max-w-5xl px-6 md:px-12 py-12 w-full">
      <header className="mb-8">
        <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          System
        </div>
        <div className="mt-1 flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-3">
          <h1 className="text-2xl font-sans font-medium tracking-tight">
            Provider status
          </h1>
          <div className="flex items-center gap-3">
            <span className="font-mono text-[11px] text-subtle-foreground">
              {health
                ? `Last checked ${formatRelativeTime(
                    new Date(health.checked_at * 1000).toISOString(),
                  )}`
                : "—"}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => load(true)}
              disabled={refreshing}
              className="font-mono text-[10px] uppercase tracking-[0.12em]"
            >
              <RefreshCw
                size={12}
                className={cn("mr-1.5", refreshing && "animate-spin")}
                aria-hidden
              />
              {refreshing ? "Refreshing" : "Force refresh"}
            </Button>
          </div>
        </div>
      </header>

      {loading && !health && (
        <div className="space-y-3">
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      )}
      {error && (
        <div className="rounded border border-red-500/30 bg-red-500/5 p-4 text-sm">
          {error}
        </div>
      )}

      {health && (
        <div className="space-y-6">
          <OverallBanner health={health} />

          <div className="overflow-x-auto rounded border border-border">
            <table className="w-full text-sm min-w-[640px]">
              <thead className="bg-muted/30">
                <tr className="text-left font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
                  <th className="px-4 py-2">Provider</th>
                  <th className="px-4 py-2">Role</th>
                  <th className="px-4 py-2">Status</th>
                  <th className="px-4 py-2 text-right">Latency</th>
                  <th className="px-4 py-2">Detail</th>
                </tr>
              </thead>
              <tbody>
                {health.providers.map((p) => {
                  const dimmed =
                    p.status === "not_configured" || p.status === "missing_key";
                  return (
                    <tr
                      key={p.name}
                      className={cn(
                        "border-t border-border",
                        dimmed && "opacity-60",
                      )}
                    >
                      <td className="px-4 py-3 font-mono text-[13px]">{p.name}</td>
                      <td className="px-4 py-3 font-sans text-muted-foreground">{p.role}</td>
                      <td className="px-4 py-3">
                        <StatusPill status={p.status} />
                      </td>
                      <td
                        className={cn(
                          "px-4 py-3 text-right font-mono text-[12px] tabular-nums",
                          dimmed ? "text-muted-foreground" : latencyColor(p.latency_ms),
                        )}
                      >
                        {dimmed ? "—" : p.latency_ms != null ? `${p.latency_ms} ms` : "—"}
                      </td>
                      <td className="px-4 py-3 font-sans text-muted-foreground text-[12px]">
                        {p.detail || "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
              <span>Latency:</span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block size-1.5 rounded-full bg-emerald-300" aria-hidden />
                <span>&lt;500ms good</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block size-1.5 rounded-full bg-amber-300" aria-hidden />
                <span>500–1500ms acceptable</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block size-1.5 rounded-full bg-red-400" aria-hidden />
                <span>&gt;1500ms slow</span>
              </span>
            </div>
            <span>cache TTL {health.cache_ttl_s}s · auto-refresh every 60s</span>
          </div>
        </div>
      )}
    </div>
  );
}
