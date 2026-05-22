"use client";

import { useCallback, useEffect, useState } from "react";
import { getProviderHealth, type ProviderHealth, type ProviderProbe } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const STATUS_LABELS: Record<ProviderProbe["status"], string> = {
  ok: "OK",
  degraded: "Degraded",
  down: "Down",
  missing_key: "Missing key",
};

function StatusPill({ status }: { status: ProviderProbe["status"] }) {
  const color =
    status === "ok"
      ? "bg-accent/15 text-accent border-accent/30"
      : status === "degraded"
        ? "bg-amber-500/15 text-amber-400 border-amber-500/30"
        : status === "missing_key"
          ? "bg-surface-emphasis text-muted-foreground border-border-strong"
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

function OverallBanner({ health }: { health: ProviderHealth }) {
  const messages = {
    ok: "All providers reachable. Search, planning, and synthesis are nominal.",
    degraded:
      "One or more non-critical providers are unreachable. Fallbacks should keep requests working.",
    down: "A critical provider (Parallel, Gemini, or Groq) is unreachable. Requests may fail.",
  };
  const tone =
    health.overall === "ok"
      ? "border-accent/30 bg-accent/5 text-foreground"
      : health.overall === "degraded"
        ? "border-amber-500/30 bg-amber-500/5 text-foreground"
        : "border-red-500/30 bg-red-500/5 text-foreground";
  return (
    <div className={cn("rounded border p-4 text-sm", tone)}>
      <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
        Overall · {health.overall}
      </div>
      <p className="mt-1 leading-relaxed">{messages[health.overall]}</p>
    </div>
  );
}

export default function StatusPage() {
  const [health, setHealth] = useState<ProviderHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  // Wrapped in useCallback so the setInterval closure below doesn't capture
  // a stale `load` reference when React re-renders during refresh.
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
    <div className="mx-auto max-w-3xl px-6 py-10">
      <header className="mb-8 flex items-baseline justify-between">
        <div>
          <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            System
          </div>
          <h1 className="mt-1 text-2xl font-sans tracking-tight">
            Provider status
          </h1>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => load(true)}
          disabled={refreshing}
        >
          {refreshing ? "Refreshing…" : "Force refresh"}
        </Button>
      </header>

      {loading && !health && (
        <div className="text-sm text-muted-foreground">Checking providers…</div>
      )}
      {error && (
        <div className="rounded border border-red-500/30 bg-red-500/5 p-4 text-sm">
          {error}
        </div>
      )}

      {health && (
        <div className="space-y-6">
          <OverallBanner health={health} />

          <div className="overflow-hidden rounded border border-border">
            <table className="w-full text-sm">
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
                {health.providers.map((p) => (
                  <tr key={p.name} className="border-t border-border">
                    <td className="px-4 py-3 font-mono text-[13px]">{p.name}</td>
                    <td className="px-4 py-3 text-muted-foreground">{p.role}</td>
                    <td className="px-4 py-3">
                      <StatusPill status={p.status} />
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-[12px]">
                      {p.latency_ms != null ? `${p.latency_ms} ms` : "—"}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground text-[12px]">
                      {p.detail || "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="font-mono text-[11px] text-muted-foreground">
            Last checked {new Date(health.checked_at * 1000).toLocaleTimeString()} ·
            cache TTL {health.cache_ttl_s}s · auto-refresh every 60s
          </div>
        </div>
      )}
    </div>
  );
}
