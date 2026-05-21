export function formatRelativeTime(date: string | Date): string {
  const d = typeof date === "string" ? new Date(date) : date;
  if (Number.isNaN(d.getTime())) return "—";
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)}d ago`;
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function formatDateTime(date: string | Date): string {
  const d = typeof date === "string" ? new Date(date) : date;
  if (Number.isNaN(d.getTime())) return String(date);
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatPercent(n: number, digits = 1): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${(n * (n <= 1 ? 100 : 1)).toFixed(digits)}%`;
}

export function formatScore(n: number | undefined, digits = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

export function formatMs(ms: number | undefined): string {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function truncate(s: string, max = 80): string {
  if (!s) return "";
  if (s.length <= max) return s;
  return s.slice(0, max - 1) + "…";
}

// Trust-tier colors. Returns Tailwind classes for badges/dots.
export function domainColor(tier?: string | number): string {
  const t = String(tier ?? "").toLowerCase();
  if (t === "tier_1" || t === "1")
    return "bg-emerald-500/15 text-emerald-300 border-emerald-500/30";
  if (t === "tier_2" || t === "2")
    return "bg-teal-500/15 text-teal-300 border-teal-500/30";
  if (t === "tier_3" || t === "3")
    return "bg-blue-500/15 text-blue-300 border-blue-500/30";
  if (t === "tier_4" || t === "4")
    return "bg-amber-500/15 text-amber-300 border-amber-500/30";
  if (t === "tier_5" || t === "5")
    return "bg-orange-500/15 text-orange-300 border-orange-500/30";
  return "bg-muted text-muted-foreground border-border";
}

export function failureClassColor(fc?: string): string {
  switch ((fc || "").toUpperCase()) {
    case "PASS":
      return "bg-emerald-950/40 text-emerald-300 border-emerald-900/40";
    case "HALLUCINATION":
      return "bg-red-950/40 text-red-300 border-red-900/40";
    case "KNOWLEDGE_BLEED":
      return "bg-orange-950/40 text-orange-300 border-orange-900/40";
    case "RETRIEVAL_FAILURE":
      return "bg-amber-950/40 text-amber-300 border-amber-900/40";
    case "CONFLICT_MISS":
      return "bg-red-950/40 text-red-300 border-red-900/40";
    case "COHERENCE_FAIL":
      return "bg-blue-950/40 text-blue-300 border-blue-900/40";
    default:
      return "bg-zinc-900 text-muted-foreground border-border";
  }
}

export function stepStatusColor(
  status: "queued" | "active" | "done" | "skipped" | "cancelled",
): string {
  switch (status) {
    case "active":
      return "bg-accent-dim text-accent border-border-accent";
    case "done":
      return "bg-zinc-900 text-muted-foreground border-border";
    case "skipped":
      return "bg-amber-950/40 text-amber-300 border-amber-900/40";
    case "cancelled":
      return "bg-red-950/40 text-red-300 border-red-900/40";
    default:
      return "bg-zinc-900 text-subtle-foreground border-border";
  }
}
