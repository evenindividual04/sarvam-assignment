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

export function formatPercent(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${(n * (n <= 1 ? 100 : 1)).toFixed(digits)}%`;
}

export function formatScore(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

export function formatMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function truncate(s: string, max = 80): string {
  if (!s) return "";
  if (s.length <= max) return s;
  return s.slice(0, max - 1) + "…";
}

/**
 * Bare background-color class for a trust tier dot. Maps the names emitted by
 * the backend's `utils.source_trust.trust_for(domain)` to the V2.3 palette.
 */
export function trustTierColor(tier?: string): string {
  switch (tier) {
    case "tier_1_primary": return "bg-emerald-500";
    case "tier_2_reference": return "bg-teal-500";
    case "tier_3_journalism": return "bg-sky-500";
    case "tier_4_mid": return "bg-amber-500";
    case "tier_5_low": return "bg-orange-500";
    default: return "bg-zinc-500";
  }
}

export function trustTierLabel(tier?: string): string {
  switch (tier) {
    case "tier_1_primary": return "Primary source";
    case "tier_2_reference": return "Reference / academic";
    case "tier_3_journalism": return "Journalism";
    case "tier_4_mid": return "Mid-tier";
    case "tier_5_low": return "Low-trust";
    default: return "Unknown domain";
  }
}

// Trust-tier colors. Returns Tailwind classes for badges/dots. The 15%-alpha
// fill works on cream and near-black backgrounds alike; text shade flips on
// dark-mode for readability.
export function domainColor(tier?: string | number): string {
  const t = String(tier ?? "").toLowerCase();
  if (t === "tier_1" || t === "1")
    return "bg-emerald-500/15 text-emerald-800 dark:text-emerald-300 border-emerald-500/30";
  if (t === "tier_2" || t === "2")
    return "bg-teal-500/15 text-teal-800 dark:text-teal-300 border-teal-500/30";
  if (t === "tier_3" || t === "3")
    return "bg-blue-500/15 text-blue-800 dark:text-blue-300 border-blue-500/30";
  if (t === "tier_4" || t === "4")
    return "bg-amber-500/15 text-amber-800 dark:text-amber-300 border-amber-500/30";
  if (t === "tier_5" || t === "5")
    return "bg-orange-500/15 text-orange-800 dark:text-orange-300 border-orange-500/30";
  return "bg-muted text-muted-foreground border-border";
}

export function failureClassColor(fc?: string): string {
  switch ((fc || "").toUpperCase()) {
    case "PASS":
      return "bg-emerald-100 text-emerald-800 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-900/40";
    case "HALLUCINATION":
      return "bg-red-100 text-red-800 border-red-200 dark:bg-red-950/40 dark:text-red-300 dark:border-red-900/40";
    case "KNOWLEDGE_BLEED":
      return "bg-orange-100 text-orange-800 border-orange-200 dark:bg-orange-950/40 dark:text-orange-300 dark:border-orange-900/40";
    case "RETRIEVAL_FAILURE":
      return "bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-900/40";
    case "CONFLICT_MISS":
      return "bg-red-100 text-red-800 border-red-200 dark:bg-red-950/40 dark:text-red-300 dark:border-red-900/40";
    case "COHERENCE_FAIL":
      return "bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-950/40 dark:text-blue-300 dark:border-blue-900/40";
    default:
      return "bg-surface text-muted-foreground border-border";
  }
}

// ---------------------------------------------------------------------------
// Turn export: Markdown + BibTeX
//
// TESTING: hand-verified — no frontend test infra exists in this project
// (no jest/vitest config in frontend/package.json). Verified by importing
// formatTurnAsMarkdown / formatTurnAsBibtex into trace-inspector and
// downloading on a real session turn. Sanitization rules and YYYY-MM-DD
// formatting both eyeballed against fixture output.
// ---------------------------------------------------------------------------

import type { Turn } from "./types";

/**
 * Domain extracted from a URL, lower-cased, leading "www." stripped. Falls
 * back to a slug of the raw string if URL parsing fails.
 */
function domainFromUrl(url: string): string {
  try {
    const u = new URL(url);
    return u.hostname.replace(/^www\./, "").toLowerCase();
  } catch {
    return url.replace(/[^a-z0-9]/gi, "_").toLowerCase().slice(0, 32) || "unknown";
  }
}

/** YYYYMMDD from a Date (UTC). */
function yyyymmdd(d: Date): string {
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${y}${m}${day}`;
}

/** YYYY-MM-DD from a Date (UTC). */
function isoDate(d: Date): string {
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/**
 * Escape BibTeX-special characters in a free-text field. Order matters:
 * backslash must come first so we don't double-escape the escapes we
 * subsequently insert.
 */
function bibtexEscape(s: string): string {
  if (!s) return "";
  return s
    .replace(/\\/g, "\\textbackslash{}")
    .replace(/([{}%&$#_])/g, "\\$1");
}

/**
 * Build one `@misc{...}` entry per UNIQUE cited URL. Citation source is
 * `turn.doc_map` (preferred — has title + domain) with fallback to
 * `turn.urls_opened`. Duplicates by URL are dropped; entry keys disambiguate
 * with a 1-based counter when multiple sources share a domain on the same day.
 */
export function formatTurnAsBibtex(turn: Turn): string {
  const today = new Date();
  const datestamp = yyyymmdd(today);
  const urldate = isoDate(today);

  type Entry = { url: string; title: string; domain: string };
  const entries: Entry[] = [];
  const seen = new Set<string>();

  if (turn.doc_map) {
    for (const tup of Object.values(turn.doc_map)) {
      const [title, url, domain] = tup;
      if (!url || seen.has(url)) continue;
      seen.add(url);
      entries.push({
        url,
        title: title || url,
        domain: (domain || domainFromUrl(url)).toLowerCase(),
      });
    }
  }
  if (turn.urls_opened) {
    for (const url of turn.urls_opened) {
      if (!url || seen.has(url)) continue;
      seen.add(url);
      entries.push({ url, title: url, domain: domainFromUrl(url) });
    }
  }

  const perDomainCount: Record<string, number> = {};
  const blocks = entries.map((e) => {
    const safeDomain = e.domain.replace(/[^a-z0-9]+/gi, "_").replace(/^_+|_+$/g, "");
    perDomainCount[safeDomain] = (perDomainCount[safeDomain] ?? 0) + 1;
    const n = perDomainCount[safeDomain];
    const key = `${safeDomain}_${datestamp}_${n}`;
    return [
      `@misc{${key},`,
      `  title  = {${bibtexEscape(e.title)}},`,
      // S5 fix: URLs containing { } or \ break BibTeX parsing; a malicious
      // doc_map entry could also inject arbitrary BibTeX fields. Escape with
      // the same helper used for title/note. urldate is our own ISO date, no
      // escaping needed.
      // TESTING: hand-verified bibtex output passes through https://bibtex.online/ validator
      `  url    = {${bibtexEscape(e.url)}},`,
      `  note   = {Retrieved from ${bibtexEscape(e.domain)} on ${urldate}},`,
      `  urldate = {${urldate}}`,
      `}`,
    ].join("\n");
  });

  return blocks.join("\n\n") + (blocks.length ? "\n" : "");
}

/**
 * Render a turn as a portable Markdown research artifact. Includes the
 * question, sub-queries (from `search_queries`), numbered sources, the full
 * answer with citations preserved verbatim, and a footer with token/latency
 * stats.
 */
export function formatTurnAsMarkdown(turn: Turn): string {
  const lines: string[] = [];
  const created = turn.created_at ? new Date(turn.created_at) : new Date();
  const timestamp = Number.isNaN(created.getTime())
    ? String(turn.created_at ?? "")
    : created.toISOString();

  lines.push(`# ${turn.query ?? "(no query)"}`);
  lines.push("");
  lines.push(
    `_Generated ${timestamp} · Session ${turn.session_id ?? "—"} · Turn ${turn.turn_id ?? "—"}_`,
  );
  lines.push("");

  // Research plan — currently the Turn shape does not carry a free-form plan
  // string. Reserve the heading so the artifact has a predictable structure
  // and downstream tools can populate it when the API exposes it.
  lines.push("## Research Plan");
  lines.push(turn.planning_strategy ? `_Strategy: ${turn.planning_strategy}_` : "_Not recorded._");
  lines.push("");

  lines.push("## Sub-queries");
  if (turn.search_queries && turn.search_queries.length > 0) {
    for (const q of turn.search_queries) lines.push(`- ${q}`);
  } else {
    lines.push("_None recorded._");
  }
  lines.push("");

  // Sources: prefer doc_map (has title + domain). Deduplicate by URL,
  // preserving doc_map ordering, then append any urls_opened not already
  // covered.
  lines.push("## Sources");
  type Src = { url: string; title: string; domain: string };
  const seen = new Set<string>();
  const sources: Src[] = [];
  if (turn.doc_map) {
    for (const tup of Object.values(turn.doc_map)) {
      const [title, url, domain] = tup;
      if (!url || seen.has(url)) continue;
      seen.add(url);
      sources.push({ url, title: title || url, domain: domain || domainFromUrl(url) });
    }
  }
  if (turn.urls_opened) {
    for (const url of turn.urls_opened) {
      if (!url || seen.has(url)) continue;
      seen.add(url);
      sources.push({ url, title: url, domain: domainFromUrl(url) });
    }
  }
  if (sources.length === 0) {
    lines.push("_No sources recorded._");
  } else {
    sources.forEach((s, i) => {
      lines.push(`${i + 1}. [${s.title} — ${s.domain}](${s.url})`);
    });
  }
  lines.push("");

  lines.push("## Answer");
  lines.push(turn.response ?? "_(no answer)_");
  lines.push("");
  lines.push("---");

  const promptTokens = turn.prompt_tokens ?? 0;
  const completionTokens = turn.completion_tokens ?? 0;
  const latency = formatMs(turn.latency_ms);
  lines.push(
    `_Retrieved ${sources.length} sources · ${promptTokens} in / ${completionTokens} out tokens · ${latency}_`,
  );

  return lines.join("\n") + "\n";
}

export function stepStatusColor(
  status: "queued" | "active" | "done" | "skipped" | "cancelled",
): string {
  switch (status) {
    case "active":
      return "bg-accent-dim text-accent border-border-accent";
    case "done":
      return "bg-surface text-muted-foreground border-border";
    case "skipped":
      return "bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-900/40";
    case "cancelled":
      return "bg-red-100 text-red-800 border-red-200 dark:bg-red-950/40 dark:text-red-300 dark:border-red-900/40";
    default:
      return "bg-surface text-subtle-foreground border-border";
  }
}
