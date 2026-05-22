// Session list helpers: date grouping + display-title derivation.
//
// The backend's SessionListItem currently carries only session_id, updated_at,
// and turn_count. We render a friendly title when the backend supplies one
// (future enhancement); otherwise we degrade to a short session-id slug so
// the sidebar still scans cleanly.

import type { SessionListItem } from "./types";

export type SessionGroup =
  | "today"
  | "yesterday"
  | "previous 7 days"
  | "previous 30 days"
  | "older";

export const SESSION_GROUP_ORDER: readonly SessionGroup[] = [
  "today",
  "yesterday",
  "previous 7 days",
  "previous 30 days",
  "older",
] as const;

/** Bucket label for a given `updated_at` timestamp, in the local tz. */
export function bucketForDate(value: string | Date, now: Date = new Date()): SessionGroup {
  const d = typeof value === "string" ? new Date(value) : value;
  if (Number.isNaN(d.getTime())) return "older";

  const startOfDay = (x: Date): Date => {
    const c = new Date(x);
    c.setHours(0, 0, 0, 0);
    return c;
  };

  const today = startOfDay(now);
  const subjectDay = startOfDay(d);
  const diffDays = Math.floor((today.getTime() - subjectDay.getTime()) / 86_400_000);

  if (diffDays <= 0) return "today";
  if (diffDays === 1) return "yesterday";
  if (diffDays <= 7) return "previous 7 days";
  if (diffDays <= 30) return "previous 30 days";
  return "older";
}

/** Group sessions in fixed display order; empty groups are kept so callers can skip them. */
export function groupSessionsByDate(
  sessions: readonly SessionListItem[],
  now: Date = new Date(),
): Record<SessionGroup, SessionListItem[]> {
  const out: Record<SessionGroup, SessionListItem[]> = {
    "today": [],
    "yesterday": [],
    "previous 7 days": [],
    "previous 30 days": [],
    "older": [],
  };
  for (const s of sessions) {
    out[bucketForDate(s.updated_at, now)].push(s);
  }
  return out;
}

/**
 * Human-friendly title for a session in the sidebar list.
 *
 * Precedence:
 *   1. Explicit `title` (if the backend ever stamps one)
 *   2. Truncated `first_query` (the verbatim first user turn)
 *   3. "Untitled session" when the session has no turns yet
 *   4. Hash-slug fallback (`session_id.slice(0, 8)`) as last resort
 */
export function sessionDisplayTitle(
  s: SessionListItem & { title?: string | null; first_query?: string | null },
): string {
  const t = (s.title ?? "").trim();
  if (t) return truncateAtWord(t, 40);

  const fq = (s.first_query ?? "").trim();
  if (fq) return truncateAtWord(fq, 40);

  if (s.turn_count === 0) return "Untitled session";

  // Sessions land with cryptographically random IDs (uuid / s-<ts>-<rand>);
  // showing the first 8 chars in mono gives a scannable, copyable handle.
  return s.session_id.slice(0, 8);
}

/** Soft-truncate at the nearest word boundary; appends a horizontal ellipsis. */
export function truncateAtWord(s: string, max: number): string {
  if (!s) return "";
  if (s.length <= max) return s;
  const slice = s.slice(0, max);
  const lastSpace = slice.lastIndexOf(" ");
  const cut = lastSpace > max * 0.6 ? slice.slice(0, lastSpace) : slice;
  return cut.trimEnd() + "…";
}

/** Case-insensitive substring search across title + session_id. */
export function filterSessions(
  sessions: readonly SessionListItem[],
  query: string,
): SessionListItem[] {
  const q = query.trim().toLowerCase();
  if (!q) return [...sessions];
  return sessions.filter((s) => {
    const title = sessionDisplayTitle(s).toLowerCase();
    return title.includes(q) || s.session_id.toLowerCase().includes(q);
  });
}
