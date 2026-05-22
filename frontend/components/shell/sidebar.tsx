"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { PlusIcon, SearchIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  getProviderHealth,
  listSessions,
  type ProviderHealth,
} from "@/lib/api";
import type { SessionListItem } from "@/lib/types";
import { ThemeToggle } from "./theme-toggle";
import { QuotaPill } from "./quota-pill";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  SESSION_GROUP_ORDER,
  filterSessions,
  groupSessionsByDate,
  sessionDisplayTitle,
} from "@/lib/sessions";

const HEALTH_POLL_MS = 30_000;
const SESSIONS_POLL_MS = 60_000;
const SEARCH_THRESHOLD = 5; // show search input only when list grows beyond this

const ADMIN_NAV = [
  { href: "/eval", label: "Eval" },
  { href: "/settings", label: "Settings" },
  { href: "/status", label: "Status" },
] as const;

/**
 * Custom-event bus between the sidebar (in layout) and the chat page.
 * Avoids prop-drilling through the layout tree; the chat page subscribes
 * to "dra:session-select" and "dra:session-new" and acts accordingly.
 */
function emitSessionSelect(sessionId: string): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem("dra:lastSessionId", sessionId);
  window.dispatchEvent(
    new CustomEvent("dra:session-select", { detail: { sessionId } }),
  );
}

function emitNewSession(): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent("dra:session-new"));
}

function HealthDot({ health }: { health: ProviderHealth | null }) {
  const overall = health?.overall;
  const color =
    overall === "ok"
      ? "bg-accent"
      : overall === "degraded"
        ? "bg-amber-500"
        : overall === "down"
          ? "bg-red-500"
          : "bg-[var(--surface-emphasis)]";
  const summary = health
    ? `${overall?.toUpperCase()} — ${health.providers.filter((p) => p.status === "ok").length}/${health.providers.length} providers OK`
    : "Checking providers…";
  return (
    <span
      title={summary}
      aria-label={summary}
      className={cn("inline-block size-[6px] rounded-full transition-colors", color)}
    />
  );
}

export function Sidebar() {
  const router = useRouter();
  const pathname = usePathname();

  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [search, setSearch] = useState("");
  const [health, setHealth] = useState<ProviderHealth | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);

  // ---- session list polling ------------------------------------------------
  const refresh = useCallback(() => {
    listSessions()
      .then(setSessions)
      .catch(() => {
        /* keep previous list — offline or backend hiccup */
      });
  }, []);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, SESSIONS_POLL_MS);
    const onFocus = () => refresh();
    const onTurnDone = () => refresh();
    window.addEventListener("focus", onFocus);
    window.addEventListener("dra:turn-done", onTurnDone);
    return () => {
      window.clearInterval(id);
      window.removeEventListener("focus", onFocus);
      window.removeEventListener("dra:turn-done", onTurnDone);
    };
  }, [refresh]);

  // ---- active-session tracking (chat page broadcasts on change) -----------
  useEffect(() => {
    const stored = window.localStorage.getItem("dra:lastSessionId");
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (stored) setActiveSessionId(stored);
    const onSelect = (e: Event) => {
      const detail = (e as CustomEvent<{ sessionId: string }>).detail;
      if (detail?.sessionId) setActiveSessionId(detail.sessionId);
    };
    window.addEventListener("dra:session-select", onSelect);
    window.addEventListener("dra:session-new", refresh);
    return () => {
      window.removeEventListener("dra:session-select", onSelect);
      window.removeEventListener("dra:session-new", refresh);
    };
  }, [refresh]);

  // ---- provider health polling --------------------------------------------
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const snap = await getProviderHealth();
        if (alive) setHealth(snap);
      } catch {
        if (alive) setHealth(null);
      }
    };
    tick();
    const id = window.setInterval(tick, HEALTH_POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  // ---- Cmd/Ctrl-K focuses the search input --------------------------------
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const meta = e.metaKey || e.ctrlKey;
      if (meta && e.key.toLowerCase() === "k") {
        e.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const filtered = useMemo(
    () => filterSessions(sessions, search),
    [sessions, search],
  );
  const groups = useMemo(() => groupSessionsByDate(filtered), [filtered]);

  const showSearch = sessions.length > SEARCH_THRESHOLD;

  const handleSelect = (id: string) => {
    setActiveSessionId(id);
    emitSessionSelect(id);
    if (pathname !== "/") router.push("/");
  };

  const handleNew = () => {
    emitNewSession();
    if (pathname !== "/") router.push("/");
  };

  // Up/Down arrow nav across the visible session list.
  const handleListKey = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    const idx = filtered.findIndex((s) => s.session_id === activeSessionId);
    const next =
      e.key === "ArrowDown"
        ? Math.min(filtered.length - 1, idx + 1)
        : Math.max(0, idx - 1);
    if (next !== idx && filtered[next]) {
      e.preventDefault();
      handleSelect(filtered[next].session_id);
    }
  };

  return (
    <aside
      className="hidden md:flex md:flex-col h-screen sticky top-0 w-[280px] shrink-0 border-r border-border bg-background"
      aria-label="Primary navigation"
    >
      {/* Brand row */}
      <div className="px-5 pt-5 pb-4 border-b border-border flex items-center justify-between gap-3">
        <Link href="/" className="flex items-center gap-2 min-w-0 group">
          <span className="text-[15px] font-sans font-medium tracking-tight truncate">
            Deep Research
          </span>
          <HealthDot health={health} />
        </Link>
        <ThemeToggle size="sm" />
      </div>

      {/* New-session CTA */}
      <div className="px-4 pt-4 pb-2">
        <button
          onClick={handleNew}
          className="w-full flex items-center justify-center gap-2 h-9 rounded-[6px] border border-border-accent/40 bg-accent-dim/40 hover:bg-accent-dim text-accent font-sans text-[13px] font-medium tracking-tight transition-colors"
        >
          <PlusIcon size={14} aria-hidden />
          <span>New session</span>
        </button>
      </div>

      {/* Search */}
      {showSearch && (
        <div className="px-4 pb-2">
          <div className="relative">
            <SearchIcon
              size={12}
              aria-hidden
              className="absolute left-2.5 top-1/2 -translate-y-1/2 text-subtle-foreground pointer-events-none"
            />
            <input
              ref={searchRef}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search sessions…"
              aria-label="Search sessions"
              className="w-full pl-7 pr-2 h-8 rounded-[5px] bg-surface border border-border focus:border-border-strong focus:outline-none font-sans text-[12px] placeholder:text-subtle-foreground"
            />
            <kbd className="absolute right-2 top-1/2 -translate-y-1/2 hidden lg:inline-block font-mono text-[9px] text-subtle-foreground border border-border rounded px-1 leading-none py-[2px]">
              ⌘K
            </kbd>
          </div>
        </div>
      )}

      {/* Session list */}
      <ScrollArea className="flex-1">
        <div
          className="py-2 outline-none"
          tabIndex={0}
          role="listbox"
          aria-label="Sessions"
          onKeyDown={handleListKey}
        >
          {sessions.length === 0 && (
            <div className="px-5 py-4 font-mono text-[11px] text-subtle-foreground leading-relaxed">
              No sessions yet — start one ↑
            </div>
          )}

          {sessions.length > 0 && filtered.length === 0 && (
            <div className="px-5 py-4 font-mono text-[11px] text-subtle-foreground leading-relaxed">
              No matches for &ldquo;{search}&rdquo;.
            </div>
          )}

          {SESSION_GROUP_ORDER.map((group) => {
            const items = groups[group];
            if (items.length === 0) return null;
            return (
              <div key={group} className="mb-3">
                <div className="px-5 pt-2 pb-1 font-display italic text-[12px] text-subtle-foreground lowercase tracking-tight">
                  {group}
                </div>
                {items.map((s) => {
                  const active = s.session_id === activeSessionId;
                  const title = sessionDisplayTitle(s);
                  return (
                    <button
                      key={s.session_id}
                      onClick={() => handleSelect(s.session_id)}
                      role="option"
                      aria-selected={active}
                      title={s.session_id}
                      className={cn(
                        "group w-full text-left px-5 py-2 border-l-2 transition-colors flex flex-col gap-0.5",
                        active
                          ? "border-accent"
                          : "border-transparent hover:bg-surface-hover/60",
                      )}
                    >
                      <span
                        className={cn(
                          "font-sans text-[13px] truncate leading-tight",
                          active ? "text-foreground" : "text-foreground/90",
                        )}
                      >
                        {title}
                      </span>
                      <span className="font-mono text-[10px] text-subtle-foreground tabular-nums">
                        {s.turn_count} {s.turn_count === 1 ? "turn" : "turns"}
                        {typeof s.source_count === "number" && (
                          <> · {s.source_count} sources</>
                        )}
                      </span>
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>
      </ScrollArea>

      {/* Admin nav */}
      <nav className="px-5 py-3 border-t border-border flex flex-col gap-0.5">
        {ADMIN_NAV.map((item) => {
          const active = pathname?.startsWith(item.href);
          const isStatus = item.href === "/status";
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center justify-between py-1.5 font-sans text-[12px] tracking-tight transition-colors",
                active
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <span>{item.label}</span>
              {isStatus && <HealthDot health={health} />}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-3 border-t border-border space-y-2">
        <QuotaPill />
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground pt-1">
          Built for Sarvam — 2026
        </div>
      </div>
    </aside>
  );
}
