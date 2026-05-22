"use client";

import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { MenuIcon } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ThemeToggle } from "./theme-toggle";
import { QuotaPill } from "./quota-pill";
import { listSessions } from "@/lib/api";
import { formatRelativeTime, truncate } from "@/lib/format";
import type { SessionListItem } from "@/lib/types";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Chat" },
  { href: "/sessions", label: "Sessions" },
  { href: "/eval", label: "Eval" },
  { href: "/settings", label: "Settings" },
  { href: "/status", label: "Status" },
];

const RECENT_LIMIT = 10;

export function MobileNav() {
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [sessionsLoaded, setSessionsLoaded] = useState(false);

  // Fetch recent sessions only when the sheet opens — keeps cold paint cheap.
  useEffect(() => {
    if (!open || sessionsLoaded) return;
    let alive = true;
    listSessions()
      .then((s) => {
        if (alive) {
          setSessions(s.slice(0, RECENT_LIMIT));
          setSessionsLoaded(true);
        }
      })
      .catch(() => {
        if (alive) setSessionsLoaded(true);
      });
    return () => {
      alive = false;
    };
  }, [open, sessionsLoaded]);

  const onPickSession = (id: string) => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem("dra:lastSessionId", id);
    }
    setOpen(false);
    router.push("/");
  };

  return (
    <div className="md:hidden flex items-center gap-3 px-5 h-14 border-b border-border sticky top-0 bg-background/80 backdrop-blur-sm z-20">
      <Sheet open={open} onOpenChange={setOpen}>
        <SheetTrigger
          render={
            <Button variant="ghost" size="icon">
              <MenuIcon className="size-5" />
            </Button>
          }
        />

        <SheetContent
          side="left"
          className="w-[280px] p-0 bg-background border-r border-border flex flex-col"
        >
          <SheetTitle className="sr-only">Navigation</SheetTitle>
          <div className="px-6 pt-6 pb-5 border-b border-border">
            <div className="text-[15px] font-medium tracking-tight">
              Deep Research
            </div>
          </div>
          <nav className="px-6 py-5 flex flex-col gap-0.5 border-b border-border">
            {NAV.map((item) => {
              const active =
                item.href === "/"
                  ? pathname === "/"
                  : pathname?.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setOpen(false)}
                  className={cn(
                    "py-2 font-sans text-[13px] uppercase tracking-[0.12em] transition-colors",
                    active
                      ? "text-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>

          {/* Sessions section — fixes the "mobile users can't switch sessions"
              gap. The desktop SessionsRail (`hidden lg:block`) was unreachable
              below 1024px; here we surface the 10 most-recent inline. */}
          <div className="px-6 py-4 border-b border-border">
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                Recent sessions
              </span>
              <Link
                href="/sessions"
                onClick={() => setOpen(false)}
                className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground hover:text-accent transition-colors"
              >
                View all →
              </Link>
            </div>
          </div>
          <ScrollArea className="flex-1">
            <div className="py-1">
              {!sessionsLoaded && (
                <div className="px-6 py-3 space-y-2">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <div
                      key={i}
                      className="h-7 rounded-[4px] animate-pulse bg-[var(--surface-hover)]"
                    />
                  ))}
                </div>
              )}
              {sessionsLoaded && sessions.length === 0 && (
                <div className="px-6 py-4 font-mono text-[11px] text-subtle-foreground leading-relaxed">
                  No sessions yet. Send a query to start one.
                </div>
              )}
              {sessions.map((s) => (
                <button
                  key={s.session_id}
                  onClick={() => onPickSession(s.session_id)}
                  className="w-full text-left px-5 py-2.5 border-l-2 border-transparent hover:bg-surface-hover/60 hover:border-border-strong transition-colors flex flex-col gap-0.5"
                >
                  <span className="font-mono text-[11px] text-foreground truncate">
                    {truncate(s.session_id, 24)}
                  </span>
                  <span className="font-mono text-[10px] text-subtle-foreground tabular-nums">
                    {s.turn_count}T · {formatRelativeTime(s.updated_at)}
                  </span>
                </button>
              ))}
            </div>
          </ScrollArea>

          <div className="px-6 py-4 border-t border-border space-y-3">
            <QuotaPill />
            <div className="flex items-center justify-end">
              <ThemeToggle size="sm" />
            </div>
          </div>
        </SheetContent>
      </Sheet>
      <div className="flex flex-col flex-1">
        <span className="text-sm font-medium tracking-tight leading-none">
          Deep Research
        </span>
      </div>
      <ThemeToggle size="sm" />
    </div>
  );
}
