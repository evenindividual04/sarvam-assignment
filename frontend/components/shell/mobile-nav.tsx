"use client";

import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Activity, BarChart3, MenuIcon, PlusIcon, Settings as SettingsIcon, type LucideIcon } from "lucide-react";
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
import type { SessionListItem } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  SESSION_GROUP_ORDER,
  groupSessionsByDate,
  sessionDisplayTitle,
} from "@/lib/sessions";

const ADMIN_NAV: ReadonlyArray<{
  href: string;
  label: string;
  icon: LucideIcon;
}> = [
  { href: "/eval", label: "Eval", icon: BarChart3 },
  { href: "/settings", label: "Settings", icon: SettingsIcon },
  { href: "/status", label: "Status", icon: Activity },
] as const;

export function MobileNav() {
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [loaded, setLoaded] = useState(false);

  // Lazy-load the session list when the drawer opens so the cold paint stays cheap.
  useEffect(() => {
    if (!open || loaded) return;
    let alive = true;
    listSessions()
      .then((s) => {
        if (alive) {
          setSessions(s);
          setLoaded(true);
        }
      })
      .catch(() => {
        if (alive) setLoaded(true);
      });
    return () => {
      alive = false;
    };
  }, [open, loaded]);

  const onPickSession = (id: string) => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem("dra:lastSessionId", id);
      window.dispatchEvent(
        new CustomEvent("dra:session-select", { detail: { sessionId: id } }),
      );
    }
    setOpen(false);
    router.push("/");
  };

  const onNewSession = () => {
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("dra:session-new"));
    }
    setOpen(false);
    router.push("/");
  };

  const groups = groupSessionsByDate(sessions);

  return (
    <div className="md:hidden flex items-center gap-3 px-5 h-14 border-b border-border sticky top-0 bg-background/80 backdrop-blur-sm z-20">
      <Sheet open={open} onOpenChange={setOpen}>
        <SheetTrigger
          render={
            <Button variant="ghost" size="icon" aria-label="Open menu">
              <MenuIcon className="size-5" />
            </Button>
          }
        />

        <SheetContent
          side="left"
          className="w-[280px] p-0 bg-background border-r border-border flex flex-col"
        >
          <SheetTitle className="sr-only">Navigation</SheetTitle>

          <div className="px-5 pt-5 pb-4 border-b border-border flex items-center justify-between">
            <span className="text-[15px] font-medium tracking-tight">
              Deep Research
            </span>
            <ThemeToggle size="sm" />
          </div>

          <div className="px-4 pt-4 pb-2">
            <button
              onClick={onNewSession}
              className="w-full flex items-center justify-center gap-2 h-9 rounded-[6px] border border-border-accent/40 bg-accent-dim/40 hover:bg-accent-dim text-accent font-sans text-[13px] font-medium tracking-tight transition-colors"
            >
              <PlusIcon size={14} aria-hidden />
              <span>New session</span>
            </button>
          </div>

          <ScrollArea className="flex-1 min-h-0">
            <div className="py-1">
              {!loaded && (
                <div className="px-5 py-3 space-y-2">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <div
                      key={i}
                      className="h-7 rounded-[4px] animate-pulse bg-[var(--surface-hover)]"
                    />
                  ))}
                </div>
              )}
              {loaded && sessions.length === 0 && (
                <div className="px-5 py-4 font-mono text-[11px] text-subtle-foreground leading-relaxed">
                  No sessions yet — start one ↑
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
                    {items.map((s) => (
                      <button
                        key={s.session_id}
                        onClick={() => onPickSession(s.session_id)}
                        className="w-full text-left px-5 py-2 border-l-2 border-transparent hover:bg-surface-hover/60 transition-colors flex flex-col gap-0.5"
                      >
                        <span className="font-sans text-[13px] text-foreground/90 truncate leading-tight">
                          {sessionDisplayTitle(s)}
                        </span>
                        <span className="font-mono text-[10px] text-subtle-foreground tabular-nums">
                          {s.turn_count} {s.turn_count === 1 ? "turn" : "turns"}
                        </span>
                      </button>
                    ))}
                  </div>
                );
              })}
            </div>
          </ScrollArea>

          <div className="border-t border-border px-3 pt-3 pb-1 shrink-0">
            <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground px-2 pb-1">
              More
            </div>
            <nav className="flex flex-col gap-0.5">
              {ADMIN_NAV.map((item) => {
                const active = pathname?.startsWith(item.href);
                const ItemIcon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    onClick={() => setOpen(false)}
                    aria-current={active ? "page" : undefined}
                    className={cn(
                      "group flex items-center gap-2.5 px-3 py-2 rounded-[5px] border-l-2 font-sans text-[13px] tracking-tight transition-colors",
                      active
                        ? "border-accent bg-surface-hover/60 text-foreground"
                        : "border-transparent text-foreground/80 hover:text-foreground hover:bg-surface-hover/40",
                    )}
                  >
                    <ItemIcon
                      size={14}
                      aria-hidden
                      className={cn(
                        "transition-colors",
                        active
                          ? "text-foreground"
                          : "text-muted-foreground group-hover:text-foreground",
                      )}
                    />
                    {item.label}
                  </Link>
                );
              })}
            </nav>
          </div>

          <div className="px-5 py-3 border-t border-border space-y-2">
            <QuotaPill />
            <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground pt-1">
              Built for Sarvam — 2026
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
