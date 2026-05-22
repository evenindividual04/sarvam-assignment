"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { getProviderHealth, type ProviderHealth } from "@/lib/api";
import { ThemeToggle } from "./theme-toggle";
import { QuotaPill } from "./quota-pill";

const NAV = [
  { href: "/", label: "Chat" },
  { href: "/sessions", label: "Sessions" },
  { href: "/eval", label: "Eval" },
  { href: "/settings", label: "Settings" },
  { href: "/status", label: "Status" },
];

const POLL_INTERVAL_MS = 60_000;

function HealthDot() {
  const [health, setHealth] = useState<ProviderHealth | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const snap = await getProviderHealth();
        if (alive) {
          setHealth(snap);
          setLoading(false);
        }
      } catch {
        if (alive) {
          setHealth(null);
          setLoading(false);
        }
      }
    };
    tick();
    const id = setInterval(tick, POLL_INTERVAL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const overall = health?.overall;
  const color =
    loading
      ? "bg-[var(--surface-emphasis)]"
      : overall === "ok"
        ? "bg-accent"
        : overall === "degraded"
          ? "bg-amber-500"
          : "bg-red-500";

  const summary = (() => {
    if (loading) return "Checking providers…";
    if (!health) return "Health unknown";
    const okCount = health.providers.filter((p) => p.status === "ok").length;
    const total = health.providers.length;
    return `${overall?.toUpperCase()} — ${okCount}/${total} providers OK`;
  })();

  return (
    <Link href="/status" title={summary} aria-label={summary} className="shrink-0">
      <span
        className={cn(
          "inline-block size-[6px] rounded-full transition-colors",
          color,
        )}
      />
    </Link>
  );
}

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="hidden md:flex md:flex-col h-screen sticky top-0 w-[240px] shrink-0 border-r border-border bg-background">
      <div className="px-6 pt-6 pb-5 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="text-[15px] font-sans font-medium tracking-tight leading-none">
            Deep Research
          </div>
          <HealthDot />
        </div>
      </div>

      <nav className="px-6 py-5 flex flex-col gap-0.5 flex-1">
        {NAV.map((item) => {
          const active =
            item.href === "/"
              ? pathname === "/"
              : pathname?.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "group relative flex items-center justify-between py-2 font-sans text-[13px] uppercase tracking-[0.12em] transition-colors",
                active
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <span className="relative">
                {item.label}
                <span
                  className={cn(
                    "absolute -bottom-1 left-0 h-px transition-all",
                    active
                      ? "w-full bg-accent"
                      : "w-0 bg-accent group-hover:w-full",
                  )}
                />
              </span>
              {active && (
                <span className="font-mono text-[10px] text-accent">·</span>
              )}
            </Link>
          );
        })}
      </nav>

      <div className="px-6 py-4 border-t border-border space-y-3">
        <QuotaPill />
        <div className="flex items-center justify-between">
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground">
            Built for Sarvam — 2026
          </div>
          <ThemeToggle size="sm" />
        </div>
      </div>
    </aside>
  );
}
