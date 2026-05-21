"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { getHealth } from "@/lib/api";

const NAV = [
  { href: "/", label: "Chat" },
  { href: "/sessions", label: "Sessions" },
  { href: "/eval", label: "Eval" },
  { href: "/settings", label: "Settings" },
];

function HealthDot() {
  const [ok, setOk] = useState<boolean | null>(null);
  useEffect(() => {
    let alive = true;
    getHealth()
      .then(() => alive && setOk(true))
      .catch(() => alive && setOk(false));
    return () => {
      alive = false;
    };
  }, []);
  const color =
    ok === true
      ? "bg-accent"
      : ok === false
        ? "bg-zinc-700"
        : "bg-zinc-800";
  const label =
    ok === true ? "Backend online" : ok === false ? "Backend offline" : "Checking";
  return (
    <span
      title={label}
      aria-label={label}
      className={cn("inline-block size-[6px] rounded-full", color)}
    />
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

      <div className="px-6 py-4 border-t border-border">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground">
          Built for Sarvam — 2026
        </div>
      </div>
    </aside>
  );
}
