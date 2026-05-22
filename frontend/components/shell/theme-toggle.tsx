"use client";

import { useEffect, useState } from "react";
import { useTheme } from "@/components/shell/theme-provider";
import { MoonIcon, SunIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface ThemeToggleProps {
  className?: string;
  size?: "sm" | "md";
}

/**
 * Sun/moon toggle. Renders a placeholder until `mounted === true` to avoid
 * the React hydration mismatch that occurs with any next-themes consumer
 * — the server doesn't know which theme will be active on the client.
 */
export function ThemeToggle({ className, size = "md" }: ThemeToggleProps) {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  const isDark = mounted ? resolvedTheme === "dark" : true;
  const iconSize = size === "sm" ? 14 : 16;

  return (
    <button
      type="button"
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className={cn(
        "inline-flex items-center justify-center rounded-[6px] border border-border bg-surface",
        "text-muted-foreground hover:text-foreground hover:border-border-strong transition-colors",
        size === "sm" ? "size-7" : "size-8",
        className,
      )}
    >
      {/* Render both icons to keep the markup stable across hydration. */}
      <span aria-hidden className={cn("transition-opacity", mounted && isDark ? "block" : "hidden")}>
        <MoonIcon size={iconSize} />
      </span>
      <span aria-hidden className={cn("transition-opacity", mounted && !isDark ? "block" : "hidden")}>
        <SunIcon size={iconSize} />
      </span>
    </button>
  );
}
