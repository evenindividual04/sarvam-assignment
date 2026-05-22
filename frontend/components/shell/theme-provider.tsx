"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";
import type { ThemeProviderProps } from "next-themes";

/**
 * Client-only wrapper around `next-themes` so the server-rendered
 * `app/layout.tsx` can mount the provider without becoming a client
 * component itself.
 *
 * Configured for class-based theming on `<html>`, dark default, and no
 * system-preference auto-switch — the toggle is explicit so user intent
 * is the only signal. (System preference following can be re-enabled by
 * flipping `enableSystem` to `true` at the call site.)
 */
export function ThemeProvider({ children, ...props }: ThemeProviderProps) {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem={false}
      disableTransitionOnChange
      {...props}
    >
      {children}
    </NextThemesProvider>
  );
}
