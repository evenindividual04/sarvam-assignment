"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";
import type { ComponentProps } from "react";

// next-themes 1.0+ no longer exports ThemeProviderProps as a named type.
// Derive it from the component itself so future shape changes stay in sync.
type ThemeProviderProps = ComponentProps<typeof NextThemesProvider>;

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
