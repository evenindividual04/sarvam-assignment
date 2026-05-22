import type { Metadata } from "next";
import { Instrument_Serif, Geist, JetBrains_Mono } from "next/font/google";
import Script from "next/script";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { Sidebar } from "@/components/shell/sidebar";
import { MobileNav } from "@/components/shell/mobile-nav";
import { ThemeProvider } from "@/components/shell/theme-provider";

const display = Instrument_Serif({
  subsets: ["latin"],
  weight: ["400"],
  style: ["normal", "italic"],
  variable: "--font-display",
  display: "swap",
});

const sans = Geist({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Deep Research Agent",
  description:
    "Multi-source web research with conflict detection, claim verification, and LLM-as-judge evaluation.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${display.variable} ${sans.variable} ${mono.variable}`}
    >
      <body className="bg-background text-foreground h-screen overflow-hidden antialiased">
        {/* FOUC-prevention: applies the persisted theme class to <html> before
            first paint. next/script with strategy="beforeInteractive" injects
            this into <head> at server-render time, executes synchronously
            before React hydrates, and avoids React 19's "script in render tree"
            warning that a bare <script> would trigger. */}
        <Script
          id="theme-fouc-prevent"
          strategy="beforeInteractive"
        >
          {`(function(){try{var t=localStorage.getItem('theme');if(t!=='light'&&t!=='dark')t='dark';var d=document.documentElement;d.classList.remove('light','dark');d.classList.add(t);d.style.colorScheme=t;}catch(e){document.documentElement.classList.add('dark');}})();`}
        </Script>
        <ThemeProvider>
          {/* Subtle radial vignette: bright glow in dark mode, soft dark glow in
              light mode. Avoids the "flat wall of color" look in either theme. */}
          <div
            aria-hidden
            className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(ellipse_at_top,rgba(0,0,0,0.025),transparent_60%)] dark:bg-[radial-gradient(ellipse_at_top,rgba(255,255,255,0.025),transparent_60%)]"
          />
          <div className="flex h-screen overflow-hidden">
            <Sidebar />
            <div className="flex-1 flex flex-col min-w-0 min-h-0">
              <MobileNav />
              <main className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden">{children}</main>
            </div>
          </div>
          <Toaster position="bottom-right" />
        </ThemeProvider>
      </body>
    </html>
  );
}
