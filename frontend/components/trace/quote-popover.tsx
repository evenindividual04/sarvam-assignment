"use client";

// Inline citation popover with verbatim quote + keyword highlighting.
//
// Anti-copying note: competitor exposes only the URL behind each [N] citation.
// Ours shows the verbatim quote AND bolds the words from the surrounding
// claim sentence that actually appear in the quote — turning every citation
// into a click-free verification surface (cf. Perplexity / OpenAI Deep
// Research). No new dependency; lightweight portal-free positioning.

import React, { useId, useMemo, useState } from "react";

interface QuotePopoverProps {
  href: string;
  quote: string;
  /** Surrounding claim text used for keyword highlighting. */
  claim?: string;
  /** Optional source domain rendered in the popover header. */
  domain?: string;
  children: React.ReactNode;
}

const STOPWORDS = new Set([
  "the",
  "a",
  "an",
  "of",
  "in",
  "on",
  "at",
  "to",
  "for",
  "and",
  "or",
  "but",
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "being",
  "this",
  "that",
  "these",
  "those",
  "it",
  "its",
  "with",
  "by",
  "from",
  "as",
  "also",
  "not",
  "no",
  "yes",
  "has",
  "have",
  "had",
  "do",
  "does",
  "did",
  "will",
  "would",
  "should",
  "could",
  "than",
  "then",
  "so",
  "if",
  "into",
  "over",
  "about",
]);

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightKeywords(quote: string, claim: string): React.ReactNode {
  if (!claim) return quote;
  const claimTokens = Array.from(
    new Set(
      claim
        .toLowerCase()
        .match(/\b[\w-]{4,}\b/g)
        ?.filter((t) => !STOPWORDS.has(t)) ?? [],
    ),
  );
  if (claimTokens.length === 0) return quote;

  const pattern = new RegExp(
    `\\b(${claimTokens.map(escapeRegExp).join("|")})\\b`,
    "gi",
  );
  const parts: React.ReactNode[] = [];
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(quote)) !== null) {
    if (match.index > last) parts.push(quote.slice(last, match.index));
    parts.push(
      <mark
        key={`m-${match.index}`}
        className="bg-accent/15 text-foreground rounded-[2px] px-[1px] font-medium"
      >
        {match[1]}
      </mark>,
    );
    last = match.index + match[1].length;
  }
  if (last < quote.length) parts.push(quote.slice(last));
  return parts;
}

export function QuotePopover({
  href,
  quote,
  claim,
  domain,
  children,
}: QuotePopoverProps) {
  const [open, setOpen] = useState(false);
  const popoverId = useId();
  const highlighted = useMemo(
    () => highlightKeywords(quote, claim ?? ""),
    [quote, claim],
  );

  if (!quote) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="cite-anchor"
      >
        {children}
      </a>
    );
  }

  return (
    <span className="relative inline-block align-baseline">
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        aria-describedby={open ? popoverId : undefined}
        className="cite-anchor underline decoration-dotted decoration-accent/60 underline-offset-2"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={(e) => {
          // Tap toggles on touch; let the cmd/ctrl-click and middle-click open the link.
          if (e.metaKey || e.ctrlKey || e.button === 1) return;
          if (!open) {
            e.preventDefault();
            setOpen(true);
          }
        }}
        onKeyDown={(e) => {
          if (e.key === "Escape") setOpen(false);
        }}
      >
        {children}
      </a>
      {open && (
        <span
          id={popoverId}
          role="tooltip"
          className="absolute z-50 left-0 top-full mt-1 w-[min(420px,calc(100vw-2rem))] rounded-md border border-border bg-popover shadow-md p-3 text-[12px] text-foreground leading-snug pointer-events-none"
        >
          {domain && (
            <span className="block font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground mb-1.5">
              {domain}
            </span>
          )}
          <span className="block italic text-foreground/90">
            “{highlighted}”
          </span>
        </span>
      )}
    </span>
  );
}
