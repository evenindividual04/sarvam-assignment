"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import React from "react";
import { QuotePopover } from "@/components/trace/quote-popover";

interface RichMarkdownProps {
  children: string;
  className?: string;
  /**
   * Phase 1.875: set of numeric tokens (numbers/years/dates) the citation
   * guard could not ground in any cited document. When provided, the
   * markdown renderer wraps each occurrence inside a small ⚠ unverified
   * inline span. Pass `undefined` (or empty) to disable.
   */
  unverifiedNumericTokens?: ReadonlySet<string>;
  /**
   * B5: map of URL → verbatim quote extracted from the cited source.
   * When present, each citation anchor renders a hover tooltip showing
   * the quote so readers can verify the claim without opening the link.
   */
  citeQuoteByUrl?: Readonly<Record<string, string>>;
}

// Walks children, replacing literal "[UNVERIFIED]" tokens with an inline Badge.
function injectUnverified(nodes: React.ReactNode): React.ReactNode {
  return React.Children.map(nodes, (node) => {
    if (typeof node === "string") {
      if (!node.includes("[UNVERIFIED]")) return node;
      const parts = node.split("[UNVERIFIED]");
      const out: React.ReactNode[] = [];
      parts.forEach((part, i) => {
        out.push(part);
        if (i < parts.length - 1) {
          out.push(
            <span
              key={`uv-${i}`}
              className="inline-flex items-center font-mono text-[10px] uppercase tracking-[0.10em] px-1 py-px mx-0.5 align-middle rounded-[3px] border border-red-900/60 bg-red-950/40 text-red-300"
            >
              Unverified
            </span>,
          );
        }
      });
      return out;
    }
    return node;
  });
}

// Mirrors injectUnverified for `ambiguous_resolved` claims — claim_verifier's
// LLM-tier resolved the claim as supported, but the deterministic check was
// inconclusive. We want this visible to the reader without alarming them.
function injectAmbiguous(nodes: React.ReactNode): React.ReactNode {
  return React.Children.map(nodes, (node) => {
    if (typeof node === "string") {
      if (!node.includes("[AMBIGUOUS]")) return node;
      const parts = node.split("[AMBIGUOUS]");
      const out: React.ReactNode[] = [];
      parts.forEach((part, i) => {
        out.push(part);
        if (i < parts.length - 1) {
          out.push(
            <span
              key={`amb-${i}`}
              title="Resolved by LLM-tier verifier; deterministic check inconclusive"
              className="inline-flex items-center font-mono text-[10px] uppercase tracking-[0.10em] px-1 py-px mx-0.5 align-middle rounded-[3px] border border-border bg-muted text-subtle-foreground"
            >
              Ambiguous
            </span>,
          );
        }
      });
      return out;
    }
    return node;
  });
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Phase 1.875: wrap any numeric token the citation guard flagged as
// ungrounded with a small inline ⚠ unverified badge. Operates only on
// string children so we never mangle existing React nodes.
function injectUnverifiedNumeric(
  nodes: React.ReactNode,
  tokens: ReadonlySet<string>,
): React.ReactNode {
  if (tokens.size === 0) return nodes;
  const pattern = Array.from(tokens).map(escapeRegExp).join("|");
  const re = new RegExp(`(${pattern})`, "g");
  return React.Children.map(nodes, (node) => {
    if (typeof node !== "string") return node;
    const out: React.ReactNode[] = [];
    let lastIdx = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(node)) !== null) {
      if (m.index > lastIdx) out.push(node.slice(lastIdx, m.index));
      out.push(
        <span
          key={`unum-${m.index}`}
          title="Not found in cited sources"
          className="inline-flex items-baseline gap-0.5 font-mono text-[11px] px-1 py-px mx-0.5 align-baseline rounded-[3px] border border-amber-500/60 bg-amber-500/10 text-amber-700"
        >
          <span aria-hidden>⚠</span>
          {m[1]}
        </span>,
      );
      lastIdx = m.index + m[1].length;
    }
    if (lastIdx === 0) return node;
    if (lastIdx < node.length) out.push(node.slice(lastIdx));
    return out;
  });
}

export function RichMarkdown({
  children,
  className,
  unverifiedNumericTokens,
  citeQuoteByUrl,
}: RichMarkdownProps) {
  const tokens = unverifiedNumericTokens ?? new Set<string>();
  const quoteByUrl = citeQuoteByUrl ?? {};
  const decorate = (n: React.ReactNode): React.ReactNode =>
    injectUnverifiedNumeric(injectAmbiguous(injectUnverified(n)), tokens);
  return (
    <div className={cn("rich-md", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children: anchorChildren }) => {
            const quote = href ? quoteByUrl[href] : undefined;
            if (!href) {
              return <span>{anchorChildren}</span>;
            }
            if (!quote) {
              return (
                <a href={href} target="_blank" rel="noopener noreferrer">
                  {anchorChildren}
                </a>
              );
            }
            let domain: string | undefined;
            try {
              domain = new URL(href).hostname.replace(/^www\./, "");
            } catch {
              domain = undefined;
            }
            return (
              <QuotePopover href={href} quote={quote} domain={domain}>
                {anchorChildren}
              </QuotePopover>
            );
          },
          p: ({ children, ...rest }) => <p {...rest}>{decorate(children)}</p>,
          li: ({ children, ...rest }) => <li {...rest}>{decorate(children)}</li>,
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
