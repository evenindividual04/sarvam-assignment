"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import React from "react";

interface RichMarkdownProps {
  children: string;
  className?: string;
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

export function RichMarkdown({ children, className }: RichMarkdownProps) {
  return (
    <div className={cn("rich-md", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children, ...rest }) => (
            <a href={href} target="_blank" rel="noopener noreferrer" {...rest}>
              {children}
            </a>
          ),
          p: ({ children, ...rest }) => (
            <p {...rest}>{injectUnverified(children)}</p>
          ),
          li: ({ children, ...rest }) => (
            <li {...rest}>{injectUnverified(children)}</li>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
