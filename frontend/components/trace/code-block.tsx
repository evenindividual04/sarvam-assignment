"use client";

import { useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

interface CodeBlockProps {
  code: string;
  language?: string;
  maxHeight?: string;
}

export function CodeBlock({
  code,
  language = "xml",
  maxHeight = "60vh",
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code ?? "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* no-op */
    }
  };
  return (
    <div className="relative border border-border rounded-[6px] overflow-hidden bg-[#0a0a0c]">
      <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-surface">
        <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
          {language}
        </span>
        <button
          onClick={onCopy}
          className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground hover:text-accent transition-colors"
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <div style={{ maxHeight, overflow: "auto" }}>
        <SyntaxHighlighter
          language={language}
          style={oneDark}
          customStyle={{
            margin: 0,
            padding: "0.9rem 1rem",
            background: "transparent",
            fontSize: "12px",
            lineHeight: 1.6,
            fontFamily: "var(--font-mono)",
          }}
          wrapLongLines
        >
          {code || ""}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
