"use client";

import { cn } from "@/lib/utils";

export type SourceChipStatus = "pending" | "ok" | "error";

interface SourceChipProps {
  url: string;
  title: string;
  domain: string;
  status?: SourceChipStatus;
}

const STATUS_CLASS: Record<SourceChipStatus, string> = {
  pending: "border-border bg-surface text-muted-foreground",
  ok: "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
  error: "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-300",
};

/**
 * Phase 1.25: per-source chip rendered on the pipeline rail. Color reflects
 * fetch status (pending → ok → error). Favicon uses Google's S2 endpoint to
 * avoid bundling a heavy favicon library.
 */
export function SourceChip({
  url,
  title,
  domain,
  status = "pending",
}: SourceChipProps) {
  const short = title.length > 40 ? `${title.slice(0, 37)}...` : title;
  const favicon = `https://www.google.com/s2/favicons?domain=${encodeURIComponent(domain || "")}&sz=32`;
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      title={`${title} — ${domain}`}
      className={cn(
        "inline-flex items-center gap-1.5 max-w-[200px] px-2 py-1 rounded-[4px] border text-[10px] font-mono truncate transition-colors",
        STATUS_CLASS[status],
      )}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={favicon}
        alt=""
        width={12}
        height={12}
        className="shrink-0"
        loading="lazy"
      />
      <span className="font-medium shrink-0">{domain}</span>
      <span className="text-subtle-foreground truncate">{short}</span>
    </a>
  );
}
