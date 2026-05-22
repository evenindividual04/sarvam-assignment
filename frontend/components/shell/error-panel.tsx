"use client";

import { Button } from "@/components/ui/button";
import { BACKEND } from "@/lib/api";

interface ErrorPanelProps {
  /** Shown as the panel headline. Defaults to "Backend unreachable". */
  title?: string;
  /** Underlying error message for diagnosis. Rendered in muted mono. */
  detail?: string;
  /** Click handler for the retry button. When omitted, the button is hidden. */
  onRetry?: () => void;
}

/**
 * Replaces the previous bare "backend unreachable." text on every page. The
 * HF Space sleeps after inactivity on the free tier — the first request
 * after a wake takes ~30s. A clear hint + retry button stops reviewers
 * from concluding the deployment is broken.
 */
export function ErrorPanel({
  title = "Backend unreachable",
  detail,
  onRetry,
}: ErrorPanelProps) {
  return (
    <div className="border border-border bg-surface rounded-[8px] p-6 max-w-[640px] space-y-3">
      <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-destructive">
        {title}
      </div>
      <p className="text-[14px] leading-normal max-w-prose text-foreground">
        The backend isn’t responding. The deployed Hugging Face Space
        sleeps on the free tier — the first request after a wake takes about
        30 seconds. Try again in a moment.
      </p>
      <div className="font-mono text-[10px] text-subtle-foreground break-all">
        Backend: {BACKEND}
      </div>
      {detail && (
        <details className="font-mono text-[10px] text-muted-foreground">
          <summary className="cursor-pointer hover:text-foreground transition-colors uppercase tracking-[0.12em]">
            Error detail
          </summary>
          <div className="mt-2 whitespace-pre-wrap">{detail}</div>
        </details>
      )}
      {onRetry && (
        <div className="pt-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={onRetry}
            className="font-mono text-[11px] uppercase tracking-[0.12em]"
          >
            ↻ Retry
          </Button>
        </div>
      )}
    </div>
  );
}
