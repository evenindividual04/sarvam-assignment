"use client";

/**
 * Ephemeral-storage banner.
 *
 * Fetches `/health/storage` on mount and renders a dismissable warning
 * banner when the backend reports `persistent: false` — i.e. the agent is
 * writing to the container's local filesystem and every redeploy or
 * Space restart wipes session + turn history.
 *
 * The banner is purely informational; nothing in the UI changes
 * functionally based on persistence state. Dismissal is per-session
 * (sessionStorage) so the user isn't nagged on every page navigation
 * within the same tab, but a fresh tab gets the warning again until
 * the deployer wires up `DB_PATH=/data/research.db` on the Space.
 */

import { useEffect, useState } from "react";
import { BACKEND } from "@/lib/api";

interface StorageStatus {
  db_path: string;
  persistent: boolean;
  size_bytes: number;
  row_counts?: Record<string, number | string>;
}

const DISMISS_KEY = "dra:ephemeralBannerDismissed";

export function EphemeralDbBanner() {
  const [status, setStatus] = useState<StorageStatus | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      if (window.sessionStorage.getItem(DISMISS_KEY) === "1") {
        setDismissed(true);
        return;
      }
    }
    let alive = true;
    fetch(`${BACKEND}/health/storage`)
      .then((r) => (r.ok ? r.json() : null))
      .then((s: StorageStatus | null) => {
        if (!alive) return;
        setStatus(s);
      })
      .catch(() => {
        // Health endpoint not reachable — don't render anything; we're
        // not going to spam a banner about a banner-source being down.
      });
    return () => {
      alive = false;
    };
  }, []);

  if (dismissed) return null;
  if (!status) return null;
  if (status.persistent) return null;

  const handleDismiss = () => {
    setDismissed(true);
    if (typeof window !== "undefined") {
      window.sessionStorage.setItem(DISMISS_KEY, "1");
    }
  };

  return (
    <div
      role="status"
      aria-live="polite"
      className="border-b border-amber-500/40 bg-amber-500/10 px-6 py-2 flex items-center justify-between gap-4"
    >
      <div className="flex items-baseline gap-3 min-w-0">
        <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-amber-700">
          Ephemeral storage
        </span>
        <span className="font-sans text-[12px] text-foreground truncate">
          Sessions and turns are written to <code className="font-mono text-[11px]">{status.db_path}</code>
          {" — this directory is not persistent on the deployed host."}
          {" Set "}
          <code className="font-mono text-[11px]">DB_PATH=/data/research.db</code>
          {" in the Space environment to keep history across restarts."}
        </span>
      </div>
      <button
        type="button"
        onClick={handleDismiss}
        className="font-mono text-[10px] uppercase tracking-[0.12em] text-amber-700 hover:text-amber-900 shrink-0"
      >
        Dismiss
      </button>
    </div>
  );
}
