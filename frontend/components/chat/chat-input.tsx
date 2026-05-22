"use client";

import { Textarea } from "@/components/ui/textarea";
import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  onSubmit: (query: string, options: { approvalRequired: boolean }) => void;
  onCancel?: () => void;
  busy?: boolean;
  placeholder?: string;
  autoFocus?: boolean;
}

export interface ChatInputHandle {
  /** Phase 1.5: pre-fill the input with a query and focus, without auto-submitting. */
  prefill: (query: string) => void;
  focus: () => void;
}

export const ChatInput = forwardRef<ChatInputHandle, ChatInputProps>(function ChatInput(
  {
    onSubmit,
    onCancel,
    busy = false,
    placeholder = "Ask a research question…",
    autoFocus,
  },
  forwardedRef,
) {
  const [value, setValue] = useState("");
  // Phase 2: opt-in plan-approval gate. Persisted to localStorage so a user
  // who turns it on stays on it across page reloads — but only for THIS
  // browser; the request itself is stateless.
  const [approvePlan, setApprovePlan] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return window.localStorage.getItem("dra:approvePlanFirst") === "1";
  });
  const ref = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(
      "dra:approvePlanFirst",
      approvePlan ? "1" : "0",
    );
  }, [approvePlan]);

  useEffect(() => {
    if (autoFocus) ref.current?.focus();
  }, [autoFocus]);

  useImperativeHandle(forwardedRef, () => ({
    prefill: (q: string) => {
      setValue(q);
      // Defer focus to next tick so the new value lands before selection.
      setTimeout(() => {
        const el = ref.current;
        if (!el) return;
        el.focus();
        const len = q.length;
        try {
          el.setSelectionRange(len, len);
        } catch {
          /* some browsers throw for unfocused/non-text inputs */
        }
      }, 0);
    },
    focus: () => ref.current?.focus(),
  }));

  const submit = () => {
    const q = value.trim();
    if (!q || busy) return;
    onSubmit(q, { approvalRequired: approvePlan });
    setValue("");
  };

  return (
    <div
      className={cn(
        "relative border border-border bg-surface rounded-[8px]",
        "focus-within:border-border-accent transition-colors",
      )}
    >
      <Textarea
        ref={ref}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder={placeholder}
        rows={3}
        className="resize-none border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 px-4 py-3 text-[15px] font-sans placeholder:text-subtle-foreground"
      />
      <div className="flex items-center justify-between px-3 pb-2.5 gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground">
            Enter to send · Shift+Enter newline
          </span>
          <label
            htmlFor="approve-plan-first"
            title="Pause after planning to review and edit the research strategy before searching."
            className={cn(
              "flex items-center gap-1.5 cursor-pointer select-none",
              "font-mono text-[10px] uppercase tracking-[0.14em]",
              approvePlan
                ? "text-accent"
                : "text-subtle-foreground hover:text-muted-foreground",
            )}
          >
            <input
              id="approve-plan-first"
              type="checkbox"
              checked={approvePlan}
              onChange={(e) => setApprovePlan(e.target.checked)}
              className="size-3 accent-accent cursor-pointer"
            />
            Review plan first
          </label>
        </div>
        {busy ? (
          <button
            type="button"
            onClick={onCancel}
            className="font-mono text-[11px] uppercase tracking-[0.12em] text-destructive hover:text-destructive/80 px-2 py-1 transition-colors"
          >
            Cancel
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={!value.trim()}
            className={cn(
              "font-mono text-[11px] uppercase tracking-[0.12em] px-3 py-1.5 rounded-[6px] transition-colors",
              value.trim()
                ? "bg-accent text-accent-foreground hover:bg-teal-500"
                : "bg-surface-emphasis text-subtle-foreground cursor-not-allowed",
            )}
          >
            Send
          </button>
        )}
      </div>
    </div>
  );
});
