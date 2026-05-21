"use client";

import { Textarea } from "@/components/ui/textarea";
import { useState, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  onSubmit: (query: string) => void;
  onCancel?: () => void;
  busy?: boolean;
  placeholder?: string;
  autoFocus?: boolean;
}

export function ChatInput({
  onSubmit,
  onCancel,
  busy = false,
  placeholder = "Ask a research question…",
  autoFocus,
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    if (autoFocus) ref.current?.focus();
  }, [autoFocus]);

  const submit = () => {
    const q = value.trim();
    if (!q || busy) return;
    onSubmit(q);
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
      <div className="flex items-center justify-between px-3 pb-2.5">
        <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground">
          Enter to send · Shift+Enter newline
        </span>
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
}
