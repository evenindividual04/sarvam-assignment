"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { BACKEND, cancelResearch } from "./api";
import type { DoneEventData, ExecutionEvent } from "./types";

export type SseStatus = "idle" | "streaming" | "done" | "cancelled" | "error";

export interface UseSseResearchReturn {
  events: ExecutionEvent[];
  status: SseStatus;
  finalData: DoneEventData | null;
  error: string | null;
  turnId: string | null;
  currentText: string;
  start: (query: string, sessionId: string) => Promise<void>;
  cancel: () => void;
  reset: () => void;
}

// Parses SSE `data:` frames out of a streaming POST response body.
export function useSseResearch(): UseSseResearchReturn {
  const [events, setEvents] = useState<ExecutionEvent[]>([]);
  const [status, setStatus] = useState<SseStatus>("idle");
  const [finalData, setFinalData] = useState<DoneEventData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [turnId, setTurnId] = useState<string | null>(null);
  const [currentText, setCurrentText] = useState<string>("");

  const abortRef = useRef<AbortController | null>(null);
  const turnIdRef = useRef<string | null>(null);

  const reset = useCallback(() => {
    setEvents([]);
    setStatus("idle");
    setFinalData(null);
    setError(null);
    setTurnId(null);
    setCurrentText("");
    turnIdRef.current = null;
  }, []);

  const cancel = useCallback(() => {
    const tid = turnIdRef.current;
    if (tid) void cancelResearch(tid);
    abortRef.current?.abort();
  }, []);

  // On hook unmount (e.g. navigating to /sessions or /eval mid-stream),
  // abort the in-flight fetch so we don't leak a detached reader/decoder.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const start = useCallback(async (query: string, sessionId: string) => {
    reset();
    setStatus("streaming");
    const ac = new AbortController();
    abortRef.current = ac;

    try {
      const res = await fetch(`${BACKEND}/research`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, session_id: sessionId }),
        signal: ac.signal,
      });
      if (!res.ok || !res.body) {
        throw new Error(`Server returned ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          // SSE messages may have multiple lines; we only need `data:` lines.
          const dataLine = part
            .split("\n")
            .find((l) => l.startsWith("data: "));
          if (!dataLine) continue;
          const raw = dataLine.slice(6);
          let ev: ExecutionEvent;
          try {
            ev = JSON.parse(raw) as ExecutionEvent;
          } catch {
            continue;
          }

          // Capture turn_id from the first planning event.
          if (ev.step === "planning") {
            const d = ev.data as { turn_id?: string } | undefined;
            if (d?.turn_id) {
              turnIdRef.current = d.turn_id;
              setTurnId(d.turn_id);
            }
          }

          if (ev.step === "generating" && typeof ev.data === "string") {
            setCurrentText((prev) => prev + ev.data);
          }

          if (ev.step === "done") {
            setFinalData(ev.data as DoneEventData);
            setStatus("done");
          }

          if (ev.step === "error") {
            if (ev.label === "cancelled") {
              setStatus("cancelled");
            } else {
              setStatus("error");
              setError(
                typeof ev.data === "string"
                  ? ev.data
                  : ev.label || "Stream error",
              );
            }
          }

          setEvents((prev) => [...prev, ev]);
        }
      }
    } catch (e: unknown) {
      // AbortController.abort() → DOMException name 'AbortError'.
      if (e instanceof DOMException && e.name === "AbortError") {
        setStatus((s) => (s === "cancelled" || s === "done" ? s : "cancelled"));
        return;
      }
      setStatus("error");
      setError(e instanceof Error ? e.message : "Unknown error");
    }
  }, [reset]);

  return {
    events,
    status,
    finalData,
    error,
    turnId,
    currentText,
    start,
    cancel,
    reset,
  };
}
