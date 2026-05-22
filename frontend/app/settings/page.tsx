"use client";

import { useEffect, useState } from "react";
import { getSettingsDefaults } from "@/lib/api";
import {
  clearOverrides,
  loadOverrides,
  saveOverrides,
  type DefaultsResponse,
  type KnobDef,
  type RuntimeOverrides,
} from "@/lib/settings";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

export default function SettingsPage() {
  const [defaults, setDefaults] = useState<DefaultsResponse | null>(null);
  const [overrides, setOverrides] = useState<RuntimeOverrides>(() =>
    loadOverrides(),
  );
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getSettingsDefaults()
      .then(setDefaults)
      .catch((e) => setErr(e instanceof Error ? e.message : "error"))
      .finally(() => setLoading(false));
  }, []);

  const setOverride = <K extends keyof RuntimeOverrides>(
    key: K,
    value: RuntimeOverrides[K] | undefined,
  ) => {
    const next: RuntimeOverrides = { ...overrides };
    if (value === undefined) {
      delete next[key];
    } else {
      next[key] = value;
    }
    setOverrides(next);
    saveOverrides(next);
  };

  const resetAll = () => {
    clearOverrides();
    setOverrides({});
    toast.info("Overrides cleared. Falling back to env defaults.");
  };

  return (
    <div className="px-8 md:px-12 py-12 max-w-[760px] mx-auto w-full">
      <h1 className="text-2xl font-sans font-medium tracking-tight">
        Runtime Settings
      </h1>
      <p className="mt-1 font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
        Per-request overrides · stored locally · sent with each turn
      </p>

      <div className="mt-3 font-mono text-[11px] text-subtle-foreground leading-relaxed max-w-[560px]">
        These knobs are read at request time. Eval runs from the CLI ignore
        them and stay reproducible from env vars. Every saved turn stamps the
        resolved config into its trace, so changes here don&apos;t rewrite history.
        Secrets (API keys, DB path) are never exposed here by design.
      </div>

      {/* V3.11: active-overrides banner makes it impossible to forget that
          settings flips persist across browser sessions until cleared. */}
      {Object.keys(overrides).length > 0 && (
        <div className="mt-6 border border-accent/40 bg-accent-dim rounded-[6px] px-4 py-3 flex items-center justify-between gap-4">
          <div className="space-y-0.5">
            <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-accent">
              {Object.keys(overrides).length} override
              {Object.keys(overrides).length === 1 ? "" : "s"} active
            </div>
            <div className="font-mono text-[11px] text-muted-foreground">
              Sent with every research request from this browser until cleared.
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={resetAll}
            className="font-mono text-[10px] uppercase tracking-[0.12em] shrink-0"
          >
            Reset all
          </Button>
        </div>
      )}

      {loading && (
        <div className="mt-10 space-y-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="h-16 rounded-[4px] animate-pulse bg-[var(--surface-hover)]" />
          ))}
        </div>
      )}
      {err && !loading && (
        <div className="mt-10 font-mono text-[11px] uppercase tracking-[0.12em] text-subtle-foreground">
          Backend unreachable. Defaults will load on next refresh.
        </div>
      )}

      {defaults && (
        <>
          <div className="mt-12 mb-6 flex items-baseline justify-between">
            <h2 className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              Knobs
            </h2>
            {Object.keys(overrides).length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={resetAll}
                className="font-mono text-[10px] uppercase tracking-[0.14em]"
              >
                Reset all to env defaults
              </Button>
            )}
          </div>

          <div className="border-t border-border">
            {defaults.knobs.map((knob) => (
              <KnobRow
                key={knob.key}
                knob={knob}
                value={overrides[knob.key]}
                onChange={(v) => setOverride(knob.key, v)}
                defaultValue={resolveDefault(knob, defaults)}
              />
            ))}
          </div>

          <div className="mt-12 pt-6 border-t border-border">
            <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground mb-3">
              Effective config (no overrides)
            </div>
            <pre className="font-mono text-[11px] text-muted-foreground bg-surface border border-border rounded-[6px] p-4 overflow-x-auto">
{JSON.stringify(defaults.effective, null, 2)}
            </pre>
          </div>
        </>
      )}
    </div>
  );
}

function resolveDefault(knob: KnobDef, defaults: DefaultsResponse): string {
  const e = defaults.effective as Record<string, unknown>;
  if (knob.key === "RETRIEVAL_MODE") return String(e.retrieval_mode ?? "auto");
  if (knob.key === "FAILURE_POLICY_MAX_HOPS") return String(e.max_hops);
  if (knob.key === "CONTEXT_SELECTION_STRATEGY")
    return String(e.selection_strategy);
  if (knob.key === "RETRIEVAL_DOMAIN_BLOCKLIST") {
    const list = e.domain_blocklist;
    if (Array.isArray(list) && list.length > 0) {
      return list.length > 4 ? `${list.length} domains` : list.join(",");
    }
    return "none";
  }
  return "—";
}

interface KnobRowProps {
  knob: KnobDef;
  value: RuntimeOverrides[keyof RuntimeOverrides] | undefined;
  onChange: (v: RuntimeOverrides[keyof RuntimeOverrides] | undefined) => void;
  defaultValue: string;
}

function KnobRow({ knob, value, onChange, defaultValue }: KnobRowProps) {
  const overridden = value !== undefined;
  return (
    <div
      className={
        "grid grid-cols-[1fr_220px] gap-6 py-5 pl-4 border-b border-border items-start border-l-2 " +
        (overridden ? "border-l-accent" : "border-l-transparent")
      }
    >
      <div>
        <div className="text-[14px] text-foreground">{knob.label}</div>
        <div className="mt-1 font-mono text-[10px] uppercase tracking-[0.12em] text-subtle-foreground">
          {knob.key}
          <span className="ml-2 normal-case tracking-normal">
            env default: <span className="text-muted-foreground">{defaultValue}</span>
          </span>
          {overridden && (
            <span className="ml-2 normal-case tracking-normal text-accent">
              · overridden
            </span>
          )}
        </div>
        {knob.note && (
          <div className="mt-2 font-mono text-[10px] text-subtle-foreground leading-relaxed">
            {knob.note}
          </div>
        )}
      </div>

      <div className="flex flex-col items-end gap-2">
        {knob.type === "boolean" && (
          <BoolControl
            value={value as "0" | "1" | undefined}
            disabled={knob.available === false}
            onChange={onChange as (v: "0" | "1" | undefined) => void}
          />
        )}
        {knob.type === "integer" && (
          <IntControl
            value={value as number | undefined}
            min={knob.min}
            max={knob.max}
            onChange={onChange as (v: number | undefined) => void}
          />
        )}
        {knob.type === "enum" && (
          <EnumControl
            value={value as string | undefined}
            choices={knob.choices || []}
            effectivePerChoice={knob.effective_per_choice}
            onChange={onChange as (v: string | undefined) => void}
          />
        )}
        {knob.type === "string" && (
          <StringControl
            value={value as string | undefined}
            onChange={onChange as (v: string | undefined) => void}
          />
        )}
      </div>
    </div>
  );
}

function StringControl({
  value,
  onChange,
}: {
  value: string | undefined;
  onChange: (v: string | undefined) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <button
        onClick={() => onChange(undefined)}
        className={`font-mono text-[11px] uppercase tracking-[0.12em] px-3 py-1.5 border ${
          value === undefined
            ? "border-accent text-accent bg-accent-dim"
            : "border-border text-muted-foreground hover:border-border-strong"
        }`}
      >
        default
      </button>
      <input
        type="text"
        value={value ?? ""}
        placeholder="extra.com, another.com  (or 'none')"
        onChange={(e) => {
          const v = e.target.value;
          onChange(v === "" ? undefined : v);
        }}
        className="w-[240px] font-mono text-[12px] bg-background border border-border rounded-[4px] px-2 py-1.5 text-foreground"
      />
    </div>
  );
}

function BoolControl({
  value,
  disabled,
  onChange,
}: {
  value: "0" | "1" | undefined;
  disabled?: boolean;
  onChange: (v: "0" | "1" | undefined) => void;
}) {
  const pickerCls = (active: boolean) =>
    `font-mono text-[11px] uppercase tracking-[0.12em] px-3 py-1.5 border ${
      active
        ? "border-accent text-accent bg-accent-dim"
        : "border-border text-muted-foreground hover:border-border-strong"
    } ${disabled ? "opacity-40 cursor-not-allowed" : ""}`;
  return (
    <div className="flex gap-1">
      <button
        disabled={disabled}
        onClick={() => onChange(undefined)}
        className={pickerCls(value === undefined)}
      >
        default
      </button>
      <button
        disabled={disabled}
        onClick={() => onChange("0")}
        className={pickerCls(value === "0")}
      >
        off
      </button>
      <button
        disabled={disabled}
        onClick={() => onChange("1")}
        className={pickerCls(value === "1")}
      >
        on
      </button>
    </div>
  );
}

function IntControl({
  value,
  min,
  max,
  onChange,
}: {
  value: number | undefined;
  min?: number;
  max?: number;
  onChange: (v: number | undefined) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <button
        onClick={() => onChange(undefined)}
        className={`font-mono text-[11px] uppercase tracking-[0.12em] px-3 py-1.5 border ${
          value === undefined
            ? "border-accent text-accent bg-accent-dim"
            : "border-border text-muted-foreground hover:border-border-strong"
        }`}
      >
        default
      </button>
      <input
        type="number"
        min={min}
        max={max}
        value={value ?? ""}
        placeholder={`${min ?? ""}–${max ?? ""}`}
        onChange={(e) => {
          const v = e.target.value;
          onChange(v === "" ? undefined : Number(v));
        }}
        className="w-20 font-mono text-[12px] tabular-nums bg-background border border-border rounded-[4px] px-2 py-1.5 text-foreground"
      />
    </div>
  );
}

function EnumControl({
  value,
  choices,
  effectivePerChoice,
  onChange,
}: {
  value: string | undefined;
  choices: string[];
  effectivePerChoice?: Record<string, string>;
  onChange: (v: string | undefined) => void;
}) {
  const pickerCls = (active: boolean, unavailable?: boolean) =>
    `font-mono text-[11px] uppercase tracking-[0.12em] px-3 py-1.5 border ${
      unavailable
        ? "opacity-50 cursor-not-allowed border-border text-muted-foreground"
        : active
        ? "border-accent text-accent bg-accent-dim"
        : "border-border text-muted-foreground hover:border-border-strong"
    }`;
  return (
    <div className="flex flex-col items-end gap-2">
      <div className="flex flex-wrap gap-1 justify-end">
        <button
          onClick={() => onChange(undefined)}
          className={pickerCls(value === undefined)}
        >
          default
        </button>
        {choices.map((c) => {
          const eff = effectivePerChoice?.[c];
          const unavailable = eff === "unavailable";
          return (
            <button
              key={c}
              disabled={unavailable}
              onClick={() => onChange(c)}
              title={
                eff && eff !== c
                  ? `→ effective: ${eff}`
                  : unavailable
                  ? "Not available on this host"
                  : undefined
              }
              className={pickerCls(value === c, unavailable)}
            >
              {c}
              {eff && eff !== c && eff !== "unavailable" && (
                <span className="ml-1 opacity-60 normal-case tracking-normal">
                  → {eff}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
