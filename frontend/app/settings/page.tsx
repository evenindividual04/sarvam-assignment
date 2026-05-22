"use client";

import { useEffect, useMemo, useState } from "react";
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
import { Skeleton } from "@/components/ui/skeleton";
import { toast } from "sonner";

type SectionKey =
  | "Retrieval"
  | "Routing"
  | "Approval gate"
  | "Supplementary providers";

const SECTION_ORDER: SectionKey[] = [
  "Retrieval",
  "Routing",
  "Approval gate",
  "Supplementary providers",
];

// Map each knob key to a UI section. Anything unmapped falls into "Retrieval"
// so the page never silently drops a knob the backend just learned.
const KNOB_SECTION: Record<string, SectionKey> = {
  RETRIEVAL_MODE: "Retrieval",
  FAILURE_POLICY_MAX_HOPS: "Retrieval",
  CONTEXT_SELECTION_STRATEGY: "Retrieval",
  MMR_LAMBDA: "Retrieval",
  RETRIEVAL_DOMAIN_BLOCKLIST: "Retrieval",
  PLANNER_PROVIDER: "Routing",
  CLAIM_VERIFIER_PROVIDER: "Routing",
  CONFLICT_PROBE_PROVIDER: "Routing",
  FOLLOW_UP_PROVIDER: "Routing",
  APPROVAL_TIMEOUT_S: "Approval gate",
  SARVAM_INDIC_AUTO: "Supplementary providers",
  LANG_DETECT_DISABLE_FASTTEXT: "Supplementary providers",
  WIKIPEDIA_DISABLED: "Supplementary providers",
  SCHOLAR_DISABLED: "Supplementary providers",
  JINA_READER_DISABLED: "Supplementary providers",
};

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

  const setOverride = (
    key: string,
    value: RuntimeOverrides[keyof RuntimeOverrides] | undefined,
  ) => {
    const next = { ...overrides } as Record<string, unknown>;
    if (value === undefined) {
      delete next[key];
    } else {
      next[key] = value;
    }
    setOverrides(next as RuntimeOverrides);
    saveOverrides(next as RuntimeOverrides);
  };

  const resetAll = () => {
    clearOverrides();
    setOverrides({});
    toast.info("Overrides cleared. Falling back to env defaults.");
  };

  const grouped = useMemo(() => {
    const g: Record<SectionKey, KnobDef[]> = {
      Retrieval: [],
      Routing: [],
      "Approval gate": [],
      "Supplementary providers": [],
    };
    if (!defaults) return g;
    for (const k of defaults.knobs) {
      const section = (KNOB_SECTION[k.key] ?? "Retrieval") as SectionKey;
      g[section].push(k);
    }
    return g;
  }, [defaults]);

  return (
    <div className="px-6 md:px-12 py-12 max-w-5xl mx-auto w-full">
      <h1 className="text-2xl font-sans font-medium tracking-tight">
        Runtime Settings
      </h1>
      <p className="mt-1 font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
        Per-request overrides · stored locally · sent with each turn
      </p>

      <p className="mt-3 font-sans text-[13px] text-muted-foreground leading-normal max-w-prose">
        These knobs are read at request time. Eval runs from the CLI ignore them
        and stay reproducible from env vars. Every saved turn stamps the
        resolved config into its trace, so changes here don’t rewrite
        history. Secrets (API keys, DB path) are never exposed here by design.
      </p>

      {Object.keys(overrides).length > 0 && (
        <div className="mt-6 border border-accent/40 bg-accent-dim rounded-[6px] px-4 py-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div className="space-y-0.5">
            <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-accent">
              {Object.keys(overrides).length} override
              {Object.keys(overrides).length === 1 ? "" : "s"} active
            </div>
            <div className="font-sans text-[12px] text-muted-foreground">
              Sent with every research request from this browser until cleared.
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={resetAll}
            className="font-mono text-[10px] uppercase tracking-[0.12em] shrink-0 self-start sm:self-auto"
          >
            Reset all
          </Button>
        </div>
      )}

      {loading && (
        <div className="mt-10 space-y-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-full" />
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
          {SECTION_ORDER.map((section) => {
            const knobs = grouped[section];
            if (knobs.length === 0) return null;
            return (
              <section key={section} className="mt-12">
                <div className="flex items-baseline justify-between border-b border-border pb-3 mb-2">
                  <h2 className="font-sans text-[14px] font-medium tracking-tight text-foreground">
                    {section}
                  </h2>
                  <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-subtle-foreground">
                    {knobs.length} knob{knobs.length === 1 ? "" : "s"}
                  </span>
                </div>
                {knobs.map((knob) => (
                  <KnobRow
                    key={knob.key}
                    knob={knob}
                    value={
                      (overrides as Record<string, unknown>)[knob.key] as
                        | RuntimeOverrides[keyof RuntimeOverrides]
                        | undefined
                    }
                    onChange={(v) => setOverride(knob.key, v)}
                    defaultValue={resolveDefault(knob, defaults)}
                  />
                ))}
              </section>
            );
          })}

          <section className="mt-14 pt-6 border-t border-border">
            <div className="flex items-baseline justify-between mb-3">
              <h2 className="font-sans text-[14px] font-medium tracking-tight text-foreground">
                Effective config
              </h2>
              <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-subtle-foreground">
                {Object.keys(overrides).length === 0
                  ? "no overrides — pure env defaults"
                  : `${Object.keys(overrides).length} override(s) layered on top`}
              </span>
            </div>
            <p className="font-sans text-[12px] text-muted-foreground mb-3">
              Effective baseline returned by{" "}
              <code className="font-mono text-[11px] bg-surface-hover border border-border rounded-[3px] px-1 py-px">
                GET /settings/defaults
              </code>
              , merged with active overrides (if any).
            </p>
            <pre className="font-mono text-[11px] text-muted-foreground bg-surface border border-border rounded-[6px] p-4 overflow-x-auto leading-normal tabular-nums-lining">
              {JSON.stringify(mergedEffective(defaults, overrides), null, 2)}
            </pre>
            <div className="mt-2 font-mono text-[10px] uppercase tracking-[0.14em] text-subtle-foreground">
              Local overrides:{" "}
              {Object.keys(overrides).length === 0
                ? "{}"
                : JSON.stringify(overrides)}
            </div>
          </section>
        </>
      )}
    </div>
  );
}

/**
 * Returns the merged "what would actually be in effect right now" payload —
 * backend's effective baseline overlaid with the user's local overrides. We
 * keep the merge shallow because overrides are flat string/number values.
 */
function mergedEffective(
  defaults: DefaultsResponse,
  overrides: RuntimeOverrides,
): Record<string, unknown> {
  const base: Record<string, unknown> = { ...defaults.effective };
  for (const [k, v] of Object.entries(overrides)) {
    if (v === undefined) continue;
    base[k] = v;
  }
  return base;
}

function resolveDefault(knob: KnobDef, defaults: DefaultsResponse): string {
  const e = defaults.effective as Record<string, unknown>;
  if (knob.key === "RETRIEVAL_MODE") return String(e.retrieval_mode ?? "auto");
  if (knob.key === "FAILURE_POLICY_MAX_HOPS") return String(e.max_hops);
  if (knob.key === "CONTEXT_SELECTION_STRATEGY")
    return String(e.selection_strategy);
  if (knob.key === "MMR_LAMBDA")
    return e.mmr_lambda != null ? String(e.mmr_lambda) : "0.7";
  if (knob.key === "RETRIEVAL_DOMAIN_BLOCKLIST") {
    const list = e.domain_blocklist;
    if (Array.isArray(list) && list.length > 0) {
      return list.length > 4 ? `${list.length} domains` : list.join(",");
    }
    return "none";
  }
  return "env";
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
        "grid grid-cols-1 md:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] gap-4 md:gap-8 py-5 pl-4 border-b border-border items-start border-l-2 " +
        (overridden ? "border-l-accent" : "border-l-transparent")
      }
    >
      <div className="min-w-0">
        <div className="text-[14px] font-sans text-foreground">{knob.label}</div>
        <div className="mt-1 font-mono text-[10px] uppercase tracking-[0.12em] text-subtle-foreground break-all">
          {knob.key}
          <span className="ml-2 normal-case tracking-normal">
            env default:{" "}
            <span className="text-muted-foreground">{defaultValue}</span>
          </span>
          {overridden && (
            <span className="ml-2 normal-case tracking-normal text-accent">
              · overridden
            </span>
          )}
        </div>
        {knob.note && (
          <div className="mt-2 font-sans text-[12px] text-muted-foreground leading-relaxed">
            {knob.note}
          </div>
        )}
      </div>

      <div className="min-w-0 md:justify-self-end w-full md:max-w-full">
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
        {(knob.type === "float") && (
          <FloatControl
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

/**
 * Unified chip styling so every selectable control across the page reads as
 * one component family — bordered when inactive, filled teal when active.
 */
function chipClass(active: boolean, disabled?: boolean): string {
  const base =
    "font-mono text-[11px] uppercase tracking-[0.12em] px-3 py-1.5 border rounded-[4px] transition-colors shrink-0";
  if (disabled) {
    return `${base} opacity-40 cursor-not-allowed border-border text-muted-foreground`;
  }
  return active
    ? `${base} border-accent bg-accent text-accent-foreground`
    : `${base} border-border text-muted-foreground hover:border-border-strong hover:text-foreground`;
}

function StringControl({
  value,
  onChange,
}: {
  value: string | undefined;
  onChange: (v: string | undefined) => void;
}) {
  return (
    <div className="flex flex-col gap-2 w-full">
      <div className="flex items-center gap-2 flex-wrap">
        <button
          type="button"
          onClick={() => onChange(undefined)}
          className={chipClass(value === undefined)}
        >
          default
        </button>
      </div>
      <input
        type="text"
        value={value ?? ""}
        placeholder="extra.com, another.com (or ‘none’)"
        aria-label="Domain blocklist override"
        onChange={(e) => {
          const v = e.target.value;
          onChange(v === "" ? undefined : v);
        }}
        className="w-full min-w-0 font-mono text-[12px] bg-background border border-border rounded-[4px] px-3 py-2 text-foreground focus:outline-none focus:border-accent"
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
  return (
    <div className="flex flex-nowrap gap-1 overflow-x-auto md:justify-end">
      <button
        type="button"
        disabled={disabled}
        onClick={() => onChange(undefined)}
        className={chipClass(value === undefined, disabled)}
      >
        default
      </button>
      <button
        type="button"
        disabled={disabled}
        onClick={() => onChange("0")}
        className={chipClass(value === "0", disabled)}
      >
        off
      </button>
      <button
        type="button"
        disabled={disabled}
        onClick={() => onChange("1")}
        className={chipClass(value === "1", disabled)}
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
    <div className="flex flex-nowrap items-center gap-2 overflow-x-auto md:justify-end">
      <button
        type="button"
        onClick={() => onChange(undefined)}
        className={chipClass(value === undefined)}
      >
        default
      </button>
      <input
        type="number"
        min={min}
        max={max}
        value={value ?? ""}
        placeholder={`${min ?? ""}–${max ?? ""}`}
        aria-label="Override value"
        onChange={(e) => {
          const v = e.target.value;
          onChange(v === "" ? undefined : Number(v));
        }}
        className="w-24 font-mono text-[12px] tabular-nums bg-background border border-border rounded-[4px] px-2 py-1.5 text-foreground focus:outline-none focus:border-accent shrink-0"
      />
    </div>
  );
}

function FloatControl({
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
    <div className="flex flex-nowrap items-center gap-2 overflow-x-auto md:justify-end">
      <button
        type="button"
        onClick={() => onChange(undefined)}
        className={chipClass(value === undefined)}
      >
        default
      </button>
      <input
        type="number"
        step="0.05"
        min={min}
        max={max}
        value={value ?? ""}
        placeholder={`${min ?? "0.0"}–${max ?? "1.0"}`}
        aria-label="Override value"
        onChange={(e) => {
          const v = e.target.value;
          onChange(v === "" ? undefined : Number(v));
        }}
        className="w-24 font-mono text-[12px] tabular-nums bg-background border border-border rounded-[4px] px-2 py-1.5 text-foreground focus:outline-none focus:border-accent shrink-0"
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
  return (
    <div className="flex flex-nowrap items-center gap-1 overflow-x-auto md:justify-end -mx-1 px-1">
      <button
        type="button"
        onClick={() => onChange(undefined)}
        className={chipClass(value === undefined)}
      >
        default
      </button>
      {choices.map((c) => {
        const eff = effectivePerChoice?.[c];
        const unavailable = eff === "unavailable";
        return (
          <button
            key={c}
            type="button"
            disabled={unavailable}
            onClick={() => onChange(c)}
            title={
              eff && eff !== c
                ? `→ effective: ${eff}`
                : unavailable
                  ? "Not available on this host"
                  : undefined
            }
            className={chipClass(value === c, unavailable)}
          >
            {c}
            {eff && eff !== c && eff !== "unavailable" && (
              <span className="ml-1 opacity-70 normal-case tracking-normal">
                → {eff}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
