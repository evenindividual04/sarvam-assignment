"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface LatencyBarProps {
  planning_ms?: number;
  search_ms?: number;
  fetch_ms?: number;
  select_ms?: number;
  probe_ms?: number;
  synthesize_ms?: number;
}

// All bars share teal — the chart is editorial restraint, not a color carousel.
// The fill varies in opacity by stage to give passive depth without competing colors.
const FILLS: Record<string, string> = {
  Planning: "rgba(13, 148, 136, 0.40)",
  Search: "rgba(13, 148, 136, 0.55)",
  Fetch: "rgba(13, 148, 136, 0.70)",
  Select: "rgba(13, 148, 136, 0.85)",
  Probe: "rgba(13, 148, 136, 0.65)",
  Synthesize: "rgba(13, 148, 136, 0.95)",
};

export function LatencyBar(props: LatencyBarProps) {
  const data = [
    { name: "Planning", ms: props.planning_ms ?? 0 },
    { name: "Search", ms: props.search_ms ?? 0 },
    { name: "Fetch", ms: props.fetch_ms ?? 0 },
    { name: "Select", ms: props.select_ms ?? 0 },
    { name: "Probe", ms: props.probe_ms ?? 0 },
    { name: "Synthesize", ms: props.synthesize_ms ?? 0 },
  ];
  return (
    <div className="w-full h-40">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 10, right: 4, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="name"
            stroke="#6b6b73"
            fontSize={10}
            tickLine={false}
            axisLine={false}
            tick={{ fontFamily: "var(--font-mono)" }}
          />
          <YAxis
            stroke="#6b6b73"
            fontSize={10}
            tickLine={false}
            axisLine={false}
            tick={{ fontFamily: "var(--font-mono)" }}
            tickFormatter={(v) => {
              const n = Number(v);
              return n >= 1000 ? `${(n / 1000).toFixed(1)}s` : `${n}ms`;
            }}
          />
          <Tooltip
            cursor={{ fill: "rgba(255, 255, 255, 0.04)" }}
            contentStyle={{
              background: "#111114",
              border: "1px solid rgba(255, 255, 255, 0.12)",
              borderRadius: 6,
              fontSize: 11,
              fontFamily: "var(--font-mono)",
              padding: "6px 8px",
            }}
            labelStyle={{
              color: "#9b9ba3",
              fontSize: 10,
              textTransform: "uppercase",
              letterSpacing: "0.12em",
            }}
            formatter={(value) => {
              const v = Number(value ?? 0);
              return v >= 1000 ? `${(v / 1000).toFixed(2)}s` : `${v}ms`;
            }}
          />
          <Bar dataKey="ms" radius={[2, 2, 0, 0]}>
            {data.map((d) => (
              <Cell key={d.name} fill={FILLS[d.name]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
