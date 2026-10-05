import { useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell as PieCell, Line, LineChart, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

import type { ChartType, QueryResult } from "../api/types";
import { useCssVars } from "../hooks/useTheme";
import { availableChartTypes, CHART_LABELS, planChart, SERIES_COUNT, toRecords } from "../lib/chart";
import { formatCompact, formatNumber, humanize } from "../lib/format";

const TOKENS = [
  "chart-surface", "chart-grid", "chart-axis", "chart-label", "ink", "ink-secondary", "surface-raised", "border",
  ...Array.from({ length: SERIES_COUNT }, (_, i) => `series-${i + 1}`),
] as const;
type Tokens = Record<(typeof TOKENS)[number], string>;

const ROW_HEIGHT = 32;
const INITIAL = { width: 640, height: 320 }; // before the first measurement (and in jsdom)

function tooltipStyle(t: Tokens) {
  return {
    contentStyle: { background: t["surface-raised"], border: `1px solid ${t.border}`, borderRadius: 8, color: t.ink, fontSize: 13 },
    labelStyle: { color: t["ink-secondary"], marginBottom: 2 },
    itemStyle: { color: t.ink, padding: 0 },
  };
}

export function StatCard({ label, value }: { label: string; value: number | string | null }) {
  return (
    <div className="flex flex-col items-start gap-1 py-6" data-testid="stat-card">
      <span className="text-sm text-ink-secondary">{label}</span>
      <span className="text-5xl font-semibold tracking-tight text-ink">
        {typeof value === "number" ? formatNumber(value) : (value ?? "—")}
      </span>
    </div>
  );
}

export function ResultChart({ result, theme }: { result: QueryResult; theme: string }) {
  const tokens = useCssVars(TOKENS, theme) as Tokens;
  const available = useMemo(() => availableChartTypes(result), [result]);
  const [picked, setPicked] = useState<ChartType | null>(null);
  const plan = planChart(result, picked ?? undefined);
  const series = Array.from({ length: SERIES_COUNT }, (_, i) => tokens[`series-${i + 1}` as keyof Tokens]);

  const title = plan.y ? humanize(plan.y) + (plan.x ? ` by ${humanize(plan.x)}` : "") : "Result";

  return (
    <section aria-label="Chart" className="panel flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-base font-semibold">{plan.type === "table" ? "Chart" : plan.type === "stat" ? "Answer" : title}</h2>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-ink-secondary">Chart type</span>
          <select
            aria-label="Chart type"
            value={plan.type}
            onChange={(e) => setPicked(e.target.value as ChartType)}
            className="rounded-[var(--radius-ctl)] border border-border-strong bg-surface-raised px-2 py-1"
          >
            {available.map((t) => (
              <option key={t} value={t}>{CHART_LABELS[t]}</option>
            ))}
          </select>
        </label>
      </div>
      <ChartBody result={result} plan={plan} tokens={tokens} series={series} />
      {result.chart.reason && picked === null && plan.type === result.chart.type && (
        <p className="text-xs text-ink-muted">Suggested: {CHART_LABELS[plan.type].toLowerCase()} ({result.chart.reason}).</p>
      )}
    </section>
  );
}

function ChartBody({ result, plan, tokens, series }: { result: QueryResult; plan: ReturnType<typeof planChart>; tokens: Tokens; series: string[] }) {
  if (plan.type === "table") {
    return <p className="text-sm text-ink-secondary">This result reads best as a table (below). Pick a chart type to plot it anyway.</p>;
  }
  if (plan.type === "stat") {
    const yi = result.columns.findIndex((c) => c.name === plan.y);
    return <StatCard label={humanize(plan.y ?? "Value")} value={yi >= 0 ? (result.rows[0]?.[yi] as number) : null} />;
  }
  const data = toRecords(result, plan.x!, plan.y!);
  const tip = tooltipStyle(tokens);
  const axisTick = { fill: tokens["chart-label"], fontSize: 12 };
  const yName = humanize(plan.y!);

  if (plan.type === "bar") {
    // Horizontal bars: category labels stay readable at any width, including phones.
    const height = Math.max(160, data.length * ROW_HEIGHT + 40);
    const longest = Math.max(...data.map((d) => d.x.length));
    return (
      <div data-testid="chart-bar" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%" initialDimension={{ ...INITIAL, height }}>
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 4 }}>
            <CartesianGrid horizontal={false} stroke={tokens["chart-grid"]} />
            <XAxis type="number" tickFormatter={formatCompact} tick={axisTick} stroke={tokens["chart-axis"]} tickLine={false} />
            <YAxis type="category" dataKey="x" width={Math.min(160, Math.max(56, longest * 7))} tick={axisTick} stroke={tokens["chart-axis"]} tickLine={false} interval={0} />
            <Tooltip {...tip} cursor={{ fill: tokens["chart-grid"], opacity: 0.5 }} formatter={(v) => [formatNumber(Number(v)), yName]} />
            <Bar dataKey="y" name={yName} fill={series[0]} maxBarSize={24} radius={[0, 4, 4, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }

  if (plan.type === "line") {
    return (
      <div data-testid="chart-line" className="h-72">
        <ResponsiveContainer width="100%" height="100%" initialDimension={INITIAL}>
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
            <CartesianGrid vertical={false} stroke={tokens["chart-grid"]} />
            <XAxis dataKey="x" tick={axisTick} stroke={tokens["chart-axis"]} tickLine={false} minTickGap={16} />
            <YAxis tickFormatter={formatCompact} tick={axisTick} stroke={tokens["chart-axis"]} tickLine={false} width={56} />
            <Tooltip {...tip} cursor={{ stroke: tokens["chart-axis"] }} formatter={(v) => [formatNumber(Number(v)), yName]} />
            <Line
              type="linear" dataKey="y" name={yName} stroke={series[0]} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"
              dot={data.length <= 40 ? { r: 4, fill: series[0], stroke: tokens["chart-surface"], strokeWidth: 2 } : false}
              activeDot={{ r: 5, fill: series[0], stroke: tokens["chart-surface"], strokeWidth: 2 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  }

  // pie (≤ 8 slices, one fixed-order series color each) + a labelled legend so identity isn't color-alone
  const total = data.reduce((s, d) => s + d.y, 0) || 1;
  return (
    <div data-testid="chart-pie" className="flex flex-col items-center gap-4 sm:flex-row">
      <div className="h-60 w-full max-w-60">
        <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 240, height: 240 }}>
          <PieChart>
            <Tooltip {...tip} formatter={(v, n) => [formatNumber(Number(v)), String(n)]} />
            <Pie data={data} dataKey="y" nameKey="x" innerRadius="55%" outerRadius="95%" stroke={tokens["chart-surface"]} strokeWidth={2} isAnimationActive={false}>
              {data.map((d, i) => (
                <PieCell key={d.x} fill={series[i % series.length]} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      </div>
      <ul aria-label="Legend" className="flex w-full flex-col gap-1.5 text-sm">
        {data.map((d, i) => (
          <li key={d.x} className="flex items-center gap-2">
            <span aria-hidden className="size-3 shrink-0 rounded-sm" style={{ background: series[i % series.length] }} />
            <span className="min-w-0 flex-1 truncate text-ink">{d.x}</span>
            <span className="tabular text-ink-secondary">{formatNumber(d.y)}</span>
            <span className="tabular w-12 text-right text-ink-muted">{Math.round((d.y / total) * 100)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
