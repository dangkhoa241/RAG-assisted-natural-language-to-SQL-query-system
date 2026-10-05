import type { Cell, ChartType, QueryResult } from "../api/types";

export const MAX_BAR_ROWS = 100;
export const MAX_PIE_SLICES = 8; // one per categorical series color; never cycle colors
export const SERIES_COUNT = 8;

export interface Axes {
  x: string | null;
  y: string | null;
}

export interface ChartPlan extends Axes {
  type: ChartType;
}

type Shape = Pick<QueryResult, "columns" | "rows" | "chart">;

function axesFor(result: Shape): Axes {
  const numeric = result.columns.filter((c) => c.type === "number").map((c) => c.name);
  const text = result.columns.filter((c) => c.type === "text").map((c) => c.name);
  if (text.length > 0 && numeric.length > 0) return { x: text[0], y: numeric[0] };
  if (numeric.length >= 2) return { x: numeric[0], y: numeric[1] };
  return { x: null, y: numeric[0] ?? null };
}

function columnValues(result: Shape, name: string): Cell[] {
  const i = result.columns.findIndex((c) => c.name === name);
  return i < 0 ? [] : result.rows.map((r) => r[i]);
}

/** The chart types that make sense for this result, in dropdown order. "table" is always allowed. */
export function availableChartTypes(result: Shape): ChartType[] {
  const types: ChartType[] = [];
  const rows = result.rows.length;
  const { x, y } = axesFor(result);
  if (rows === 1 && y) types.push("stat");
  if (x && y && rows >= 1) {
    if (rows <= MAX_BAR_ROWS) types.push("bar");
    if (rows >= 2) types.push("line");
    const ys = columnValues(result, y);
    if (rows >= 2 && rows <= MAX_PIE_SLICES && ys.every((v) => typeof v === "number" && v >= 0)) {
      types.push("pie");
    }
  }
  types.push("table");
  return types;
}

/** The chart to draw for `type` (default: the server's suggestion), on the server's axes when they exist. */
export function planChart(result: Shape, type?: ChartType): ChartPlan {
  const available = availableChartTypes(result);
  const wanted = type ?? result.chart.type;
  const chosen = available.includes(wanted) ? wanted : available[0];
  if (chosen === "table") return { type: chosen, x: null, y: null };
  const names = new Set(result.columns.map((c) => c.name));
  const fallback = axesFor(result);
  const { x: sx, y: sy } = result.chart;
  const y = sy && names.has(sy) ? sy : fallback.y;
  if (chosen === "stat") return { type: chosen, x: null, y };
  return { type: chosen, x: sx && names.has(sx) ? sx : fallback.x, y };
}

/** Rows as objects keyed by column name, for Recharts. */
export function toRecords(result: Shape, x: string, y: string): { x: string; y: number }[] {
  const xi = result.columns.findIndex((c) => c.name === x);
  const yi = result.columns.findIndex((c) => c.name === y);
  return result.rows
    .filter((r) => typeof r[yi] === "number")
    .map((r) => ({ x: r[xi] === null ? "(empty)" : String(r[xi]), y: r[yi] as number }));
}

export const CHART_LABELS: Record<ChartType, string> = {
  bar: "Bar",
  line: "Line",
  pie: "Pie",
  stat: "Number",
  table: "Table only",
};
