import type { Cell } from "../api/types";

const full = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

export function formatNumber(v: number): string {
  return full.format(v);
}

/** Axis ticks: 1,284 → 1.3K, 4,200,000 → 4.2M. */
export function formatCompact(v: number): string {
  return Math.abs(v) < 10_000 ? full.format(v) : compact.format(v);
}

export function formatCell(v: Cell): string {
  if (v === null) return "—";
  if (typeof v === "number") return formatNumber(v);
  return String(v);
}

const AGGREGATES: Record<string, string> = { avg: "Average", sum: "Total", min: "Minimum", max: "Maximum", count: "Count of" };

/** Column name → label: "total_revenue" → "Total revenue", "AVG(`Billing Amount`)" → "Average Billing Amount". */
export function humanize(name: string): string {
  const clean = (s: string) => s.replace(/[_`"]+/g, " ").replace(/\s+/g, " ").trim();
  const agg = /^\s*(avg|sum|min|max|count)\s*\((.*)\)\s*$/i.exec(name);
  if (agg) {
    const inner = clean(agg[2]);
    if (agg[1].toLowerCase() === "count") return inner === "*" || inner === "" ? "Count" : `Count of ${inner}`;
    return `${AGGREGATES[agg[1].toLowerCase()]} ${inner}`;
  }
  const words = clean(name);
  return words.charAt(0).toUpperCase() + words.slice(1);
}
