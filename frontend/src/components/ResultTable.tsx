import { useMemo, useState } from "react";

import type { Cell, QueryResult } from "../api/types";
import { formatCell } from "../lib/format";

const PAGE_SIZES = [10, 25, 50];
type Sort = { col: number; dir: "asc" | "desc" } | null;

function compare(a: Cell, b: Cell): number {
  if (a === null) return b === null ? 0 : 1; // empty cells sort last
  if (b === null) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

export function ResultTable({ result }: { result: QueryResult }) {
  const [sort, setSort] = useState<Sort>(null);
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(PAGE_SIZES[0]);

  const rows = useMemo(() => {
    if (!sort) return result.rows;
    const sorted = [...result.rows].sort((r1, r2) => compare(r1[sort.col], r2[sort.col]));
    return sort.dir === "asc" ? sorted : sorted.reverse();
  }, [result.rows, sort]);

  const pages = Math.max(1, Math.ceil(rows.length / pageSize));
  const current = Math.min(page, pages - 1);
  const shown = rows.slice(current * pageSize, (current + 1) * pageSize);

  function toggleSort(col: number) {
    setPage(0);
    setSort((s) => (s?.col !== col ? { col, dir: "asc" } : s.dir === "asc" ? { col, dir: "desc" } : null));
  }

  return (
    <section aria-label="Result table" className="panel flex flex-col gap-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-base font-semibold">Table</h2>
        <span className="text-sm text-ink-secondary">
          {result.row_count.toLocaleString()} row{result.row_count === 1 ? "" : "s"}
          {result.truncated && ` · showing the first ${result.rows.length.toLocaleString()}`}
        </span>
      </div>

      {result.rows.length === 0 ? (
        <p className="py-6 text-center text-sm text-ink-secondary">The query ran but returned no rows. Try loosening the filters in your question.</p>
      ) : (
        <>
          <div className="-mx-[var(--panel-padding)] overflow-x-auto">
            <table className="w-full min-w-max border-collapse text-sm">
              <thead>
                <tr className="border-b border-border-strong">
                  {result.columns.map((c, i) => {
                    const dir = sort?.col === i ? sort.dir : null;
                    return (
                      <th
                        key={c.name}
                        scope="col"
                        aria-sort={dir === "asc" ? "ascending" : dir === "desc" ? "descending" : "none"}
                        className={`px-[var(--panel-padding)] py-2 font-medium text-ink-secondary first:pl-[var(--panel-padding)] ${c.type === "number" ? "text-right" : "text-left"}`}
                      >
                        <button type="button" onClick={() => toggleSort(i)} className="inline-flex items-center gap-1 hover:text-ink">
                          {c.name}
                          <span aria-hidden className={dir ? "text-ink" : "text-ink-muted/60"}>{dir === "asc" ? "▲" : dir === "desc" ? "▼" : "↕"}</span>
                        </button>
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {shown.map((row, ri) => (
                  <tr key={current * pageSize + ri} className="border-b border-border last:border-0 hover:bg-surface-sunken">
                    {row.map((v, ci) => (
                      <td
                        key={ci}
                        className={`max-w-xs truncate px-[var(--panel-padding)] py-1.5 ${result.columns[ci]?.type === "number" ? "tabular text-right" : ""}`}
                        title={typeof v === "string" && v.length > 40 ? v : undefined}
                      >
                        {formatCell(v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <nav aria-label="Pagination" className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <label className="flex items-center gap-2 text-ink-secondary">
              Rows per page
              <select value={pageSize} onChange={(e) => { setPageSize(Number(e.target.value)); setPage(0); }} className="rounded-[var(--radius-ctl)] border border-border-strong bg-surface-raised px-2 py-1 text-ink">
                {PAGE_SIZES.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <div className="flex items-center gap-2">
              <button type="button" aria-label="Previous page" disabled={current === 0} onClick={() => setPage(current - 1)} className="rounded-[var(--radius-ctl)] border border-border-strong px-2.5 py-1 disabled:opacity-40">‹</button>
              <span className="tabular text-ink-secondary">Page {current + 1} of {pages}</span>
              <button type="button" aria-label="Next page" disabled={current >= pages - 1} onClick={() => setPage(current + 1)} className="rounded-[var(--radius-ctl)] border border-border-strong px-2.5 py-1 disabled:opacity-40">›</button>
            </div>
          </nav>
        </>
      )}
    </section>
  );
}
