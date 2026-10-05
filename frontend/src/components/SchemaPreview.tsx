import type { DatasetInfo } from "../api/types";
import { formatCell } from "../lib/format";

const KIND_LABEL = { numeric: "number", categorical: "category", date: "date", text: "text" } as const;

// Open beside the results on wide screens; collapsed on phones, where it would push the question box down.
const startOpen = () => typeof window === "undefined" || !window.matchMedia || window.matchMedia("(min-width: 1024px)").matches;

export function SchemaPreview({ dataset }: { dataset: DatasetInfo }) {
  return (
    <details open={startOpen()} className="group">
      <summary className="flex cursor-pointer list-none items-center justify-between text-xs font-semibold tracking-wide text-ink-muted uppercase">
        <span>Schema · {dataset.schema.length} columns</span>
        <span aria-hidden className="transition-transform group-open:rotate-90">›</span>
      </summary>
      {dataset.description && <p className="mt-2 text-sm text-ink-secondary">{dataset.description}</p>}
      <ul className="mt-2 flex max-h-72 flex-col divide-y divide-border overflow-y-auto text-sm">
        {dataset.schema.map((col) => (
          <li key={col.name} className="flex flex-col gap-0.5 py-1.5">
            <div className="flex items-baseline justify-between gap-2">
              <span className="truncate font-medium" title={col.name}>{col.name}</span>
              <span className="shrink-0 rounded bg-surface-sunken px-1.5 py-0.5 text-[11px] text-ink-secondary">{KIND_LABEL[col.type]}</span>
            </div>
            <span className="truncate text-xs text-ink-muted" title={col.sample_values.map(formatCell).join(", ")}>
              {col.sample_values.map(formatCell).join(", ")}
            </span>
          </li>
        ))}
      </ul>
    </details>
  );
}
