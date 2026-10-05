import { useRef, useState, type DragEvent } from "react";

import type { DatasetInfo } from "../api/types";
import { Spinner } from "./StateViews";

interface Props {
  samples: DatasetInfo[];
  uploaded: DatasetInfo | null;
  selectedId: string | null;
  onSelect: (dataset: DatasetInfo) => void;
  onUpload: (file: File) => void;
  uploading: boolean;
  uploadError: string | null;
  maxUploadMb: number;
  loadingSamples: boolean;
}

export function DatasetPicker({ samples, uploaded, selectedId, onSelect, onUpload, uploading, uploadError, maxUploadMb, loadingSamples }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  function pick(file: File | undefined) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) return setLocalError("Only .csv files are supported.");
    if (file.size > maxUploadMb * 1024 * 1024) return setLocalError(`The file is larger than ${maxUploadMb} MB.`);
    setLocalError(null);
    onUpload(file);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    pick(e.dataTransfer.files?.[0]);
  }

  const options = uploaded ? [...samples, uploaded] : samples;
  const error = localError ?? uploadError;

  return (
    <section aria-labelledby="dataset-heading" className="flex flex-col gap-3">
      <h2 id="dataset-heading" className="text-xs font-semibold tracking-wide text-ink-muted uppercase">Dataset</h2>
      {loadingSamples ? (
        <div className="flex items-center gap-2 text-sm text-ink-secondary"><Spinner /> Loading datasets…</div>
      ) : (
        <div role="radiogroup" aria-label="Dataset" className="flex flex-col gap-2">
          {options.map((d) => {
            const active = d.dataset_id === selectedId;
            return (
              <button
                key={d.dataset_id}
                type="button"
                role="radio"
                aria-checked={active}
                onClick={() => onSelect(d)}
                className={`flex items-center justify-between gap-3 rounded-[var(--radius-ctl)] border px-3 py-2 text-left text-sm transition-colors ${
                  active ? "border-accent bg-accent-soft/60 text-ink" : "border-border bg-surface-raised hover:border-border-strong"
                }`}
              >
                <span className="flex min-w-0 items-center gap-2">
                  <span aria-hidden className={`size-3 shrink-0 rounded-full border-2 ${active ? "border-accent bg-accent" : "border-border-strong"}`} />
                  <span className="truncate font-medium">{d.name}</span>
                  {d.dataset_id === uploaded?.dataset_id && <span className="text-xs text-ink-muted">uploaded</span>}
                </span>
                <span className="tabular shrink-0 text-xs text-ink-muted">{d.rows.toLocaleString()} rows</span>
              </button>
            );
          })}
        </div>
      )}

      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex flex-col items-center gap-1 rounded-[var(--radius-ctl)] border border-dashed px-3 py-4 text-center text-sm transition-colors ${
          dragging ? "border-accent bg-accent-soft/40" : "border-border-strong"
        }`}
      >
        {uploading ? (
          <span className="flex items-center gap-2 text-ink-secondary"><Spinner label="Uploading" /> Uploading and profiling…</span>
        ) : (
          <>
            <button type="button" onClick={() => input.current?.click()} className="font-medium text-accent underline-offset-2 hover:underline">
              Upload a CSV
            </button>
            <span className="text-xs text-ink-muted">or drop it here · up to {maxUploadMb} MB · kept 30 min</span>
          </>
        )}
        <input
          ref={input}
          type="file"
          accept=".csv,text/csv"
          className="sr-only"
          aria-label="CSV file"
          onChange={(e) => { pick(e.target.files?.[0]); e.target.value = ""; }}
        />
      </div>
      {error && <p role="alert" className="text-sm text-danger">{error}</p>}
    </section>
  );
}
