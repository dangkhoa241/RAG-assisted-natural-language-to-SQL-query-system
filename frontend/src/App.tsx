import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError, withWakeRetry } from "./api/client";
import type { AppConfig, DatasetInfo, Mode, QueryResult } from "./api/types";
import { DatasetPicker } from "./components/DatasetPicker";
import { HowItWorks } from "./components/HowItWorks";
import { QuestionBox } from "./components/QuestionBox";
import { ResultChart } from "./components/ResultChart";
import { ResultTable } from "./components/ResultTable";
import { SchemaPreview } from "./components/SchemaPreview";
import { SqlBlock } from "./components/SqlBlock";
import { EmptyState, ErrorAlert, ResultsSkeleton, WakingNotice } from "./components/StateViews";
import { useTheme } from "./hooks/useTheme";

const FALLBACK_CONFIG: AppConfig = {
  default_mode: "auto", glossary_default: true, llm_provider: null, llm_model: null, llm_fallback_models: [], llm_budget_remaining: 0, max_upload_mb: 10, max_rows_returned: 500,
};

interface Failure {
  message: string;
  retryAfter: number | null;
}

function toFailure(e: unknown): Failure {
  if (e instanceof ApiError) return { message: e.message, retryAfter: e.retryAfter };
  return { message: "Something unexpected went wrong.", retryAfter: null };
}

export default function App() {
  const { theme, toggle } = useTheme();
  const [config, setConfig] = useState<AppConfig>(FALLBACK_CONFIG);
  const [samples, setSamples] = useState<DatasetInfo[]>([]);
  const [loadingSamples, setLoadingSamples] = useState(true);
  const [startupError, setStartupError] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState<DatasetInfo | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dataset, setDataset] = useState<DatasetInfo | null>(null);

  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("auto");
  const [useGlossary, setUseGlossary] = useState(true);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [resultKey, setResultKey] = useState(0);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [waking, setWaking] = useState(false);   // the free-tier backend is asleep and starting up
  const lastAsked = useRef<string>("");

  const loadStartup = useCallback(() => {
    setLoadingSamples(true);
    setStartupError(null);
    withWakeRetry(() => Promise.all([api.config(), api.samples()]), setWaking)
      .then(([cfg, list]) => {
        setConfig(cfg);
        setMode(cfg.default_mode);
        setUseGlossary(cfg.glossary_default);
        setSamples(list);
      })
      .catch((e) => setStartupError(toFailure(e).message))
      .finally(() => setLoadingSamples(false));
  }, []);

  useEffect(loadStartup, [loadStartup]);

  function selectDataset(d: DatasetInfo) {
    if (d.dataset_id === dataset?.dataset_id) return;
    setDataset(d);
    setResult(null);
    setFailure(null);
  }

  async function upload(file: File) {
    setUploading(true);
    setUploadError(null);
    try {
      const info = await withWakeRetry(() => api.upload(file), setWaking);
      setUploaded(info);
      selectDataset(info);
    } catch (e) {
      setUploadError(toFailure(e).message);
    } finally {
      setUploading(false);
    }
  }

  async function ask(q?: string) {
    const text = (q ?? question).trim();
    if (!dataset || !text) return;
    if (q !== undefined) setQuestion(q);
    lastAsked.current = text;
    setBusy(true);
    setFailure(null);
    try {
      const body = {
        dataset_id: dataset.dataset_id, question: text, mode, use_glossary: useGlossary && dataset.has_glossary,
      };
      const res = await withWakeRetry(() => api.query(body), setWaking);
      setResult(res);
      setResultKey((k) => k + 1);
    } catch (e) {
      setFailure(toFailure(e));
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen bg-page text-ink">
      <header className="z-10 border-b border-border bg-page/90 backdrop-blur lg:sticky lg:top-0">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-2.5">
            <span aria-hidden className="grid size-8 place-items-center rounded-lg bg-accent text-sm font-bold text-accent-ink">N</span>
            <div>
              <h1 className="text-base leading-tight font-semibold">NL→SQL Data Assistant</h1>
              <p className="hidden text-xs text-ink-muted sm:block">Ask questions about a dataset in plain English</p>
            </div>
          </div>
          <button
            type="button"
            onClick={toggle}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            className="grid size-9 place-items-center rounded-full border border-border-strong text-lg hover:bg-surface-sunken"
          >
            <span aria-hidden>{theme === "dark" ? "☀" : "☾"}</span>
          </button>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-4 px-4 py-4 sm:px-6 sm:py-6 lg:grid-cols-[20rem_minmax(0,1fr)] lg:grid-rows-[auto_1fr] lg:items-start lg:gap-6">
        <aside className="panel flex flex-col gap-5 lg:col-start-1 lg:row-start-1">
          {waking && loadingSamples && <WakingNotice />}
          {startupError ? (
            <ErrorAlert message={startupError} onRetry={loadStartup} />
          ) : (
            <DatasetPicker
              samples={samples}
              uploaded={uploaded}
              selectedId={dataset?.dataset_id ?? null}
              onSelect={selectDataset}
              onUpload={upload}
              uploading={uploading}
              uploadError={uploadError}
              maxUploadMb={config.max_upload_mb}
              loadingSamples={loadingSamples}
            />
          )}
          {dataset && <SchemaPreview key={dataset.dataset_id} dataset={dataset} />}
        </aside>

        <div className="flex min-w-0 flex-col gap-4 lg:col-start-2 lg:row-span-2 lg:row-start-1">
          <QuestionBox
            question={question}
            onQuestionChange={setQuestion}
            mode={mode}
            onModeChange={setMode}
            useGlossary={useGlossary}
            onGlossaryChange={setUseGlossary}
            glossaryAvailable={!!dataset?.has_glossary}
            examples={dataset?.example_questions ?? []}
            onAsk={ask}
            busy={busy}
            disabled={!dataset}
          />

          {failure && <ErrorAlert message={failure.message} retryAfter={failure.retryAfter} onRetry={() => ask(lastAsked.current)} />}

          {busy ? (
            <>
              {waking && <WakingNotice />}
              <ResultsSkeleton />
            </>
          ) : !dataset ? (
            <EmptyState title="Pick a dataset to start" icon="▤">
              Choose one of the sample datasets, or upload your own CSV.
            </EmptyState>
          ) : result ? (
            <div className="flex flex-col gap-4" key={resultKey}>
              <ResultChart result={result} theme={theme} />
              <ResultTable result={result} />
              <SqlBlock sql={result.sql} />
            </div>
          ) : (
            !failure && (
              <EmptyState title="Ask your first question" icon="?">
                Type a question above or click one of the examples.
              </EmptyState>
            )
          )}
        </div>

        <div className="lg:col-start-1 lg:row-start-2">
          <HowItWorks result={result} />
        </div>
      </main>
    </div>
  );
}
