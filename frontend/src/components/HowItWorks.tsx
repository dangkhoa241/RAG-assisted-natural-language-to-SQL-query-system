import type { QueryResult } from "../api/types";

const STRATEGY = { zero_shot: "zero-shot", example_rag: "with retrieved examples", glossary_rag: "with glossary definitions" } as const;
const STEPS = ["intent", "retrieval", "generation", "execution"] as const;
const PROVIDER = { groq: "Groq", cerebras: "Cerebras" } as const;
const FALLBACK_TITLE: Record<string, string> = {
  daily_budget: "Daily LLM budget used up",
  rate_limited: "LLM rate limit reached",
  llm_unavailable: "No LLM configured",
  provider_quota: "LLM provider quota exhausted",
  provider_rate_limited: "LLM provider rate limit reached",
  llm_error: "LLM call failed",
  unsafe_sql: "LLM SQL blocked by the safety check",
  execution_error: "LLM SQL failed to run",
};

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <dt className="text-xs text-ink-muted">{label}</dt>
      <dd className="text-sm">{children}</dd>
    </div>
  );
}

export function HowItWorks({ result }: { result: QueryResult | null }) {
  return (
    <section aria-labelledby="how-heading" className="panel flex flex-col gap-4">
      <h2 id="how-heading" className="text-xs font-semibold tracking-wide text-ink-muted uppercase">How it works</h2>
      {!result ? (
        <p className="text-sm text-ink-secondary">
          Ask a question to see each step: the predicted intent, which generator wrote the SQL, which business-glossary terms matched, and how long each step took.
        </p>
      ) : (
        <Details result={result} />
      )}
    </section>
  );
}

function Details({ result }: { result: QueryResult }) {
  const { intent, generator: gen, context, latency_ms: lat } = result;
  const pct = intent.confidence === null ? null : Math.round(intent.confidence * 100);
  const maxStep = Math.max(1, ...STEPS.map((s) => lat[s]));
  let step = 2;

  return (
    <dl className="flex flex-col gap-4">
      <Row label="1 · Intent">
        <div className="flex items-center gap-2">
          <span className="font-medium capitalize">{intent.label}</span>
          <span className="text-xs text-ink-muted">{intent.source === "bert" ? "BERT classifier" : "keyword rules"}</span>
        </div>
        {pct !== null && (
          <div className="mt-1 flex items-center gap-2">
            <div className="h-1.5 flex-1 rounded-full bg-accent-soft" role="meter" aria-label="Intent confidence" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
              <div className="h-full rounded-full bg-accent" style={{ width: `${pct}%` }} />
            </div>
            <span className="tabular text-xs text-ink-secondary">{pct}%</span>
          </div>
        )}
      </Row>

      <Row label="2 · Generator">
        <span className="font-medium">
          {gen.used === "llm" ? `LLM ${STRATEGY[gen.llm_strategy ?? "zero_shot"]}` : "Rule-based"}
        </span>
        {gen.model && (
          <span className="block font-mono text-xs text-ink-muted">
            {gen.model}{gen.provider && ` · ${PROVIDER[gen.provider]}`}
          </span>
        )}
        {gen.model_note && (
          <p role="note" className="mt-1 text-xs text-warning-ink">↳ Answered by {gen.model_note}</p>
        )}
        {gen.fallback_reason && (
          <div role="note" className="mt-2 rounded-[var(--radius-ctl)] bg-warning-soft px-3 py-2 text-xs text-ink">
            <p className="font-semibold text-warning-ink">↳ Fell back: {FALLBACK_TITLE[gen.fallback_reason] ?? gen.fallback_reason}</p>
            {gen.fallback_detail && <p className="mt-0.5 text-ink-secondary">{gen.fallback_detail}</p>}
            {gen.rejected_sql && (
              <details className="mt-1">
                <summary className="cursor-pointer text-ink-secondary">Rejected SQL</summary>
                <pre className="mt-1 overflow-x-auto font-mono whitespace-pre-wrap">{gen.rejected_sql}</pre>
              </details>
            )}
          </div>
        )}
      </Row>

      {context.glossary_checked && (
        <Row label={`${++step} · Business glossary`}>
          {context.glossary.length > 0 ? (
            <>
              <p className="text-xs text-ink-secondary">
                {context.glossary.length === 1 ? "1 term" : `${context.glossary.length} terms`} matched in the question. Only these definitions were sent:
              </p>
              <ul className="mt-1.5 flex flex-col gap-2" aria-label="Matched glossary terms">
                {context.glossary.map((d) => (
                  <li key={d.id} className="rounded-[var(--radius-ctl)] border border-border px-2.5 py-2">
                    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                      <span className="font-medium">{d.term}</span>
                      <span className="rounded bg-accent-soft px-1.5 py-0.5 font-mono text-[11px] text-ink" title="Matched in the question">
                        “{d.matched}”
                      </span>
                    </div>
                    <p className="mt-1 text-xs leading-relaxed text-ink-secondary">{d.definition}</p>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p className="text-xs text-ink-secondary">No glossary term appears in the question, so no definitions were sent.</p>
          )}
        </Row>
      )}

      {context.examples.length > 0 && (
        <Row label={`${++step} · Retrieved examples`}>
          <ul className="flex flex-col gap-1.5">
            {context.examples.map((e) => (
              <li key={e.question} className="text-xs">
                <span className="text-ink">{e.question}</span> <span className="tabular text-ink-muted">({e.score.toFixed(2)})</span>
              </li>
            ))}
          </ul>
        </Row>
      )}

      <Row label="Latency">
        <ul className="flex flex-col gap-1">
          {STEPS.map((s) => (
            <li key={s} className="grid grid-cols-[5.5rem_1fr_3.5rem] items-center gap-2 text-xs">
              <span className="text-ink-secondary capitalize">{s}</span>
              <span className="h-1.5 rounded-full bg-surface-sunken">
                <span className="block h-full rounded-full bg-accent" style={{ width: `${(lat[s] / maxStep) * 100}%` }} />
              </span>
              <span className="tabular text-right text-ink-secondary">{lat[s].toLocaleString()} ms</span>
            </li>
          ))}
          <li className="grid grid-cols-[5.5rem_1fr_3.5rem] gap-2 border-t border-border pt-1 text-xs font-medium">
            <span>Total</span><span /><span className="tabular text-right">{lat.total.toLocaleString()} ms</span>
          </li>
        </ul>
      </Row>

      {result.notes.length > 0 && (
        <ul className="flex flex-col gap-1 text-xs text-ink-secondary">
          {result.notes.map((n) => <li key={n}>ⓘ {n}</li>)}
        </ul>
      )}
    </dl>
  );
}
