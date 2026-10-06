import type { FormEvent } from "react";

import type { ExampleQuestion, Mode } from "../api/types";
import { Spinner } from "./StateViews";

const MODES: { value: Mode; label: string; hint: string }[] = [
  { value: "auto", label: "Auto", hint: "LLM first, rule-based fallback" },
  { value: "llm", label: "LLM only", hint: "No fallback" },
  { value: "rule_based", label: "Rule-based", hint: "No LLM call" },
];

interface Props {
  question: string;
  onQuestionChange: (q: string) => void;
  mode: Mode;
  onModeChange: (m: Mode) => void;
  useGlossary: boolean;
  onGlossaryChange: (v: boolean) => void;
  glossaryAvailable: boolean;
  examples: ExampleQuestion[];
  onAsk: (question?: string) => void;
  busy: boolean;
  disabled: boolean;
}

export const MAX_QUESTION_CHARS = 500;

export function QuestionBox(p: Props) {
  function submit(e: FormEvent) {
    e.preventDefault();
    if (p.question.trim()) p.onAsk();
  }

  return (
    <section aria-label="Ask a question" className="panel flex flex-col gap-3">
      <form onSubmit={submit} className="flex flex-col gap-3">
        <label htmlFor="question" className="text-sm font-medium">Ask a question about the data</label>
        <div className="flex flex-col gap-2 sm:flex-row">
          <input
            id="question"
            value={p.question}
            maxLength={MAX_QUESTION_CHARS}
            onChange={(e) => p.onQuestionChange(e.target.value)}
            placeholder="e.g. average billing amount by insurance provider"
            disabled={p.disabled}
            autoComplete="off"
            className="min-w-0 flex-1 rounded-[var(--radius-ctl)] border border-border-strong bg-surface-raised px-3 py-2 text-base placeholder:text-ink-muted disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={p.disabled || p.busy || !p.question.trim()}
            className="flex items-center justify-center gap-2 rounded-[var(--radius-ctl)] bg-accent px-5 py-2 font-medium text-accent-ink hover:opacity-90 disabled:opacity-50"
          >
            {p.busy ? <><Spinner label="Asking" /> Asking…</> : "Ask"}
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
          <label className="flex items-center gap-2">
            <span className="text-ink-secondary">Mode</span>
            <select
              value={p.mode}
              onChange={(e) => p.onModeChange(e.target.value as Mode)}
              className="rounded-[var(--radius-ctl)] border border-border-strong bg-surface-raised px-2 py-1"
            >
              {MODES.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
            <span className="hidden text-xs text-ink-muted md:inline">{MODES.find((m) => m.value === p.mode)?.hint}</span>
          </label>
          <label className={`flex items-center gap-2 ${p.glossaryAvailable ? "" : "opacity-60"}`} title={p.glossaryAvailable ? "Sends a definition only when its term appears in the question" : "Only the sample datasets have a glossary"}>
            <input
              type="checkbox"
              checked={p.useGlossary && p.glossaryAvailable}
              disabled={!p.glossaryAvailable || p.mode === "rule_based"}
              onChange={(e) => p.onGlossaryChange(e.target.checked)}
              className="size-4 accent-[var(--accent)]"
            />
            <span>Use business glossary</span>
          </label>
        </div>
      </form>

      {p.examples.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="text-xs text-ink-muted">Try an example:</span>
          <ul className="flex flex-wrap gap-2">
            {p.examples.map((ex) => (
              <li key={ex.question}>
                <button
                  type="button"
                  disabled={p.disabled || p.busy}
                  onClick={() => p.onAsk(ex.question)}
                  className="rounded-full border border-border bg-surface-raised px-3 py-1 text-left text-sm text-ink-secondary hover:border-accent hover:text-ink disabled:opacity-50"
                >
                  {ex.question}
                  {ex.needs_glossary && <span className="ml-1.5 rounded bg-accent-soft px-1 text-[10px] font-semibold text-ink uppercase">glossary</span>}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
