import { useEffect, useState } from "react";

export function SqlBlock({ sql, title = "SQL" }: { sql: string; title?: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");

  useEffect(() => {
    if (state === "idle") return;
    const id = setTimeout(() => setState("idle"), 1800);
    return () => clearTimeout(id);
  }, [state]);

  async function copy() {
    try {
      await navigator.clipboard.writeText(sql);
      setState("copied");
    } catch {
      setState("failed");
    }
  }

  return (
    <section aria-label={title} className="panel flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-base font-semibold">{title}</h2>
        <button type="button" onClick={copy} className="rounded-[var(--radius-ctl)] border border-border-strong px-3 py-1 text-sm hover:bg-surface-sunken">
          {state === "copied" ? "Copied ✓" : state === "failed" ? "Copy failed" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto rounded-[var(--radius-ctl)] bg-surface-sunken p-3 font-mono text-sm leading-relaxed whitespace-pre-wrap text-ink">
        <code>{sql}</code>
      </pre>
      <span aria-live="polite" className="sr-only">{state === "copied" ? "SQL copied to clipboard" : ""}</span>
    </section>
  );
}
