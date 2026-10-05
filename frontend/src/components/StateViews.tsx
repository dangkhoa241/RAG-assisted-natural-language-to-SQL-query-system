import { useEffect, useState, type ReactNode } from "react";

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <span role="status" aria-label={label} className="inline-block size-4 animate-spin rounded-full border-2 border-current border-r-transparent motion-reduce:animate-none" />
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div aria-hidden className={`skeleton ${className}`} />;
}

export function ResultsSkeleton() {
  return (
    <div role="status" aria-label="Running query" className="flex flex-col gap-4">
      <div className="panel flex flex-col gap-3">
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-56 w-full" />
      </div>
      <div className="panel flex flex-col gap-2">
        {Array.from({ length: 5 }, (_, i) => (
          <Skeleton key={i} className="h-6 w-full" />
        ))}
      </div>
    </div>
  );
}

/** An error message, with a live countdown when the server asked us to retry later. */
export function ErrorAlert({ message, retryAfter, onRetry }: { message: string; retryAfter?: number | null; onRetry?: () => void }) {
  const [left, setLeft] = useState(retryAfter ?? 0);
  useEffect(() => {
    setLeft(retryAfter ?? 0);
    if (!retryAfter) return;
    const id = setInterval(() => setLeft((s) => (s <= 1 ? (clearInterval(id), 0) : s - 1)), 1000);
    return () => clearInterval(id);
  }, [retryAfter, message]);

  return (
    <div role="alert" className="flex flex-wrap items-start gap-3 rounded-[var(--radius-ctl)] border border-danger/40 bg-danger-soft px-4 py-3 text-sm text-ink">
      <span aria-hidden className="mt-0.5 font-bold text-danger">!</span>
      <p className="min-w-0 flex-1">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          disabled={left > 0}
          className="rounded-[var(--radius-ctl)] border border-border-strong px-3 py-1 text-xs font-medium hover:bg-surface-sunken disabled:opacity-60"
        >
          {left > 0 ? `Retry in ${left} s` : "Retry"}
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, children, icon = "◇" }: { title: string; children?: ReactNode; icon?: string }) {
  return (
    <div className="panel flex flex-col items-center justify-center gap-2 py-12 text-center">
      <span aria-hidden className="text-3xl text-ink-muted">{icon}</span>
      <p className="font-medium text-ink">{title}</p>
      {children && <div className="max-w-md text-sm text-ink-secondary">{children}</div>}
    </div>
  );
}
