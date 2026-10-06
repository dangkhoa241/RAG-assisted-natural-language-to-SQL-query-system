import type { AppConfig, DatasetInfo, QueryRequest, QueryResult } from "./types";

/** An error the UI can show as-is: the backend's message, or a plain-language network message. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string,
    readonly retryAfter: number | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** The API's origin in production (e.g. https://nl2sql-api.onrender.com). Empty in development, where
 * Vite proxies /api to the local backend, so requests stay same-origin. */
export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");

const UNREACHABLE = API_BASE
  ? "Can't reach the API server. Try again in a minute."
  : "Can't reach the API server. Is the backend running on port 8000?";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_BASE + path, init);
  } catch {
    // Also what a sleeping or restarting free-tier backend looks like: its proxy's error page has no CORS
    // headers, so the browser reports a network error.
    throw new ApiError(UNREACHABLE, 0, "network_error");
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // non-JSON (e.g. a proxy error page); handled below
  }
  if (!res.ok) {
    const err = (body as { error?: { code?: string; message?: string } } | null)?.error;
    // The backend always answers with a JSON error body. A bare 502/503/504 comes from the proxy in
    // front of it (Vite in development, the host's router in production) and means it isn't reachable.
    if (!err && [502, 503, 504].includes(res.status)) {
      throw new ApiError(UNREACHABLE, res.status, "backend_unreachable");
    }
    const retry = Number(res.headers.get("Retry-After"));
    throw new ApiError(
      err?.message ?? `The server returned an error (${res.status}).`,
      res.status,
      err?.code ?? "http_error",
      Number.isFinite(retry) && retry > 0 ? retry : null,
    );
  }
  return body as T;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),
  config: () => request<AppConfig>("/api/config"),
  samples: () => request<{ samples: DatasetInfo[] }>("/api/samples").then((r) => r.samples),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DatasetInfo>("/api/datasets", { method: "POST", body: form });
  },
  query: (body: QueryRequest) =>
    request<QueryResult>("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};

// --- Waking a sleeping backend --------------------------------------------------------------------------
// The free host stops the backend after 15 idle minutes and takes up to a minute to start it again. Until
// then requests fail at the network level or get a bare 502/503 from the host's router. Those (and only
// those: a JSON error from the backend is a real answer) are retried with backoff while the UI says the
// server is waking up.

export const WAKING_MESSAGE = "Waking up the server, this can take up to a minute…";

/** Timing of the retries; tests shorten it. */
export const wakePolicy = {
  slowMs: 2500, // a first response slower than this is treated as a cold start too
  delaysMs: [1000, 2000, 4000, 8000, 10000], // between attempts; the last one repeats
  maxWaitMs: 120_000, // give up (and show the error) after this long
};

export function isWaking(e: unknown): boolean {
  return e instanceof ApiError && (e.code === "network_error" || e.code === "backend_unreachable");
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Runs `call`, retrying while the backend looks asleep. `onWaking(true)` fires once the call has failed
 * that way or taken longer than `wakePolicy.slowMs`; `onWaking(false)` when it settles either way. */
export async function withWakeRetry<T>(call: () => Promise<T>, onWaking: (waking: boolean) => void): Promise<T> {
  const started = Date.now();
  const slow = setTimeout(() => onWaking(true), wakePolicy.slowMs);
  try {
    for (let attempt = 0; ; attempt++) {
      try {
        return await call();
      } catch (e) {
        if (!isWaking(e) || Date.now() - started >= wakePolicy.maxWaitMs) throw e;
        onWaking(true);
        await sleep(wakePolicy.delaysMs[Math.min(attempt, wakePolicy.delaysMs.length - 1)]);
      }
    }
  } finally {
    clearTimeout(slow);
    onWaking(false);
  }
}

/** Fire-and-forget request on page load, so a sleeping backend starts booting before the user does anything. */
export function pingBackend(): void {
  api.health().catch(() => {
    // the startup requests that follow retry and report; nothing to do here
  });
}
