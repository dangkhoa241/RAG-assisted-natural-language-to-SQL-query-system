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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError("Can't reach the server. Is the backend running?", 0, "network_error");
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // non-JSON (e.g. a proxy error page); handled below
  }
  if (!res.ok) {
    const err = (body as { error?: { code?: string; message?: string } } | null)?.error;
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
