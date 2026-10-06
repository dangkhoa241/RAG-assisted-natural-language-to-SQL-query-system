import { afterEach, describe, expect, it, vi } from "vitest";

describe("API base URL", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  async function firstUrl(base: string | undefined) {
    if (base !== undefined) vi.stubEnv("VITE_API_BASE_URL", base);
    const fetchMock = vi.fn(() => Promise.resolve(new Response(JSON.stringify({ status: "ok" }), { status: 200 })));
    vi.stubGlobal("fetch", fetchMock);
    const { api } = await import("../api/client");
    await api.health();
    return (fetchMock.mock.calls[0] as unknown as [string])[0];
  }

  it("prefixes VITE_API_BASE_URL (without a trailing slash) in production", async () => {
    expect(await firstUrl("https://nl2sql-api.onrender.com/")).toBe("https://nl2sql-api.onrender.com/api/health");
  });

  it("uses same-origin paths when it isn't set (the Vite dev proxy)", async () => {
    expect(await firstUrl(undefined)).toBe("/api/health");
  });
});

describe("withWakeRetry", () => {
  it("does not retry a real error from the backend", async () => {
    const { ApiError, withWakeRetry } = await import("../api/client");
    const call = vi.fn(() => Promise.reject(new ApiError("Too many LLM questions.", 429, "rate_limited")));
    const waking = vi.fn();
    await expect(withWakeRetry(call, waking)).rejects.toThrow("Too many LLM questions.");
    expect(call).toHaveBeenCalledTimes(1);
    expect(waking).toHaveBeenLastCalledWith(false);
  });
});
