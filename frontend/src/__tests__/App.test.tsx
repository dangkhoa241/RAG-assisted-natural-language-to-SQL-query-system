import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "../App";
import { makeResult, SAMPLE } from "./fixtures";

const CONFIG = {
  default_mode: "auto", glossary_default: false, llm_model: "openai/gpt-oss-120b",
  llm_budget_remaining: 500, max_upload_mb: 10, max_rows_returned: 500,
};

function json(body: unknown, status = 200, headers: Record<string, string> = {}) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } }));
}

let queryResponse: () => Promise<Response>;
const fetchMock = vi.fn((url: string, init?: RequestInit) => {
  if (url === "/api/config") return json(CONFIG);
  if (url === "/api/samples") return json({ samples: [SAMPLE] });
  if (url === "/api/query") return queryResponse();
  if (url === "/api/datasets") {
    return json({ ...SAMPLE, dataset_id: "up1", name: "mine.csv", has_glossary: false, example_questions: [] }, 201);
  }
  throw new Error(`unexpected ${init?.method ?? "GET"} ${url}`);
});

function lastQueryBody() {
  const call = fetchMock.mock.calls.filter(([u]) => u === "/api/query").at(-1)!;
  return JSON.parse(call[1]!.body as string);
}

beforeEach(() => {
  queryResponse = () => json(makeResult());
  fetchMock.mockClear();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

describe("App", () => {
  it("goes from picking a dataset to a chart, table and SQL", async () => {
    const user = userEvent.setup();
    render(<App />);
    expect(await screen.findByText("Pick a dataset to start")).toBeInTheDocument();

    await user.click(await screen.findByRole("radio", { name: /Retail orders/ }));
    expect(screen.getByText("Schema · 2 columns")).toBeInTheDocument();
    expect(screen.getByText("Ask your first question")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "total revenue by region" }));
    expect(await screen.findByTestId("chart-bar")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Result table" })).toHaveTextContent("East");
    expect(screen.getByRole("region", { name: "SQL" })).toHaveTextContent("SUM(`Revenue`)");
    expect(screen.getByLabelText("Ask a question about the data")).toHaveValue("total revenue by region");
    expect(lastQueryBody()).toEqual({ dataset_id: "retail", question: "total revenue by region", mode: "auto", use_glossary: false });
  });

  it("sends the chosen mode and glossary toggle", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("radio", { name: /Retail orders/ }));
    await user.selectOptions(screen.getByRole("combobox", { name: /Mode/ }), "llm");
    await user.click(screen.getByLabelText("Use business glossary"));
    await user.type(screen.getByLabelText("Ask a question about the data"), "how many repeat customers{Enter}");
    await screen.findByTestId("chart-bar");
    expect(lastQueryBody()).toMatchObject({ mode: "llm", use_glossary: true, question: "how many repeat customers" });
  });

  it("shows the server's error with a retry countdown when rate-limited", async () => {
    queryResponse = () =>
      json({ error: { code: "rate_limited", message: "Too many LLM questions. Try again in 30 s." } }, 429, { "Retry-After": "30" });
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("radio", { name: /Retail orders/ }));
    await user.click(screen.getByRole("button", { name: "total revenue by region" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Too many LLM questions");
    expect(screen.getByRole("button", { name: "Retry in 30 s" })).toBeDisabled();
  });

  it("shows a loading state while the query runs", async () => {
    let resolve!: (r: Response) => void;
    queryResponse = () => new Promise((r) => { resolve = r; });
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("radio", { name: /Retail orders/ }));
    await user.click(screen.getByRole("button", { name: "total revenue by region" }));
    expect(screen.getByRole("status", { name: "Running query" })).toBeInTheDocument();
    resolve(new Response(JSON.stringify(makeResult()), { status: 200 }));
    expect(await screen.findByTestId("chart-bar")).toBeInTheDocument();
  });

  it("explains when the backend is unreachable", async () => {
    fetchMock.mockImplementationOnce(() => Promise.reject(new TypeError("Failed to fetch")));
    render(<App />);
    expect(await screen.findByText(/Can't reach the API server/)).toBeInTheDocument();
  });

  it("explains a bare proxy 502 (backend down behind Vite) the same way", async () => {
    const proxy502 = () => Promise.resolve(new Response("", { status: 502 }));
    fetchMock.mockImplementationOnce(proxy502).mockImplementationOnce(proxy502); // /api/config + /api/samples
    render(<App />);
    expect(await screen.findByText(/Can't reach the API server\. Is the backend running on port 8000\?/)).toBeInTheDocument();
  });

  it("still shows the backend's own JSON 502 message", async () => {
    queryResponse = () => json({ error: { code: "llm_error", message: "The LLM call failed. Try again." } }, 502);
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole("radio", { name: /Retail orders/ }));
    await user.selectOptions(screen.getByRole("combobox", { name: /Mode/ }), "llm");
    await user.click(screen.getByRole("button", { name: "total revenue by region" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The LLM call failed. Try again.");
  });

  it("uploads a CSV, selects it and disables the glossary for it", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByRole("radio", { name: /Retail orders/ });
    await user.upload(screen.getByLabelText("CSV file"), new File(["a,b\n1,2\n"], "mine.csv", { type: "text/csv" }));
    await waitFor(() => expect(screen.getByRole("radio", { name: /mine\.csv/ })).toHaveAttribute("aria-checked", "true"));
    expect(screen.getByLabelText("Use business glossary")).toBeDisabled();
  });

  it("rejects non-CSV files before uploading", async () => {
    const user = userEvent.setup({ applyAccept: false });
    render(<App />);
    await screen.findByRole("radio", { name: /Retail orders/ });
    await user.upload(screen.getByLabelText("CSV file"), new File(["x"], "data.xlsx"));
    expect(screen.getByText("Only .csv files are supported.")).toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([u]) => u === "/api/datasets")).toBe(false);
  });
});
