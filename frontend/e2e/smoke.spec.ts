import { expect, test } from "@playwright/test";

// Smoke test: sample dataset → question → chart + table. Every /api call is mocked, so no backend runs.
const SAMPLE = {
  dataset_id: "healthcare",
  name: "Healthcare admissions",
  description: "Hospital admissions.",
  rows: 999,
  has_glossary: true,
  example_questions: [{ question: "average billing amount by insurance provider", needs_glossary: false }],
  schema: [
    { name: "Insurance Provider", type: "categorical", sample_values: ["Medicare", "Aetna"] },
    { name: "Billing Amount", type: "numeric", sample_values: [37490.98] },
  ],
};

const RESULT = {
  question: "average billing amount by insurance provider",
  intent: { label: "aggregate", confidence: 0.99, source: "bert" },
  generator: {
    used: "llm", requested_mode: "auto", llm_strategy: "zero_shot", model: "openai/gpt-oss-20b", provider: "groq",
    fallback_reason: null, fallback_detail: null, rejected_sql: null,
  },
  sql: "SELECT `Insurance Provider`, AVG(`Billing Amount`) AS avg_billing FROM data GROUP BY `Insurance Provider`",
  columns: [{ name: "Insurance Provider", type: "text" }, { name: "avg_billing", type: "number" }],
  rows: [["Aetna", 25556.26], ["Blue Cross", 25801.84], ["Cigna", 25777.96], ["Medicare", 24818.21]],
  row_count: 4,
  truncated: false,
  chart: { type: "bar", x: "Insurance Provider", y: "avg_billing", reason: "categories × one metric" },
  context: { glossary_checked: true, glossary: [], examples: [] },
  latency_ms: { intent: 30, retrieval: 0, generation: 900, execution: 2, total: 940 },
  notes: [],
};

test("sample dataset → question → chart and table", async ({ page }) => {
  await page.route("**/api/config", (r) => r.fulfill({ json: {
    default_mode: "auto", glossary_default: true, llm_provider: "groq", llm_model: "openai/gpt-oss-20b",
    llm_budget_remaining: 500, max_upload_mb: 10, max_rows_returned: 500,
  } }));
  await page.route("**/api/samples", (r) => r.fulfill({ json: { samples: [SAMPLE] } }));
  let sent: unknown = null;
  await page.route("**/api/query", async (r) => {
    sent = r.request().postDataJSON();
    await r.fulfill({ json: RESULT });
  });

  await page.goto("/");
  await expect(page.getByText("Pick a dataset to start")).toBeVisible();

  await page.getByRole("radio", { name: /Healthcare admissions/ }).click();
  await page.getByRole("button", { name: "average billing amount by insurance provider" }).click();

  const chart = page.getByTestId("chart-bar");
  await expect(chart).toBeVisible();
  await expect(chart.locator(".recharts-bar-rectangle")).toHaveCount(4);
  await expect(page.getByRole("heading", { name: "Avg billing by Insurance Provider" })).toBeVisible();

  const table = page.getByRole("region", { name: "Result table" });
  await expect(table.getByRole("row")).toHaveCount(5); // header + 4
  await expect(table).toContainText("Medicare");
  await expect(table).toContainText("24,818.21");

  await expect(page.getByRole("region", { name: "SQL" })).toContainText("AVG(`Billing Amount`)");
  await expect(page.getByText("LLM zero-shot")).toBeVisible();
  await expect(page.getByText("No glossary term appears in the question, so no definitions were sent.")).toBeVisible();
  expect(sent).toEqual({ dataset_id: "healthcare", question: "average billing amount by insurance provider", mode: "auto", use_glossary: true });

  await page.getByLabel("Chart type").selectOption("pie");
  await expect(page.getByTestId("chart-pie")).toBeVisible();
});
