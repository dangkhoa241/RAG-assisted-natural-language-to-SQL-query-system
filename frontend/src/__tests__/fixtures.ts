import type { DatasetInfo, QueryResult } from "../api/types";

export function makeResult(overrides: Partial<QueryResult> = {}): QueryResult {
  return {
    question: "total revenue by region",
    intent: { label: "aggregate", confidence: 0.97, source: "bert" },
    generator: {
      used: "llm", requested_mode: "auto", llm_strategy: "zero_shot", model: "openai/gpt-oss-20b", provider: "groq",
      fallback_reason: null, fallback_detail: null, rejected_sql: null,
    },
    sql: "SELECT `Region`, SUM(`Revenue`) AS total FROM data GROUP BY `Region`",
    columns: [{ name: "Region", type: "text" }, { name: "total", type: "number" }],
    rows: [["East", 300], ["West", 100], ["North", 200]],
    row_count: 3,
    truncated: false,
    chart: { type: "bar", x: "Region", y: "total", reason: "categories × one metric" },
    context: { glossary_checked: true, glossary: [], examples: [] },
    latency_ms: { intent: 20, retrieval: 0, generation: 800, execution: 3, total: 830 },
    notes: [],
    ...overrides,
  };
}

export const SAMPLE: DatasetInfo = {
  dataset_id: "retail",
  name: "Retail orders",
  description: "Orders.",
  rows: 1000,
  has_glossary: true,
  example_questions: [
    { question: "total revenue by region", needs_glossary: false },
    { question: "how many repeat customers do we have?", needs_glossary: true },
  ],
  schema: [
    { name: "Region", type: "categorical", sample_values: ["East", "West"] },
    { name: "Revenue", type: "numeric", sample_values: [299.55] },
  ],
};
