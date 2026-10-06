// Mirrors the backend's JSON (backend/main.py, backend/query_service.py).

export type Mode = "auto" | "llm" | "rule_based";
export type ChartType = "bar" | "line" | "pie" | "table" | "stat";
export type ColumnKind = "numeric" | "categorical" | "date" | "text";
export type Cell = string | number | boolean | null;
export type ModelOutcome = "answered" | "quota_exhausted" | "rate_limited" | "failed" | "rejected";

export interface SchemaColumn {
  name: string;
  type: ColumnKind;
  sample_values: Cell[];
}

export interface ExampleQuestion {
  question: string;
  needs_glossary: boolean;
}

export interface DatasetInfo {
  dataset_id: string;
  name: string;
  description?: string;
  rows: number;
  has_glossary: boolean;
  example_questions: ExampleQuestion[];
  schema: SchemaColumn[];
  expires_in_s?: number;
}

export interface AppConfig {
  default_mode: Mode;
  glossary_default: boolean;
  llm_provider: "groq" | "cerebras" | null;
  llm_model: string | null;
  /** Tried in order when the primary model is out of quota or rate-limited. */
  llm_fallback_models: string[];
  llm_budget_remaining: number;
  max_upload_mb: number;
  max_rows_returned: number;
}

export interface ChartSpec {
  type: ChartType;
  x: string | null;
  y: string | null;
  reason?: string;
}

export interface QueryRequest {
  dataset_id: string;
  question: string;
  mode: Mode;
  use_glossary: boolean;
}

export interface QueryResult {
  question: string;
  intent: { label: string; confidence: number | null; source: "bert" | "keywords" };
  generator: {
    used: "llm" | "rule_based";
    requested_mode: Mode;
    llm_strategy: "zero_shot" | "example_rag" | "glossary_rag" | null;
    /** The model whose SQL is used (LLM answers only). */
    model: string | null;
    /** Set when a fallback model answered, e.g. "gpt-oss-20b (120b quota exhausted)". */
    model_note: string | null;
    models_tried: { model: string; outcome: ModelOutcome }[];
    provider: "groq" | "cerebras" | null;
    fallback_reason: string | null;
    fallback_detail: string | null;
    rejected_sql: string | null;
  };
  sql: string;
  columns: { name: string; type: "number" | "text" }[];
  rows: Cell[][];
  row_count: number;
  truncated: boolean;
  chart: ChartSpec;
  context: {
    /** True when the question was checked against the dataset's glossary (only matched terms are sent). */
    glossary_checked: boolean;
    /** The definitions sent to the LLM; `matched` is the term or alias found in the question. */
    glossary: { id: string; term: string; matched: string; definition: string }[];
    examples: { question: string; sql: string; score: number }[];
  };
  latency_ms: { intent: number; retrieval: number; generation: number; execution: number; total: number };
  notes: string[];
}
