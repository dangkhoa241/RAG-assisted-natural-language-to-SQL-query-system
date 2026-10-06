import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { HowItWorks } from "../components/HowItWorks";
import { ResultChart } from "../components/ResultChart";
import { ResultTable } from "../components/ResultTable";
import { SqlBlock } from "../components/SqlBlock";
import { makeResult } from "./fixtures";

describe("ResultChart", () => {
  it("renders the suggested chart and lets the user switch type", async () => {
    const user = userEvent.setup();
    render(<ResultChart result={makeResult()} theme="light" />);
    expect(screen.getByTestId("chart-bar")).toBeInTheDocument();
    const select = screen.getByLabelText("Chart type");
    expect(within(select).getAllByRole("option").map((o) => o.textContent)).toEqual(["Bar", "Line", "Pie", "Table only"]);

    await user.selectOptions(select, "pie");
    expect(screen.getByTestId("chart-pie")).toBeInTheDocument();
    const legend = screen.getByRole("list", { name: "Legend" });
    expect(within(legend).getByText("East")).toBeInTheDocument();
    expect(within(legend).getByText("50%")).toBeInTheDocument();

    await user.selectOptions(select, "line");
    expect(screen.getByTestId("chart-line")).toBeInTheDocument();

    await user.selectOptions(select, "table");
    expect(screen.getByText(/reads best as a table/)).toBeInTheDocument();
  });

  it("shows a single number as a stat card", () => {
    const r = makeResult({ columns: [{ name: "COUNT(*)", type: "number" }], rows: [[1234]], chart: { type: "stat", x: null, y: "COUNT(*)" } });
    render(<ResultChart result={r} theme="dark" />);
    const card = screen.getByTestId("stat-card");
    expect(card).toHaveTextContent("Count");
    expect(card).toHaveTextContent("1,234");
  });
});

describe("ResultTable", () => {
  const many = makeResult({ rows: Array.from({ length: 23 }, (_, i) => [`r${i + 1}`, i + 1]), row_count: 23 });

  it("paginates", async () => {
    const user = userEvent.setup();
    render(<ResultTable result={many} />);
    expect(screen.getByText("Page 1 of 3")).toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(11); // header + 10
    await user.click(screen.getByRole("button", { name: "Next page" }));
    expect(screen.getByText("Page 2 of 3")).toBeInTheDocument();
    expect(screen.getByText("r11")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Rows per page"), "25");
    expect(screen.getByText("Page 1 of 1")).toBeInTheDocument();
  });

  it("sorts ascending, then descending, then back to the original order", async () => {
    const user = userEvent.setup();
    render(<ResultTable result={makeResult()} />);
    const header = screen.getByRole("button", { name: /total/ });
    const firstCell = () => screen.getAllByRole("row")[1].querySelector("td")!.textContent;
    expect(firstCell()).toBe("East");
    await user.click(header);
    expect(firstCell()).toBe("West");
    expect(screen.getByRole("columnheader", { name: /total/ })).toHaveAttribute("aria-sort", "ascending");
    await user.click(header);
    expect(firstCell()).toBe("East");
    await user.click(header);
    expect(screen.getByRole("columnheader", { name: /total/ })).toHaveAttribute("aria-sort", "none");
  });

  it("shows an empty state and the truncation note", () => {
    const { rerender } = render(<ResultTable result={makeResult({ rows: [], row_count: 0 })} />);
    expect(screen.getByText(/returned no rows/)).toBeInTheDocument();
    rerender(<ResultTable result={makeResult({ truncated: true, row_count: 900 })} />);
    expect(screen.getByText(/showing the first 3/)).toBeInTheDocument();
  });
});

describe("SqlBlock", () => {
  it("copies the SQL", async () => {
    const user = userEvent.setup();
    const writeText = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue();
    render(<SqlBlock sql="SELECT 1" />);
    await user.click(screen.getByRole("button", { name: "Copy" }));
    expect(writeText).toHaveBeenCalledWith("SELECT 1");
    expect(await screen.findByRole("button", { name: "Copied ✓" })).toBeInTheDocument();
  });
});

describe("HowItWorks", () => {
  it("prompts before the first question", () => {
    render(<HowItWorks result={null} />);
    expect(screen.getByText(/Ask a question to see each step/)).toBeInTheDocument();
  });

  it("explains intent, generator, matched glossary terms and latency", () => {
    const r = makeResult({
      generator: { ...makeResult().generator, llm_strategy: "glossary_rag" },
      context: {
        glossary_checked: true,
        glossary: [{ id: "g1", term: "Repeat customer", matched: "repeat buyer", definition: "3 or more orders." }],
        examples: [],
      },
    });
    render(<HowItWorks result={r} />);
    expect(screen.getByText("aggregate")).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "Intent confidence" })).toHaveAttribute("aria-valuenow", "97");
    expect(screen.getByText("LLM with glossary definitions")).toBeInTheDocument();
    expect(screen.getByText("openai/gpt-oss-120b · Groq")).toBeInTheDocument();
    const terms = screen.getByRole("list", { name: "Matched glossary terms" });
    expect(terms).toHaveTextContent("Repeat customer");
    expect(terms).toHaveTextContent("“repeat buyer”");
    expect(terms).toHaveTextContent("3 or more orders.");   // the definition sent is visible, not collapsed
    expect(screen.getByText(/1 term matched/)).toBeInTheDocument();
    expect(screen.getByText("830 ms")).toBeInTheDocument();
  });

  it("says when no glossary term matched, and hides the glossary step when it was off", () => {
    const { unmount } = render(<HowItWorks result={makeResult()} />);
    expect(screen.getByText(/No glossary term appears in the question/)).toBeInTheDocument();
    unmount();
    render(<HowItWorks result={makeResult({ context: { glossary_checked: false, glossary: [], examples: [] } })} />);
    expect(screen.queryByText(/Business glossary/)).not.toBeInTheDocument();
  });

  it("shows why the generator fell back", () => {
    const r = makeResult({
      generator: {
        used: "rule_based", requested_mode: "auto", llm_strategy: "zero_shot", model: null, provider: null,
        model_note: null, models_tried: [{ model: "openai/gpt-oss-120b", outcome: "rejected" }],
        fallback_reason: "unsafe_sql",
        fallback_detail: "The LLM's SQL was rejected by the safety check (forbidden keyword(s): DROP).",
        rejected_sql: "DROP TABLE data",
      },
    });
    render(<HowItWorks result={r} />);
    expect(screen.getByText("Rule-based")).toBeInTheDocument();
    expect(screen.getByText(/Fell back: LLM SQL blocked by the safety check/)).toBeInTheDocument();
    expect(screen.getByText("DROP TABLE data")).toBeInTheDocument();
  });

  it("names the fallback model when the primary model was out of quota", () => {
    const base = makeResult();
    const r = makeResult({
      generator: {
        ...base.generator, model: "openai/gpt-oss-20b", model_note: "gpt-oss-20b (120b quota exhausted)",
        models_tried: [
          { model: "openai/gpt-oss-120b", outcome: "quota_exhausted" },
          { model: "openai/gpt-oss-20b", outcome: "answered" },
        ],
      },
    });
    render(<HowItWorks result={r} />);
    expect(screen.getByText("openai/gpt-oss-20b · Groq")).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent("Answered by gpt-oss-20b (120b quota exhausted)");
  });

  it("shows no fallback note when the primary model answered", () => {
    render(<HowItWorks result={makeResult()} />);
    expect(screen.getByText("openai/gpt-oss-120b · Groq")).toBeInTheDocument();
    expect(screen.queryByText(/Answered by/)).not.toBeInTheDocument();
  });
});
