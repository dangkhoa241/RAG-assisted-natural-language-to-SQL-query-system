import { describe, expect, it } from "vitest";

import { availableChartTypes, planChart, toRecords } from "../lib/chart";
import { humanize } from "../lib/format";
import { makeResult } from "./fixtures";

describe("availableChartTypes", () => {
  it("offers bar, line, pie and table for a few non-negative categories", () => {
    expect(availableChartTypes(makeResult())).toEqual(["bar", "line", "pie", "table"]);
  });

  it("offers only the number card and table for a single value", () => {
    const r = makeResult({ columns: [{ name: "n", type: "number" }], rows: [[42]], chart: { type: "stat", x: null, y: "n" } });
    expect(availableChartTypes(r)).toEqual(["stat", "table"]);
  });

  it("drops pie for negative values or too many slices", () => {
    expect(availableChartTypes(makeResult({ rows: [["a", -1], ["b", 2]] }))).not.toContain("pie");
    const many = Array.from({ length: 9 }, (_, i) => [`c${i}`, i]);
    expect(availableChartTypes(makeResult({ rows: many }))).not.toContain("pie");
  });

  it("offers only table when there is nothing numeric or no rows", () => {
    expect(availableChartTypes(makeResult({ columns: [{ name: "a", type: "text" }], rows: [["x"]] }))).toEqual(["table"]);
    expect(availableChartTypes(makeResult({ rows: [] }))).toEqual(["table"]);
  });
});

describe("planChart", () => {
  it("follows the server suggestion and its axes", () => {
    expect(planChart(makeResult())).toEqual({ type: "bar", x: "Region", y: "total" });
  });

  it("honours a user pick", () => {
    expect(planChart(makeResult(), "pie")).toEqual({ type: "pie", x: "Region", y: "total" });
    expect(planChart(makeResult(), "table")).toEqual({ type: "table", x: null, y: null });
  });

  it("falls back to the first valid type when the suggestion doesn't fit", () => {
    const r = makeResult({ rows: [["a", -5], ["b", 3]], chart: { type: "pie", x: "Region", y: "total" } });
    expect(planChart(r).type).toBe("bar");
  });

  it("infers axes when the server gave none", () => {
    const r = makeResult({ chart: { type: "table", x: null, y: null } });
    expect(planChart(r, "line")).toEqual({ type: "line", x: "Region", y: "total" });
  });

  it("uses two numeric columns (e.g. integer years) as x and y", () => {
    const r = makeResult({
      columns: [{ name: "year", type: "number" }, { name: "n", type: "number" }],
      rows: [[2020, 1], [2021, 2]],
      chart: { type: "line", x: "year", y: "n" },
    });
    expect(planChart(r)).toEqual({ type: "line", x: "year", y: "n" });
  });
});

describe("toRecords", () => {
  it("maps rows to {x, y} and skips non-numeric values", () => {
    const r = makeResult({ rows: [["East", 1], [null, 2], ["West", null]] });
    expect(toRecords(r, "Region", "total")).toEqual([{ x: "East", y: 1 }, { x: "(empty)", y: 2 }]);
  });
});

describe("humanize", () => {
  it.each([
    ["total_revenue", "Total revenue"],
    ["AVG(`Billing Amount`)", "Average Billing Amount"],
    ["COUNT(*)", "Count"],
    ["sum(Revenue)", "Total Revenue"],
  ])("%s → %s", (input, output) => expect(humanize(input)).toBe(output));
});
