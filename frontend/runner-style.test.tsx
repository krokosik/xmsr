import { readFileSync } from "node:fs";
import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { RunnerPanel } from "./RunnerPanel";
import { initialState } from "./test-state";

const runnerCss = readFileSync("frontend/runner.css", "utf8");

it("keeps the runner and VS Code notebook widget outputs transparent", () => {
  const style = document.createElement("style");
  style.textContent = `
    .cell-output-ipywidget-background { background-color: white; }
    ${runnerCss}
  `;
  document.head.appendChild(style);
  try {
    render(<>
      <div className="cell-output-ipywidget-background" data-testid="runner-output">
        <RunnerPanel state={initialState} logs={[]} onCommand={vi.fn()} />
      </div>
      <div className="cell-output-ipywidget-background" data-testid="other-output">Other widget</div>
    </>);
    expect(getComputedStyle(screen.getByRole("region")).backgroundColor).toBe("rgba(0, 0, 0, 0)");
    expect(getComputedStyle(screen.getByTestId("runner-output")).backgroundColor).toBe("rgba(0, 0, 0, 0)");
    expect(getComputedStyle(screen.getByTestId("other-output")).backgroundColor).toBe("rgba(0, 0, 0, 0)");
  } finally {
    style.remove();
  }
});
