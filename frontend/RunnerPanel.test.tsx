import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { RunnerPanel } from "./RunnerPanel";
import { initialState } from "./test-state";


describe("runner controls", () => {
  it("requests explicit commands and waits for Python's state", async () => {
    const onCommand = vi.fn();
    const user = userEvent.setup();
    const view = render(<RunnerPanel state={initialState} logs={[]} onCommand={onCommand} />);
    await user.click(screen.getByRole("button", { name: "Start" }));
    expect(onCommand).toHaveBeenLastCalledWith("start");
    expect(screen.getByRole("status")).toHaveTextContent("initialized");
    view.rerender(<RunnerPanel state={{ ...initialState, status: "running", completed: 3 }} logs={[]} onCommand={onCommand} />);
    await user.click(screen.getByRole("button", { name: "Pause" }));
    expect(onCommand).toHaveBeenLastCalledWith("pause");
    view.rerender(<RunnerPanel state={{ ...initialState, status: "paused" }} logs={[]} onCommand={onCommand} />);
    await user.click(screen.getByRole("button", { name: "Resume" }));
    expect(onCommand).toHaveBeenLastCalledWith("resume");
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCommand).toHaveBeenLastCalledWith("cancel");
  });

  it.each(["finished", "cancelled", "failed"] as const)("disables execution commands when %s", status => {
    render(<RunnerPanel state={{ ...initialState, status }} logs={[]} onCommand={vi.fn()} />);
    expect(screen.queryByRole("button", { name: /^(Start|Pause|Resume|Cancel)$/ })).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(status);
  });

  it("renders resumed progress, errors, and logs as text", () => {
    render(<RunnerPanel state={{ ...initialState, status: "failed", completed: 6, error: "Instrument disconnected" }}
      logs={[{ id: 1, time: "12:00:00", level: "warning", message: "<script>bad()</script>" }]} onCommand={vi.fn()} />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("value", "6");
    expect(screen.getByText(/6 \/ 10 steps/)).toHaveTextContent("60.0%");
    expect(screen.getByRole("alert")).toHaveTextContent("Instrument disconnected");
    expect(screen.getByRole("log")).toHaveTextContent("<script>bad()</script>");
    expect(document.querySelector("script")).toBeNull();
  });

  it("copies the Python-provided path and reports clipboard failures", async () => {
    const user = userEvent.setup();
    const copy = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue();
    render(<RunnerPanel state={initialState} logs={[]} onCommand={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: "Copy path" }));
    expect(copy).toHaveBeenCalledWith("/data/demo.zarr");
    expect(screen.getByText("Copied")).toBeInTheDocument();
    copy.mockRejectedValueOnce(new Error("Unavailable"));
    await user.click(screen.getByRole("button", { name: "Copy filename" }));
    expect(screen.getByText("Could not copy to clipboard")).toBeInTheDocument();
  });
});
