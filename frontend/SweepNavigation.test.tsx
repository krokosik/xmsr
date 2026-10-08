import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { RunnerPanel } from "./RunnerPanel";
import { initialState } from "./test-state";
import { SweepNavigation, type SweepAxis } from "./SweepNavigation";

const axes: SweepAxis[] = [
  { name: "gate", values: ["-0.7", "0.2", "1.4"], units: "V", numeric: true },
  { name: "mode", values: ["low", "high"], units: "", numeric: false },
];
const paused = {
  ...initialState, total: 6, completed: 4, status: "paused" as const,
  can_navigate: true, selected_index: 3, selected_indices: [1, 1],
  selection_recorded: true, selected_values: { gate: "0.2", mode: "high", gain: "0.4" },
  revision: 4,
};

it("uses discrete numeric indices and categorical options, with read-only derived values", async () => {
  const onCommand = vi.fn();
  const view = render(<SweepNavigation state={paused} axes={axes} onCommand={onCommand} />);
  const slider = screen.getByRole("slider", { name: "gate index" });
  expect(slider).toHaveAttribute("min", "0");
  expect(slider).toHaveAttribute("max", "2");
  expect(screen.getByText("0.2 V")).toBeInTheDocument();
  expect(screen.getByText("gain")).toBeInTheDocument();
  expect(screen.queryByRole("slider", { name: /gain/ })).not.toBeInTheDocument();
  fireEvent.change(slider, { target: { value: "0" } });
  expect(onCommand).toHaveBeenLastCalledWith({ command: "select", indices: [0, 1] });
  expect(screen.getByRole("button", { name: "Resume from selected point" })).toBeDisabled();
  view.rerender(<SweepNavigation state={{ ...paused, selected_index: 1, selected_indices: [0, 1] }} axes={axes} onCommand={onCommand} />);
  await userEvent.setup().click(screen.getByRole("button", { name: "Resume from selected point" }));
  expect(onCommand).toHaveBeenLastCalledWith({ command: "rewind", indices: [0, 1], revision: 4 });
  await userEvent.setup().selectOptions(screen.getByRole("combobox", { name: "mode" }), "0");
  expect(onCommand).toHaveBeenLastCalledWith({ command: "select", indices: [0, 0] });
});

it("does not allow jumping ahead of the acquired prefix", () => {
  render(<SweepNavigation state={{ ...paused, selected_index: 5, selected_indices: [2, 1], selection_recorded: false }} axes={axes} onCommand={vi.fn()} />);
  expect(screen.getByText(/No valid recorded data/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Resume from selected point" })).toBeDisabled();
});

it("disables navigation until an in-flight step acknowledges pause", () => {
  render(<RunnerPanel state={{ ...paused, status: "pausing", can_navigate: false }} axes={axes} logs={[]} onCommand={vi.fn()} />);
  expect(screen.getByRole("button", { name: "Pausing…" })).toBeDisabled();
  expect(screen.getByRole("slider", { name: "gate index" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled();
});

it("keeps recovery and navigation available for a recoverable error", async () => {
  const onCommand = vi.fn();
  render(<RunnerPanel state={{ ...paused, status: "failed", can_retry: true, error: "Instrument timeout" }} axes={axes} logs={[]} onCommand={onCommand} />);
  const user = userEvent.setup();
  expect(screen.getByRole("alert")).toHaveTextContent("Instrument timeout");
  expect(screen.getByRole("slider", { name: "gate index" })).toBeEnabled();
  await user.click(screen.getByRole("button", { name: "Retry failed point" }));
  expect(onCommand).toHaveBeenLastCalledWith("retry");
  await user.click(screen.getByRole("button", { name: "Show latest" }));
  expect(onCommand).toHaveBeenLastCalledWith("show_latest");
});

it("presents a reopened run without starting acquisition automatically", async () => {
  const onCommand = vi.fn();
  render(<RunnerPanel state={{ ...paused, status: "initialized", reopened: true }} axes={axes} logs={[]} onCommand={onCommand} />);
  expect(screen.getByText(/Reopened saved run/)).toBeInTheDocument();
  expect(onCommand).not.toHaveBeenCalled();
  await userEvent.setup().click(screen.getByRole("button", { name: "Resume saved run" }));
  expect(onCommand).toHaveBeenLastCalledWith("start");
});
