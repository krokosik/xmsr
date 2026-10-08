import { act, fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import widget from "./widget";
import { initialState } from "./test-state";

it("subscribes to Python state, sends commands, and cleans up the React view", async () => {
  const values: Record<string, unknown> = {
    state: initialState, logs: [],
    axes: [{ name: "point", values: Array.from({ length: 10 }, (_, i) => String(i)), units: "", numeric: true }],
  };
  const listeners = new Map<string, Set<() => void>>();
  const model = {
    get: vi.fn((key: string) => values[key]),
    set: vi.fn(),
    save_changes: vi.fn(),
    send: vi.fn(),
    on: vi.fn((event: string, callback: () => void) => {
      if (!listeners.has(event)) listeners.set(event, new Set());
      listeners.get(event)!.add(callback);
    }),
    off: vi.fn((event?: string | null, callback?: (() => void) | null) => {
      if (event && callback) listeners.get(event)?.delete(callback);
    }),
    widget_manager: { get_model: vi.fn() },
  };
  const el = document.createElement("div");
  document.body.appendChild(el);
  const controller = new AbortController();
  const props: Parameters<typeof widget.render>[0] = {
    model, el, signal: controller.signal,
    host: { getModel: vi.fn(), getWidget: vi.fn() },
    experimental: { invoke: vi.fn() },
  };
  let cleanup: Awaited<ReturnType<typeof widget.render>>;
  await act(async () => { cleanup = await widget.render(props); });
  try {
    await userEvent.setup().click(screen.getByRole("button", { name: "Start" }));
    expect(model.send).toHaveBeenCalledWith({ command: "start" });
    act(() => {
      values.state = { ...initialState, status: "running", completed: 4 };
      for (const callback of listeners.get("change:state") ?? []) callback();
    });
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("value", "4");
    act(() => {
      values.state = { ...initialState, status: "paused", completed: 4, can_navigate: true,
        selected_indices: [3], selected_index: 3, selection_recorded: true, revision: 4 };
      for (const callback of listeners.get("change:state") ?? []) callback();
    });
    fireEvent.change(screen.getByRole("slider", { name: "point index" }), { target: { value: "1" } });
    expect(model.send).toHaveBeenLastCalledWith({ command: "select", indices: [1] });
    act(() => {
      values.state = { ...initialState, status: "paused", completed: 4, can_navigate: true,
        selected_indices: [1], selected_index: 1, selection_recorded: true, revision: 4 };
      for (const callback of listeners.get("change:state") ?? []) callback();
    });
    await userEvent.setup().click(screen.getByRole("button", { name: "Resume from selected point" }));
    expect(model.send).toHaveBeenLastCalledWith({ command: "rewind", indices: [1], revision: 4 });
  } finally {
    await act(async () => {
      controller.abort();
      if (typeof cleanup === "function") await cleanup();
    });
    el.remove();
  }
  expect([...listeners.values()].every(callbacks => callbacks.size === 0)).toBe(true);
});
