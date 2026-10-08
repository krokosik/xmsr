import type { RunnerState } from "./RunnerPanel";

export const initialState: RunnerState = {
  name: "DemoMeasurement", status: "initialized", completed: 0, total: 10,
  elapsed: 0, remaining: null, rate: 0, filename: "demo.zarr",
  path: "/data/demo.zarr", error: "",
  can_navigate: false, can_retry: false, selected_index: 0, selected_indices: [0],
  selection_recorded: false, selected_values: { point: "0" }, revision: 0,
  reopened: false, command_error: "",
};
