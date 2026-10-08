import type { RunnerState } from "./RunnerPanel";

export const initialState: RunnerState = {
  name: "DemoMeasurement", status: "initialized", completed: 0, total: 10,
  elapsed: 0, remaining: null, rate: 0, filename: "demo.zarr",
  path: "/data/demo.zarr", error: "",
};
