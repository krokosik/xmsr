# xmsr - Xarray Measurement Wrapper

A Python framework for running parametric measurements and storing results in Zarr using xarray-native schemas.

## Overview

`Measurement` is designed for multi-dimensional parameter sweeps where metadata, dimensions, and coordinates are first-class parts of the dataset.

You define two xarray templates:

- `sweep_template`: scan dimensions and sweep coordinates
- `result_template`: per-point measurement payload schema

The framework handles iteration, storage, resume, and optional interactive controls.

## Template Contract

- `sweep_template` must be an `xr.Dataset`/`xr.DataArray` with at least one dimension.
- `result_template` defines the per-point payload schema:
  - `xr.DataArray`: single-variable output
  - `xr.Dataset`: multi-variable output
- `measure(values, indices, metadata)` is called once per sweep point.

Auto-wrap behavior:

- If `result_template` is `DataArray`, you may return a single numpy array/scalar.
- If `result_template` is `Dataset`, you may return a tuple mapped by variable order in
  `result_template.data_vars`.
- Returned shapes must exactly match template variable shapes.

## Features

- Automatic parameter scanning over sweep coordinates
- Zarr-backed storage with resume support
- Non-blocking execution in a background thread
- Xarray-native data model (no custom coord serialization layer)
- React/anywidget notebook controls, progress, and structured logs
- Optional HoloViews/Bokeh live plotting

## Examples

- `examples/01_single_variable.py`
- `examples/02_multi_variable.py`
- `examples/03_paths_and_metadata.py`
- `examples/04_multiparam_derived_coords.py`
- `examples/06_live_plotting.py`
- `examples/07_manual_operator_step.py`

## Basic Example

```python
import numpy as np
import xarray as xr

from xmsr import Measurement


class BasicMeasurement(Measurement):
    target_directory = "data"

    sweep_template = xr.Dataset().assign_coords(
        gate=np.linspace(-1.0, 1.0, 21),
        bias=np.linspace(-0.2, 0.2, 31),
        power=("gate", np.linspace(-1.0, 1.0, 21) ** 2),
    )

    result_template = xr.DataArray(
        np.empty((256,), dtype=np.float32),
        dims=("freq",),
        coords={"freq": np.linspace(1e6, 10e6, 256)},
        name="amplitude",
    )

    def measure(self, values, indices, metadata):
        # values: {'gate': ..., 'bias': ...}
        # indices: {'gate': ..., 'bias': ...}
        return np.random.randn(256).astype(np.float32)


measurement = BasicMeasurement()
measurement.run()
result = measurement.result
```

## Composite Sweep Parameters

For composite parameters, prefer scalar sweep dimensions with derived coordinates.

```python
import numpy as np
import xarray as xr

x = np.linspace(0.0, 2.0, 6)

sweep_template = xr.Dataset().assign_coords(x=x).assign_coords(
    x_gain=("x", 2.0 * x + 1.0),
    x_offset=("x", np.sin(x)),
)
```

## Multi-Variable Example

```python
import numpy as np
import xarray as xr

from xmsr import Measurement


class MultiMeasurement(Measurement):
    sweep_template = xr.Dataset().assign_coords(
        temperature=np.array([4.2, 10.0, 20.0]),
        field=np.linspace(-2.0, 2.0, 101),
    )

    result_template = xr.Dataset(
        {
            "voltage": xr.DataArray(
                np.empty((500,), dtype=np.float64),
                dims=("time",),
                coords={"time": np.linspace(0, 1, 500)},
            ),
            "current": xr.DataArray(
                np.empty((500,), dtype=np.float64),
                dims=("time",),
                coords={"time": np.linspace(0, 1, 500)},
            ),
            "resistance": xr.DataArray(np.empty((), dtype=np.float64), dims=()),
        }
    )

    def measure(self, values, indices, metadata):
        voltage = np.random.randn(500)
        current = np.random.randn(500)
        resistance = float(voltage.mean() / (current.mean() + 1e-12))
        return voltage, current, resistance
```

## Configuration

Runtime behavior can be configured via class attributes or constructor args:

- `metadata`
- `target_directory`
- `filename`
- `timestamp` (default: `True`)
- `with_coords` (default: `True`)
- `overwrite` (default: `False`)
- `zarr_format` (default: `3`, accepts `2` or `3`)

### Zarr v3 Migration Notes

- `zarr_format=3` is supported by the measurement API.
- In environments still running `zarr<3`, xmsr raises a clear error unless
  `ZARR_V3_EXPERIMENTAL_API=1` is set.
- Recommended migration flow:
  1. Run with default `zarr_format=3` in CI/dev.
  2. Keep `zarr_format=2` available for temporary fallback while validating old archives.
  3. Remove v2 fallback when no longer needed.

## Running Tests

```bash
uv run python -m pytest tests
npm ci
npm test
npm run typecheck
```

## Non-Blocking Execution

```python
measurement = BasicMeasurement()
measurement.start()

measurement.pause()
measurement.resume()  # Also starts acquisition if awaiting Start

measurement.finished.wait()
result = measurement.result
```

## Notebook Runner (VS Code)

Display a measurement in a VS Code notebook to start its background worker and
show the React runner panel:

```python
measurement = BasicMeasurement()
measurement  # Last expression in a notebook cell
```

Use **Start**, **Pause**, **Resume**, and **Cancel** in the panel. Python owns the
execution state; the panel displays synchronized snapshots and sends explicit
commands. **Pausing…** means the current step is still finishing; navigation is
enabled once it reaches a quiescent boundary. Cancellation wakes a worker
waiting for start or resume, but cannot
interrupt an instrument call already in progress. The panel retains the latest
500 log records, provides filename/path clipboard buttons, and adapts to VS Code
theme colors. Python `print()` calls made inside `measure()` (including instrument
library calls on the same thread) appear in the panel's log box. Standard output
uses info level; standard error uses error level. Lines appear on newline or
`flush=True`, and trailing partial lines are flushed when `measure()` exits.
Output from unrelated threads, subprocesses, native code writing directly to file
descriptors, and explicit file destinations such as `print(..., file=my_file)` is not captured.
Headless measurements keep their normal console output. Existing plots and the final rich xarray result are displayed
alongside the panel.

For scripts and headless tests, disable notebook UI explicitly:

```python
measurement = BasicMeasurement(notebook_ui=False)
measurement.run()  # Blocking execution on the main thread
```

Progress and timing are calculated in Python without a widget or notebook
progress bar. Elapsed time includes pauses; timing/rate restart at each retry or
rewind and count only newly completed steps in that continuation.

### Reopening Saved Runs

Recreate the same measurement with the same filename options and
`overwrite=False`. The existing filename schema selects the archive:

```python
measurement = YourMeasurement(
    filename="my-run",
    target_directory="data",
    timestamp=False,
    with_coords=False,
    overwrite=False,
)
measurement
```

If `data/my-run.zarr` exists, the runner restores its committed progress,
metadata, and latest measurement. The panel waits for **Resume saved run**;
saved data can also be browsed or rewound before starting. A new session runs
its normal `prepare()` once. Archives with incompatible coordinates, variables,
shapes, dtypes, or invalid checkpoints are rejected rather than silently treated
as new runs. Existing archives using `__CURRENT_INDEX__` remain supported.

### Browsing and Rewinding

When paused or recovering from an error, **Browse / resume from here** provides
one control per sweep dimension: index-based sliders for numeric coordinates,
and select boxes for categorical coordinates. Exact coordinate values and units
are shown alongside the controls; derived coordinates are read-only.

Changing a selection displays that point's saved measurement in the step plot.
It does not acquire data, move instruments, or change the acquisition cursor.
**Show latest** returns to the latest committed point.

**Resume from selected point** persists a rewind, invalidating the selected point
and everything after it, then continues acquisition from that point. Earlier
points stay valid. The runner cannot jump ahead of the next acquisition point.
Raw old values remain in the Zarr arrays, but `result`, point previews, and live
plots mask the invalid tail until it is remeasured. Partial integer/boolean
results are promoted to represent missing values; their on-disk dtypes are
unchanged. Reopening after a rewind uses the newly saved cursor.
New archives chunk data per sweep point, so replacing an invalid point does not
rewrite a chunk containing earlier valid measurements.

The same operations are available programmatically on a quiescent runner:

```python
measurement.pause()
measurement.paused.wait()  # Acknowledge the in-flight step
measurement.select_point((2, 1))  # Inspect sweep indices, without changing progress
measurement.rewind((2, 1))        # Persist invalidation and stay paused
measurement.resume()             # Remeasure from those indices
# Or: measurement.rewind((2, 1), resume=True)
```

Linear point indices are also accepted. Rewind changes data validity and the
acquisition cursor, not arbitrary metadata or instrument state; `measure()`
continues to apply the requested coordinates as usual. Rewinding during a
`wait_for_manual_resume()` operator pause unwinds that old step before acquiring
the selected point.

### Error Recovery

Acquisition/storage errors in a background runner keep the session, widget,
logs, and instrument setup alive. **Retry failed point** retries the uncommitted
point; browsing and rewinding to earlier parameters are also available.
**Neither retry nor rewind calls `prepare()` again.** `finish()` runs on final
completion or cancellation, not on each recoverable error.

Setup/finalization errors are terminal. Blocking main-thread `run()` still raises
step errors, but its session remains recoverable: call `run()` again, or rewind
and continue, without repeating `prepare()`. If a notebook panel was displayed,
its recovery commands launch a worker using that existing setup. Plot failures are
logged separately after the data/checkpoint commit and do not roll back progress.

### Frontend Development

The UI lives in `frontend/RunnerPanel.tsx`. `frontend/widget.tsx` is the thin
anywidget adapter; `src/xmsr/runner_widget.py` implements the Python side of the
state/command contract. React components are tested with Vitest and React Testing
Library, and Python command dispatch/lifecycle behavior with pytest. Actual
VS Code/VSCodium/code-server browser integration testing is deferred.

Use Node.js 22.12+ (or 20.19+) and install the frontend dependencies:

```bash
npm ci
npm run build
```

For live widget development, set this **before importing xmsr** in the notebook:

```python
%env ANYWIDGET_HMR=1
```

Then run `npm run watch` in a terminal. Rebuilt assets are hot-reloaded into the
displayed widget. With production builds, restart the kernel after rebuilding
to load the new assets.

### Packaging

The compiled, self-contained JavaScript/CSS in `src/xmsr/static/` are committed
and included in Python distributions. Notebook users need no Node.js, dev server,
or CDN access. After frontend changes, regenerate the assets before packaging:

```bash
npm run typecheck
npm test
npm run build
uv run ruff check src tests
uv run python -m pytest tests
uv build
```
