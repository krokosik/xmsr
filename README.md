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
commands. Cancellation wakes a worker waiting for start or resume, but cannot
interrupt an instrument call already in progress. The panel retains the latest
500 log records, provides filename/path clipboard buttons, and adapts to VS Code
theme colors. Existing plots and the final rich xarray result are displayed
alongside the panel.

For scripts and headless tests, disable notebook UI explicitly:

```python
measurement = BasicMeasurement(notebook_ui=False)
measurement.run()  # Blocking execution on the main thread
```

Progress and timing are calculated in Python without a widget or notebook
progress bar. Elapsed time includes pauses; the rate is the average for newly
completed steps in this invocation, including when resuming an existing archive.

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
