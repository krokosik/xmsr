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
- IPython widget integration for status/progress and optional live plotting

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

For composite coordinates, prefer MultiIndex or derived coordinates over tuple/object arrays.

```python
import pandas as pd
import xarray as xr

y_pairs = pd.MultiIndex.from_tuples([(12, 13), (14, 15)], names=["y_a", "y_b"])

sweep_template = xr.Dataset().assign_coords(
    x=[0, 1, 2],
    y_pair=("y_pair", y_pairs),
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
uv run python -m pytest tests/test_measurement_templates.py
```

## Non-Blocking Execution

```python
measurement = BasicMeasurement()
measurement.start()

measurement.running.clear()  # pause
measurement.running.set()    # resume

measurement.finished.wait()
result = measurement.result
```
