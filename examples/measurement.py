# %% [markdown]
"""
# Measurement

This tutorial presents the `Measurement` class for parametric scans defined in
native xarray templates.

You define two schemas:
- `sweep_template` - sweep dimensions and coordinates
- `result_template` - payload returned by each `measure` call

The framework iterates all sweep points, stores to Zarr, supports pause/resume,
and exposes the full labeled xarray result.
"""

# %%
from time import sleep
from typing import Any

import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

from xmsr import Measurement


class BasicMeasurement(Measurement):
    target_directory = "tmp"

    sweep_template = (
        xr.Dataset()
        .assign_coords(x=np.arange(15), y=np.array([12, 14]))
        .assign_coords(power=("x", np.arange(15) ** 2))
    )

    result_template = xr.DataArray(
        np.empty((100,), dtype=np.float64),
        dims=("time",),
        coords={"time": np.linspace(0, 10, 100)},
        name="noisy_wave",
    )

    def measure(self, values, indices, metadata):
        sleep(0.1)
        return np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1

    def plot_preview(self, data: DataArray | Dataset):
        if isinstance(data, Dataset):
            data = data["noisy_wave"]
        return data.mean(dim=["x", "y"]).hvplot()

    def plot_single_step(self, data: DataArray | Dataset):
        return data.hvplot.line()


bm = BasicMeasurement()
bm

# %% [markdown]
"""
## Non-blocking mode

`Measurement` inherits from a thread abstraction, so it can run in background.
"""

# %%
measurement2 = BasicMeasurement()
measurement2.start()
print("Measurement is running in a separate thread!")
sleep(0.2)
measurement2.running.clear()
print("Measurement is paused!")
sleep(1)
print("Resuming measurement...")
measurement2.running.set()
measurement2.finished.wait()
measurement2.result

# %% [markdown]
"""
## Configuring the measurement

Configuration attributes include:
- `metadata`
- `target_directory`
- `filename`
- `timestamp`
- `with_coords`
- `overwrite`
"""


class ConfiguredMeasurement(BasicMeasurement):
    result_template = xr.DataArray(
        np.empty((10, 10), dtype=np.float64),
        dims=("z", "t"),
        coords={"z": np.arange(10), "t": np.arange(10)},
        name="my_variable",
    )

    metadata = {"measurement": "test"}
    timestamp = False
    with_coords = False
    filename = "my-measurement"
    overwrite = True

    def prepare(self, metadata: dict[str, Any]):
        metadata.update({"prepared": True})

    def measure(self, values, indices, metadata):
        metadata.update({"measured": True})
        return np.random.randn(10, 10)

    def finish(self, metadata: dict[str, Any]):
        metadata.update({"finished": True})


measurement3 = ConfiguredMeasurement()
measurement3.run()
measurement3.result

# %% [markdown]
"""
## Multi-variable measurements

Return tuples from `measure` to auto-wrap into `result_template` dataset variables.
"""


class MultiMeasurement(Measurement):
    target_directory = "tmp"
    sweep_template = xr.Dataset().assign_coords(
        x=np.arange(5),
        y=np.arange(10),
    )

    result_template = xr.Dataset(
        {
            "var1": xr.DataArray(np.empty((10, 10)), dims=("row", "col")),
            "var2": xr.DataArray(
                np.empty((10,)), dims=("c",), coords={"c": np.arange(10)}
            ),
            "var3": xr.DataArray(np.empty(()), dims=()),
        }
    )

    def measure(self, values, indices, metadata):
        sleep(0.05)
        return np.random.randint(10, size=(10, 10)), np.random.randn(10), np.array(0.0)


multi_measurement = MultiMeasurement()
multi_measurement.run()
multi_measurement.result
# %%
