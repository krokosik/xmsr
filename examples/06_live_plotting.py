# %% [markdown]
"""
# Live Plotting with hvplot

Shows how to implement `plot_preview` and `plot_single_step`.
"""

# %%
from pathlib import Path

import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

from xmsr import Measurement


class LivePlotMeasurement(Measurement):
    target_directory = str(Path("tmp/examples"))
    filename = "live-plot"
    timestamp = False
    with_coords = False
    overwrite = True

    sweep_template = xr.Dataset().assign_coords(
        gate=np.linspace(-1.0, 1.0, 9),
        bias=np.linspace(-0.3, 0.3, 7),
    )

    result_template = xr.DataArray(
        np.empty((120,), dtype=np.float64),
        dims=("frequency",),
        coords={"frequency": np.linspace(1e3, 1e4, 120)},
        name="spectrum",
    )

    def measure(self, values, indices, metadata):
        center = 5500 + 600 * values["gate"]
        width = 400 + 30 * indices["bias"]
        return (
            np.exp(-((self.coords.frequency - center) ** 2) / (2 * width**2))
            + 0.1 * values["bias"]
        )

    # you can live plot a preview of the already collected data using hvplot
    def plot_preview(self, data: DataArray | Dataset):
        return data.mean(dim=["gate", "bias"]).hvplot.line(ylim=(-0.2, 1.2))

    # or a single step
    def plot_single_step(self, data: DataArray | Dataset):
        return data.hvplot.line(ylim=(-0.2, 1.2))


measurement = LivePlotMeasurement()
measurement

# %%
