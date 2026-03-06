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
        frequency = self.result_template.coords["frequency"].values
        gate = float(values["gate"])
        bias = float(values["bias"])
        center = 5500 + 600 * gate
        width = 400 + 30 * int(indices["bias"])
        return np.exp(-((frequency - center) ** 2) / (2 * width**2)) + 0.1 * bias

    def plot_preview(self, data: DataArray | Dataset):
        if isinstance(data, Dataset):
            data = data["spectrum"]
        return data.mean(dim=["gate", "bias"]).hvplot.line()

    def plot_single_step(self, data: DataArray | Dataset):
        if isinstance(data, Dataset):
            data = data["spectrum"]
        return data.hvplot.line()


measurement = LivePlotMeasurement()
measurement
