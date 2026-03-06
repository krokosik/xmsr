# %% [markdown]
"""
# Multi-Parameter Coordinates (Derived Coordinates)

Shows multiparam behavior via scalar base coordinate plus derived coordinates.
"""

# %%
from pathlib import Path

import numpy as np
import xarray as xr

from xmsr import Measurement


class DerivedCoordsSweepMeasurement(Measurement):
    target_directory = str(Path("tmp/examples"))
    filename = "multiparam-derived"
    timestamp = False
    with_coords = False
    overwrite = True

    base_x = np.linspace(0.0, 2.0, 6)
    sweep_template = (
        xr.Dataset()
        .assign_coords(x=base_x)
        .assign_coords(
            x_gain=("x", 2.0 * base_x + 1.0),
            x_offset=("x", np.sin(base_x)),
        )
    )

    result_template = xr.DataArray(
        np.empty((60,), dtype=np.float64),
        dims=("sample",),
        coords={"sample": np.arange(60)},
        name="response",
        attrs={"units": "arb"},
    )

    def measure(self, values, indices, metadata):
        x_index = int(indices["x"])
        gain = float(self._sweep.coords["x_gain"].values[x_index])
        offset = float(self._sweep.coords["x_offset"].values[x_index])
        sample = np.arange(60)
        return gain * np.exp(-sample / 40.0) + offset


measurement = DerivedCoordsSweepMeasurement()
measurement.run()
measurement.result
