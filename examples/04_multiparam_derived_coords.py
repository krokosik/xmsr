# %% [markdown]
"""
# Multi-Parameter Coordinates

Shows multiparam behavior via scalar base coordinate plus derived coordinates.
"""

# %%
from pathlib import Path
from typing import cast

import numpy as np
import xarray as xr

from xmsr import Measurement


class DerivedCoordsSweepMeasurement(Measurement):
    target_directory = str(Path("tmp/examples"))
    filename = "multiparam-derived"
    timestamp = False
    with_coords = False
    overwrite = True

    # want to sweep over two parameters? No problem! You can assign multiple derived coordinates
    # to a single dimension by passing a tuple with the relevant dimension name(s) to the `assign_coords` method.
    # The base coordinate is not required, but recommended.
    base_x = np.linspace(0.0, 2.0, 6)
    sweep_template = xr.Dataset().assign_coords(
        gain=("x", 2.0 * base_x + 1.0),
        offset=("x", np.sin(base_x)),
    )

    result_template = xr.DataArray(
        np.empty((60,), dtype=np.float64),
        dims=("sample",),
        coords={"sample": np.arange(60)},
        name="response",
    )

    def measure(self, values, indices, metadata):
        sample = np.arange(60)
        return values["gain"] * np.exp(-sample / 40.0) + values["offset"]


measurement = DerivedCoordsSweepMeasurement()
measurement

# %%
