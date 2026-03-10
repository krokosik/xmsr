# %% [markdown]
"""
# Multi-Parameter Coordinates (MultiIndex)

Shows composite sweep coordinates represented with a MultiIndex.
"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from xmsr import Measurement

y_pairs = pd.MultiIndex.from_tuples(
    [(12, 13), (14, 15), (16, 17)],
    names=["y_a", "y_b"],
)


class MultiIndexSweepMeasurement(Measurement):
    target_directory = str(Path("tmp/examples"))
    filename = "multiparam-multiindex"
    timestamp = False
    with_coords = False
    overwrite = True

    sweep_template = xr.Dataset().assign_coords(
        x=np.arange(4),
        y_pair=("y_pair", y_pairs),
    )

    result_template = xr.DataArray(
        np.empty((40,), dtype=np.float64),
        dims=("sample",),
        coords={"sample": np.arange(40)},
        name="signal",
    )

    def measure(self, values, indices, metadata):
        pair_index = int(indices["y_pair"])
        pair = y_pairs[pair_index]
        slope = float(pair[0] + pair[1]) * 0.01
        intercept = float(values["x"]) * 0.1
        return slope * np.arange(40) + intercept


measurement = MultiIndexSweepMeasurement()
measurement
