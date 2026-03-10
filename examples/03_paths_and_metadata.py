# %% [markdown]
"""
# Paths and Metadata Lifecycle

Shows directory and filepath control and metadata updates in
`prepare`, `measure`, and `finish`.
"""

# %%
from pathlib import Path

import numpy as np
import xarray as xr

from xmsr import Measurement


class MetadataLifecycleMeasurement(Measurement):
    target_directory = str(Path("tmp/custom-target"))
    filename = "manual-filepath"
    timestamp = False
    with_coords = False
    overwrite = False
    metadata = {"operator": "demo", "setup": "lab-bench-1"}

    sweep_template = xr.Dataset().assign_coords(step=np.arange(6))

    result_template = xr.DataArray(
        np.empty((50,), dtype=np.float64),
        dims=("sample",),
        coords={"sample": np.arange(50)},
        name="trace",
    )

    def prepare(self, metadata):
        metadata["prepared"] = True
        metadata["measure_calls"] = 0

    def measure(self, values, indices, metadata):
        metadata["measure_calls"] = int(metadata["measure_calls"]) + 1
        metadata["last_step"] = int(indices["step"])
        base = float(values["step"])
        return np.linspace(base, base + 1.0, 50)

    def finish(self, metadata):
        metadata["finished"] = True


measurement = MetadataLifecycleMeasurement()
measurement
# %%
print("Stored in:", measurement._path)
print("Metadata:", measurement.metadata)
measurement.result

# %%
