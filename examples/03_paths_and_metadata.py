# %% [markdown]
"""
# Paths and Metadata Lifecycle

Shows directory and filepath control and metadata updates in
`prepare`, `measure`, and `finish`.
"""

# %%
from pathlib import Path
from time import sleep

import numpy as np
import xarray as xr

from xmsr import Measurement


class MetadataLifecycleMeasurement(Measurement):
    target_directory = str(Path("tmp/custom-target"))
    filename = "manual-filepath"
    timestamp = False
    with_coords = False
    overwrite = False

    # Some metadata may not make sense to be embedded in the data. For these use the
    # metadata dictionary, which will be available in the `attr` property of your measurement
    # result. You can set measurement wide initial values:
    metadata = {"operator": "demo", "setup": "lab-bench-1"}

    sweep_template = xr.Dataset().assign_coords(step=np.arange(6))

    result_template = xr.DataArray(
        np.empty((50,), dtype=np.float64),
        dims=("sample",),
        coords={"sample": np.arange(50)},
        name="trace",
    )

    # You can update or add to the metadata in the `prepare`, `measure`, and `finish` methods.
    # These are called once per measurement, before and after the sweep loop executes.
    def prepare(self, metadata):
        metadata["prepared"] = True
        metadata["measure_calls"] = 0
        metadata["low_temperature_indices"] = []

    # You can also note some metadata from the sweep loop. For example you can record indices
    # where something has happened, but you do not want to error out.
    def measure(self, values, indices, metadata):
        metadata["measure_calls"] = metadata["measure_calls"] + 1
        metadata["last_step"] = indices["step"]

        if values["step"] == 4:  # of course this would be calling some equipemnt method
            metadata["low_temperature_indices"].append(indices["step"])

        sleep(0.1)
        return np.linspace(values["step"], values["step"] + 1.0, 50)

    def finish(self, metadata):
        metadata["finished"] = True


measurement = MetadataLifecycleMeasurement()
measurement
# %%
print("Stored in:", measurement._path)
print("Metadata:", measurement.metadata)
measurement.result

# %%
