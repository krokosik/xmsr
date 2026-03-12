# %% [markdown]
"""
# Multi Variable Measurement

Shows tuple autowrap into an `xr.Dataset` result template.
"""

# %%
from pathlib import Path
from time import sleep

import numpy as np
import xarray as xr

from xmsr import Measurement


class MultiVariableMeasurement(Measurement):
    target_directory = str(Path("tmp/examples"))
    filename = "multi-variable"
    timestamp = False
    with_coords = False
    overwrite = True

    sweep_template = xr.Dataset().assign_coords(
        temperature=np.array([4.0, 8.0, 12.0]),
        field=np.linspace(-1.5, 1.5, 7),
    )

    # You may measure more than one variable at once. In this case, you can return a tuple
    # of arrays and use a Dataset as template. Note that this makes
    # sense if they have heterogeneous metadata, otherwise you can just use a DataArray with more dimensions.
    result_template = xr.Dataset(
        # a dataset is essentially a dict of DataArrays
        {
            "voltage": xr.DataArray(
                np.empty((300,), dtype=np.float64),
                dims=("time",),
                coords={
                    "time": xr.DataArray(
                        np.linspace(0.0, 0.03, 300),
                        dims=("time",),
                        attrs={"units": "s"},
                    )
                },
                attrs={"units": "V"},
            ),
            "current": xr.DataArray(
                np.empty((300,), dtype=np.float64),
                dims=("time",),
                coords={
                    "time": xr.DataArray(
                        np.linspace(0.0, 0.03, 300),
                        dims=("time",),
                        attrs={"units": "s"},
                    )
                },
                attrs={"units": "A"},
            ),
            "resistance": xr.DataArray(attrs={"units": "Ohm"}),
        }
    )

    def measure(self, values, indices, metadata):
        index_mod = 1 + 0.05 * indices["field"]

        voltage = (
            2e-3
            * np.sin(2 * np.pi * 120.0 * self.coords.time + values["field"])
            * index_mod
        )
        current = (1e-3 + 2e-4 * values["temperature"]) * np.cos(
            2 * np.pi * 120.0 * self.coords.time
        )
        resistance = float(np.mean(voltage) / (np.mean(current) + 1e-12))
        sleep(0.1)
        return voltage, current, resistance


measurement = MultiVariableMeasurement()
measurement
