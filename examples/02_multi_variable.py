# %% [markdown]
"""
# Multi Variable Measurement

Shows tuple autowrap into an `xr.Dataset` result template.
"""

# %%
from pathlib import Path

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

    result_template = xr.Dataset(
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
            "resistance": xr.DataArray(
                np.empty((), dtype=np.float64),
                dims=(),
                attrs={"units": "Ohm"},
            ),
        }
    )

    def measure(self, values, indices, metadata):
        time = self.result_template["voltage"].coords["time"].values
        field = float(values["field"])
        temperature = float(values["temperature"])
        index_mod = 1 + 0.05 * int(indices["field"])

        voltage = 2e-3 * np.sin(2 * np.pi * 120.0 * time + field) * index_mod
        current = (1e-3 + 2e-4 * temperature) * np.cos(2 * np.pi * 120.0 * time)
        resistance = float(np.mean(voltage) / (np.mean(current) + 1e-12))
        return voltage, current, resistance


measurement = MultiVariableMeasurement()
measurement
