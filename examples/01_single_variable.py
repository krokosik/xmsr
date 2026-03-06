# %% [markdown]
"""
# Single Variable Measurement

Shows a single-variable measurement using `xr.DataArray` as `result_template`.
"""

# %%
from pathlib import Path

import numpy as np
import xarray as xr

from xmsr import Measurement


class SingleVariableMeasurement(Measurement):
    target_directory = str(Path("tmp/examples"))
    filename = "single-variable"
    timestamp = False
    with_coords = False
    overwrite = True

    sweep_template = xr.Dataset().assign_coords(
        gate=np.linspace(-1.0, 1.0, 7),
        bias=np.linspace(-0.2, 0.2, 5),
    )

    result_template = xr.DataArray(
        np.empty((200,), dtype=np.float32),
        dims=("frequency",),
        coords={
            "frequency": xr.DataArray(
                np.linspace(1e6, 5e6, 200),
                dims=("frequency",),
                attrs={"units": "Hz"},
            )
        },
        attrs={"units": "V"},
        name="amplitude",
    )

    def measure(self, values, indices, metadata):
        frequency = self.result_template.coords["frequency"].values
        phase = float(values["gate"]) * np.pi + 0.2 * int(indices["bias"])
        envelope = 1.0 + 0.1 * float(values["bias"])
        return (
            envelope * np.sin(2 * np.pi * frequency / frequency.max() + phase)
        ).astype(np.float32)


measurement = SingleVariableMeasurement()
measurement.run()
measurement.result
