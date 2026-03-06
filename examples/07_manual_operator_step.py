# %% [markdown]
"""
# Manual Operator Step

Shows how to pause during a scan and wait for manual operator intervention.
"""

# %%
from pathlib import Path

import numpy as np
import xarray as xr

from xmsr import Measurement


class ManualStepMeasurement(Measurement):
    target_directory = str(Path("tmp/examples"))
    filename = "manual-step"
    timestamp = False
    with_coords = False
    overwrite = True

    sweep_template = xr.Dataset().assign_coords(
        stage=np.arange(6),
    )

    result_template = xr.DataArray(
        np.empty((80,), dtype=np.float64),
        dims=("sample",),
        coords={"sample": np.arange(80)},
        name="trace",
    )

    def measure(self, values, indices, metadata):
        if int(indices["stage"]) == 3:
            self.wait_for_manual_resume(
                "Manual step required: adjust the hardware stage, then resume the measurement."
            )
            metadata["manual_adjustment_done"] = True

        stage = float(values["stage"])
        return np.sin(np.linspace(0, 8, 80) + stage)


measurement = ManualStepMeasurement()
measurement
