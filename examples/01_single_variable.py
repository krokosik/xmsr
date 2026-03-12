# %% [markdown]
"""
# Single Variable Measurement

Shows a single-variable measurement using `xr.DataArray` as `result_template`.
"""

# %%
from pathlib import Path
from time import sleep

import numpy as np
import xarray as xr

from xmsr import Measurement


class SingleVariableMeasurement(Measurement):
    target_directory = str(
        Path("tmp/examples")
    )  # all measurement files go into this directory
    filename = "single-variable"  # base filename for results
    timestamp = False  # append timestamp to filename?
    with_coords = False  # include coordinates in filename?
    overwrite = True  # overwrite or append to existing file?

    # define the sweep parameters and their values. Their cartesian product defines the sweep points
    # that will be looped over and passed to the meassure method.
    # You can think of Datasets as bags of variables, they are the Xarray equivalent of pandas DataFrames.
    sweep_template = xr.Dataset().assign_coords(
        gate=np.linspace(-1.0, 1.0, 7),  # you can use ndarrays or array likes
        bias=xr.DataArray(
            np.linspace(-0.5, 0.5, 5),
            dims="bias",
            attrs={"units": "V"},
        ),  # or DataArrays with dims and even units!
    )

    # You need to define the shape and coordinates of a single measurement result via a template.
    # This is basically what your results will be packed with before being assembled into a final
    # result. If you measure more than one variable, you can return a tuple of arrays and use a Dataset as template.
    # The idea is that you experiment with data collection interactively first, and once you
    # settle on a measurement format, you type in the metadata here and run the sweep.
    result_template = xr.DataArray(
        np.empty((200,), dtype=np.float32),
        dims=("frequency",),
        coords={
            # of course this can also be just an array like, but extra
            # metadata help you in the long run.
            "frequency": xr.DataArray(
                np.linspace(1e6, 5e6, 200),
                dims=("frequency",),
                attrs={"units": "Hz"},
            )
        },
        attrs={"units": "V"},  # the units are optional
        name="amplitude",  # and so is the name, but it is very useful for plotting and necessary for hvplot
    )

    def measure(self, values, indices, metadata):
        phase = values["gate"] * np.pi + 0.2 * indices["bias"]
        envelope = 1.0 + 0.1 * values["bias"]
        sleep(0.1)
        return (
            envelope
            * np.sin(
                2
                * np.pi
                # all the coords metadata is stored in a dataset
                # if you need it. However, normally you should only use
                # SWEEP values and indices, which are passed as arguments here.
                # note that this is a fake example, ideally your
                # RESULT data should come from the measurement.
                * self.coords.frequency
                / self.coords.frequency.max()
                + phase
            )
        ).astype(np.float32)


measurement = SingleVariableMeasurement()
measurement
