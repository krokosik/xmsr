# %% [markdown]
"""
# Measurement

This tutorial presents the `Measurement` class, which is the base class for all
classes that perform parametric measurements. Among its capabilities are:
- automatic scanning of all parameters
- automatic saving and chunking of data to Zarr archives with metadata
- possiblity to run a non blocking measurement in a thread
- pausing and aborting the measurement in non-blocking mode
- resuming measurements from an existing archive
- labelleing of data with dimensions and coordinates, handled by xarray (see xarray tutorial)
- ability to run in GUI mode!

## Basic usage

Import the `Measurement` class and create a subclass. The subclass must implement
the `measure` method, which takes 3 dictionaries as arguments:
- `indices` - a dictionary of indices of the parameters to be measured
- `values` - a dictionary of values of the parameters to be measured
- `metadata` - a dictionary of metadata, which can be mutated to add some information

The `measure` method should return a numpy array with the measured data. You also
have to specify the `param_coords` dictionary, which specifies the parameters
and their respective values.

Calling `run` calls the `measure` method for all combinations of parameters in blocking mode.
The `result` attribute fetches the xarray DataArray with the measured data.
"""

# %%
from typing import Any

from xarray import DataArray
from xmsr import Measurement, VariableData
from xmsr.runner import Runner
from time import sleep
import numpy as np


from xmsr.notebook_integration import notebook_extension

notebook_extension()


class BasicMeasurement(Measurement):
    target_directory = "tmp"
    param_coords = dict(
        x=list(range(5)), y=[(12, 13), (14, 15)]
    )  # param coordinates can be 2D
    variables = [VariableData("random-data", ["a", "b"], {"a": range(10), "b": range(10)})]

    def measure(self, values, indices, metadata):
        sleep(0.1)
        return np.random.randint(10, size=(10, 10))
    
    def plot_preview(self, measurement_da: DataArray):
        return measurement_da.mean(dim=["x", "y"]).hvplot.heatmap(
            x="a", y="b", title="Preview of random-data"
        )
    
    


measurement1 = BasicMeasurement()
runner = Runner(measurement1)
runner.live_info()
# %% [markdown]
"""
## Non-blocking mode

The `Measurement` class is a subclass of `QThread` so it can be run in a separate thread.
This is useful when the measurement takes a long time and you want to be able to pause it
or abort it, as well as when you want to have the IPython console available during the measurement.
"""
# %%
measurement2 = BasicMeasurement()
measurement2.start()
print("Measurement is running in a separate thread!")
sleep(0.2)
measurement2.running.clear()
print("Measurement is paused!")
sleep(5)
print("Resuming measurement...")
sleep(0.1)
measurement2.running.set()
measurement2.finished.wait()
measurement2.result
# %% [markdown]
"""
## Configuring the measurement

The `Measurement` class has a number of attributes that can be configured to change its behaviour.
- `data_dims` - a list of dimensions of the measured data
- `data_coords` - a dictionary of coordinates of the measured data
- `metadata` - a dictionary of metadata that will be saved to the archive
- `target_directory` - the directory where the archive will be saved
- `filename` - the name of the archive
- `timestamp` - whether to add a timestamp to the filename
- `with_coords` - whether to add coordinates to the filename
- `overwrite` - whether to overwrite the archive if it already exists

Additionally, you may overwrite the `prepare` and `finish` methods to perform some actions
before and after the measurement, respectively.

You can also subclass existing measurements if you want to reuse some of their functionality,
for example `measure` methods.
"""
# %%


class ConfiguredMeasurement(BasicMeasurement):
    variables = [
        VariableData("my-variable", ["z", "t"], {"t": range(10), "z": range(10)})
    ]
    # data_dims = ["z", "t"]
    # data_coords = dict(z=list(range(10)), t=list(range(10)))
    metadata = dict(measurement="test")
    timestamp = False
    with_coords = False
    filename = "my-measurement"
    overwrite = True

    def prepare(self, metadata: dict[str, Any]):
        self.metadata.update({"prepared": True})

    def measure(self, values, indices, metadata):
        metadata.update({"measured": True})
        return super().measure(values, indices, metadata)

    def finish(self, metadata: dict[str, Any]):
        self.metadata.update({"finished": True})


measurement3 = ConfiguredMeasurement()
measurement3.run()
measurement3.result

# %% [markdown]
"""
Class attributes are not the only way to specify configurations. They serve as defaults, but
you can also specify them in the constructor, as shown below. We also set `overwrite` to `False`
to demonstrate that the measurement will resume from the existing archive. Note that there
is no timestamp so the filename stays the same.
"""
# %%
measurement4 = ConfiguredMeasurement()
measurement4.start()
sleep(0.5)
measurement4.finished.set()
print("Measurement stopped!")
sleep(1)
print("Creating new measurement...")
measurement4 = ConfiguredMeasurement(overwrite=False)
measurement4.start()
measurement4.finished.wait()
measurement4.result

# %% [markdown]
"""
Multi variable measurements are also supported. Just return a tuple of numpy arrays
from the `measure` method and define the `variables` attribute as a list of `Variable`
instances.
"""


class MultiMeasurement(Measurement):
    target_directory = "tmp"
    param_coords = dict(
        x=list(range(5)), y=[(12, 13), (14, 15)]
    )  # param coordinates can be 2D
    variables = [
        VariableData("var1"),  # coordinates are optional
        VariableData("var2", ["c"], {"c": range(10)}),
        VariableData("var3", [], {}),
    ]

    def measure(self, values, indices, metadata):
        sleep(0.1)
        return np.random.randint(10, size=(10, 10)), np.random.randn(10), np.array(0)


multi_measurement = MultiMeasurement()
multi_measurement.run()
multi_measurement.result
# %%
