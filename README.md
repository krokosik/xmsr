# xmsr - Xarray Measurement Wrapper

> **WARNING**: the package is under development and not yet ready for full use.

A convenient Python class for running parametric measurements and storing data using a chunked and compressed binary format (Zarr). This wrapper combines high-level APIs for convenient parameter sweep measurements with minimal verbosity and maximum reusability.

## Overview

The `Measurement` class is designed to simplify the process of running multi-dimensional parameter sweeps in experimental physics and other scientific domains. Built on top of **xarray**, it leverages xarray's fantastic declarative syntax and semantics to embed experiment information directly into the data, creating self-documenting datasets with labeled dimensions, coordinates, and metadata.

Key capabilities:

- **Automatic parameter scanning** - Define parameter ranges and let the class handle the sweep
- **Efficient data storage** - Uses Zarr for chunked, compressed binary storage
- **Non-blocking execution** - Run measurements in a separate thread with pause/resume capabilities
- **Declarative data labeling** - Leverage xarray's powerful semantics to embed dimensions, coordinates, and metadata directly into your datasets
- **Resume capability** - Continue interrupted measurements from existing archives
- **(Planned) GUI integration** - Built-in measurement runner with progress tracking (IPython compatible)

## Design Goals

- **Minimize API verbosity** - Simple, intuitive interface for common measurement tasks
- **Maximize reusability** - Write measurement scripts once, reuse with different parameters
- **Integrate well with Interactive Python (IPython)** - Designed for notebook workflows and interactive exploration
- **Embed experiment info in data** - Use xarray's declarative approach to create self-documenting datasets where parameters, dimensions, and metadata are integral parts of the data structure

## Features

### Parameter Sweeps

Define multi-dimensional parameter sweeps with simple dictionary syntax:

```python
from xmsr import Measurement
import numpy as np

class BasicMeasurement(Measurement):
    target_directory = "data"
    param_coords = dict(
        x=list(range(5)), 
        y=[(12, 13), (14, 15)]
    )
    
    def measure(self, values, indices, metadata):
        # Your measurement code here
        return np.random.randint(10, size=(10, 10))

measurement = BasicMeasurement()
measurement.run()
result = measurement.result  # Returns xarray DataArray
```

### Non-Blocking Execution

Run measurements in a separate thread with full control:

```python
measurement = BasicMeasurement()
measurement.start()  # Runs in background

# Pause the measurement
measurement.running.clear()

# Resume the measurement
measurement.running.set()

# Wait for completion
measurement.finished.wait()
result = measurement.result
```

### Multi-Variable Measurements

Measure multiple quantities simultaneously:

```python
from xmsr import Measurement, VariableData

class MultiMeasurement(Measurement):
    target_directory = "data"
    param_coords = dict(x=range(5), y=range(10))
    
    variables = [
        VariableData("voltage", ["time"], {"time": range(100)}),
        VariableData("current", ["time"], {"time": range(100)}),
        VariableData("resistance"),  # Scalar variable
    ]
    
    def measure(self, values, indices, metadata):
        voltage = np.random.randn(100)
        current = np.random.randn(100)
        resistance = voltage.mean() / current.mean()
        return voltage, current, resistance
```

### Configuration Options

Customize measurement behavior via class attributes or constructor arguments:

- `data_dims` - Dimensions of measured data
- `data_coords` - Coordinates for measured data
- `metadata` - Additional metadata saved with the archive
- `target_directory` - Output directory for Zarr archives
- `filename` - Custom archive filename
- `timestamp` - Add timestamp to filename (default: True)
- `with_coords` - Include coordinates in filename (default: True)
- `overwrite` - Overwrite existing archives (default: False)

```python
class ConfiguredMeasurement(BasicMeasurement):
    metadata = dict(experiment="demo", sample="A1")
    timestamp = False
    with_coords = False
    filename = "my-measurement"
    overwrite = True
    
    def prepare(self, metadata):
        # Called before measurement starts
        metadata.update({"prepared": True})
    
    def finish(self, metadata):
        # Called after measurement completes
        metadata.update({"finished": True})
```

### Resume Measurements

Continue interrupted measurements from existing archives:

```python
# Start a measurement
measurement = ConfiguredMeasurement()
measurement.start()
# ... measurement gets interrupted ...
measurement.finished.set()  # Stop it

# Resume from the same archive
measurement = ConfiguredMeasurement(overwrite=False)
measurement.start()
measurement.finished.wait()
```

## Installation

```bash
pip install xmsr
```

## Current Limitations

The class currently supports:
- Blocking execution via `run()`
- Simple non-blocking execution via `start()`

**Note:** Non-blocking mode may have issues when other threads are used from within the measurement code.

## Roadmap

Planned features for future releases:

- **Enhanced GUI** 
  - Progress bar with time estimates
  - Live preview plot of current chunk
  - Partial plot of the entire sweep
  - Control buttons (pause, resume, abort)
  
- **Robust Threading Support**
  - Better handling of nested threads
  - Thread-safe measurement execution
  
- **Parameter Ordering Optimization**
  - Smart ordering for multidimensional sweeps
  - Minimize jumps in slowly varying parameters
  - Optimize measurement efficiency for specific hardware constraints
