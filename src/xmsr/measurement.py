import itertools
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Event
from typing import Any, Optional, TypedDict, Union

import holoviews as hv
import numpy as np
import numpy.typing as npt
import xarray as xr
import zarr
from tqdm import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect

from xmsr.notebook_integration import live_info, live_plot, logging_redirect_ipywidgets
from xmsr.qt_integration import Thread
from xmsr.shared import MeasurementStatus

_CURRENT_INDEX_KEY = "__CURRENT_INDEX__"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class ProgressDict(TypedDict):
    n: int
    total: int
    percentage: float
    elapsed: float
    remaining: float
    rate: float
    unit: str
    unit_scale: bool
    unit_divisor: int
    ncols: Optional[int]
    nrows: Optional[int]
    indices: tuple[int, ...]


# ============================================================================
# CLASS DEFINITION & CONFIGURATION
# ============================================================================


class Measurement(Thread):
    """Abstract base class for measurements performing a parametric scan.

    In order to implement a measurement, subclass this class and implement the
    `measure` method. The `param_coords` class attribute must be defined as well.
    The data is saved to an xarray DataArray, which can be accessed through the
    `result` property once the measurement has been completed. In case of multiple
    return values, an xarray Dataset is created instead.

    The measurement can be run in 3 ways:
        - by getting the `gui` class attribute. This will open a GUI where you can
            start the measurement, and pause or stop it at any time. You can also
            plot the results of the measurement if the `plot_preview` method is
            implemented. Note that there are some threading issues, which break when calling
            spin1 methods such as collect_slowly from within the `measure` method.
        - by instantiating the class and calling the `run` method. This will run the
            measurement in the current thread.
        - by instantiating the class and calling the `start` method. This will run the
            measurement in a separate thread. You can emit the `pause` signal to pause
            the measurement, and the `stop` signal to stop it.

    You can also optionally override the `prepare` and `finish` methods to perform
    some actions before and after the measurement, respectively. Each class instance
    also takes some optional arguments that can be used to customize the measurement.
    You can override the defaults by setting the corresponding class attributes.

    See the relevant methods' docstrings for more information.

    Attributes:
        param_coords: A mapping of coordinate names to their values.
        data_dims: The names of the dimensions of the measurement results. If not
            defined, placeholders will be used.
        data_coords: A mapping of dimension names to their values. The values must be
            iterable, and the length of each iterable must be the same.
        metadata: Serializable data to be stored in the store attributes.
        target_directory: The directory where the measurement results will be stored.
        filename: The name of the file where the measurement results will be stored,
        timestamp: Whether the filename should have a timestamp suffix.
            defaults to the name of the class.
        with_coords: Whether the filename should have a suffix with the parameter
            coordinate names and min/max values.
        overwrite: Whether existing values should be overwritten.
    """

    _is_single_run = True

    param_coords: dict[str, Any]

    _param_coords: Mapping[str, Any]

    @dataclass
    class Var:
        name: str
        dims: Collection[str] = field(default_factory=tuple)
        coords: Mapping[str, Any] = field(default_factory=dict)

    variables: list[Var]

    data_dims: Iterable[str]  # deprecated, use VariableData.dims instead
    data_coords: Mapping[str, Any] = {}  # deprecated, use VariableData.coords instead

    _combinations: list[tuple[tuple[int, ...], tuple[Any, ...]]]
    _last_measurement_shapes: Optional[tuple[tuple[int, ...], ...]]
    last_measurement: xr.DataArray | xr.Dataset | None
    _current_index: int
    _path: Path
    _exception: Exception
    LOG: logging.Logger

    metadata: dict[str, Any] = {}
    filename: str
    timestamp: bool = True
    overwrite: bool = False
    with_coords: bool = True
    target_directory: str = os.getcwd()

    live_plot_opts: hv.opts

    def __init__(
        self,
        overwrite: Optional[bool] = None,
        filename: Optional[str] = None,
        timestamp: Optional[bool] = None,
        with_coords: Optional[bool] = None,
        metadata: Optional[dict[str, Any]] = None,
        target_directory: Optional[str] = None,
    ):
        super().__init__()
        # run params
        self.metadata = metadata if metadata is not None else self.metadata
        self.filename = filename if filename is not None else self.__class__.__name__
        self.timestamp = timestamp if timestamp is not None else self.timestamp
        self.overwrite = overwrite if overwrite is not None else self.overwrite
        self.with_coords = with_coords if with_coords is not None else self.with_coords
        self.target_directory = (
            target_directory if target_directory is not None else self.target_directory
        )

        # threading events and queues
        self.running = Event()
        self.progress_queue = Queue[ProgressDict]()
        self.revert_progress = Queue[tuple[int, ...]]()
        self.last_measurement = None
        self.last_measurement_queue = Queue[xr.DataArray | xr.Dataset]()
        self.finished = Event()
        self.cancelled = Event()
        self.started = Event()

        # logger
        self.LOG = logging.getLogger(self.__class__.__name__)
        self.LOG.setLevel(logging.INFO)

    # ============================================================================
    # SUBCLASS INITIALIZATION
    # ============================================================================

    def __init_subclass__(cls):
        """Validate and initialize subclass configuration.

        Ensures required attributes are defined and computes parameter combinations
        for the measurement sweep.
        """
        if not hasattr(cls, "param_coords"):
            raise ValueError("Measurement subclasses must define 'param_coords'")

        if getattr(cls, "data_dims", None) or getattr(cls, "data_coords", None):
            logging.warning(
                "'data_dims' and 'data_coords' are deprecated, please use 'variables' instead",
            )
            if not hasattr(cls, "variables"):
                cls.variables = [
                    cls.Var(
                        "result",
                        getattr(cls, "data_dims", ()),
                        getattr(cls, "data_coords", {}),
                    )
                ]
        elif not hasattr(cls, "variables"):
            logging.warning(
                "Measurement subclasses should define 'variables' to describe the output data",
            )
            cls.variables = [cls.Var("result", (), {})]

        setattr(
            cls,
            "_param_coords",
            {k: _prepare_coord(v) for k, v in cls.param_coords.items()},
        )

        cls._combinations = list(  # type: ignore
            tuple(zip(*combination))
            for combination in itertools.product(
                *(enumerate(coord) for coord in cls._param_coords.values())
            )
        )

    # ============================================================================
    # CORE USER API
    # ============================================================================

    def prepare(self, metadata: dict[str, Any]):
        """Perform some actions before the measurement starts.

        Args:
            metadata: Serializable data to be stored in the store
                attributes. You can mutate this dictionary to add more data.
        """
        self.LOG.debug("Preparing...")

    def measure(
        self, values: dict[str, Any], indices: dict[str, int], metadata: dict[str, Any]
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """Perform a single measurement.

        This method must be implemented by subclasses. It should perform a single
        measurement and return the result as numpy array. It is called once for each
        combination of parameter values and must always return an array of the same
        shape and dtype.

        Args:
            values: values of the currently used parameters
            indices: indices of the currently used parameters
            metadata: Serializable data to be stored in the store
                attributes. You can mutate this dictionary to add more data.

        Returns:
            np.ndarray | tuple[np.ndarray, ...]: The measurement result. The shape of the array must be the
                same for all measurements. Return a tuple of arrays if there are multiple values to be stored.
                In such a scenario, the resulting structure will be an Xarray Dataset instead of a DataArray.
        """
        raise NotImplementedError

    def finish(self, metadata: dict[str, Any]):
        """Perform some actions after the measurement finishes.

        Args:
            metadata: Serializable data to be stored in the store
                attributes. You can mutate this dictionary to add more data.
        """
        self.LOG.debug("Done")

    # ============================================================================
    # EXECUTION CONTROL
    # ============================================================================

    def _start_measurement(self):
        """Initialize measurement storage and state.

        Sets up the Zarr archive, checks for existing data to resume from,
        handles overwrite logic, and prepares for measurement execution.
        """
        self.prepare(self.metadata)

        self._last_measurement_shapes = None

        filename = self.filename + self._get_filename_suffix(
            self.timestamp, self.with_coords
        )

        self._path = (Path(self.target_directory) / filename).with_suffix(".zarr")
        self.LOG.debug(f"Results will be stored in:\n{filename}")
        self.LOG.debug(f"Full path:\n{os.path.abspath(self._path)}")

        try:
            self.current_index = zarr.open(str(self._path), mode="r").attrs[
                _CURRENT_INDEX_KEY
            ]
            self.LOG.debug(
                f"Found existing data with {self.current_index} measurements, resuming..."
            )
            current_data = self.result
            for i in range(self.current_index):
                indices, _ = self._combinations[i]
                self.last_measurement_queue.put_nowait(current_data[indices])
        except Exception:
            self.current_index = 0

        if self.overwrite and self.current_index > 0:
            self.LOG.debug(f"Overwriting {self.current_index} measurements...")
            self.current_index = 0
            shutil.rmtree(self._path)

        self.LOG.debug("Starting measurements...")

    def step(self, idx: int | None = None):
        """Execute a single measurement step.

        Args:
            idx: Index of the measurement step to execute. If None, uses current_index.

        Returns:
            tuple of numpy arrays containing the measurement results
        """
        if idx is None:
            idx = self.current_index

        indices, values = self._combinations[idx]

        self.LOG.debug(f"Measurement no. {idx} indices: {indices} values: {values}")

        result = self.measure(
            {k: v for k, v in zip(self.param_coords.keys(), values)},
            {k: i for k, i in zip(self.param_coords.keys(), indices)},
            self.metadata,
        )
        result = result if isinstance(result, tuple) else (result,)

        if self._last_measurement_shapes is None:
            self._last_measurement_shapes = tuple(r.shape for r in result)
        else:
            assert all(
                r.shape == ls for r, ls in zip(result, self._last_measurement_shapes)
            ), "All measurements must have the same shape"

        return result

    def run(self) -> None:
        """Start the measurement.

        The measurement results will be stored in an zarr store wrapped with xarray.
        The store will be created if it does not exist, and the measurement will be
        resumed if it already exists. The measurement can be paused by emitting the
        `toggle_pause` signal, and stopped by emitting the `stop` signal. The
        measurement can be resumed by emitting the `toggle_pause` signal again.
        """
        try:
            self._start_measurement()
            self._live_info()

            self._wait_on(self.started)

            if hasattr(self, "_ui_thread"):
                self._ui_thread.start()

            self.running.set()
            self.finished.clear()

            # self.progress_queue.put_nowait(self._get_progress_dict())

            while not self.finished.is_set():
                self._wait_on(self.running)
                if self._is_single_run and self.current_index >= self.ntotal:
                    self.LOG.debug("Single run completed.")
                    self.finished.set()

                if self.finished.is_set():
                    break

                result = self.step()

                self.store_ndarray(self.current_point[0], result)
                self.LOG.debug("Stored measurement chunk.")
                self.current_index += 1
                self._ui_update()
                self.LOG.debug("UI updated.")

                self.LOG.debug(f"Current index: {self.current_index}")

                if self.current_index >= self.ntotal:
                    self.finished.set()

                time.sleep(0)  # process events in blocking version

            if self.finished.is_set():
                self.running.clear()
                self._finalize_measurement()

        except Exception as e:
            self.LOG.exception(e)
            self.cancel()
            self._exception = e
            raise e

    def _finalize_measurement(self):
        """Clean up and finalize after measurement completion."""
        self.LOG.debug("Finalizing measurement...")
        self.finish(self.metadata)
        self._update_metadata()
        self._ui_finish()

    def _wait_on(self, event: Event):
        """Wait on an event, skipping if in main thread.

        Args:
            event: Threading.Event to wait on
        """
        if threading.current_thread() is not threading.main_thread():
            event.wait()

    # ============================================================================
    # STATE & PROGRESS TRACKING
    # ============================================================================

    @property
    def current_index(self) -> int:
        """Get the current measurement index."""
        return self._current_index

    @current_index.setter
    def current_index(self, index: Union[tuple[int, ...], int]):
        """Set the current measurement index.

        Accepts either an integer index or a tuple of indices and finds the
        corresponding linear index.

        Args:
            index: Integer index or tuple of indices
        """
        self.LOG.debug(f"Setting current index to {index}")
        if isinstance(index, int):
            self._current_index = index
        else:
            for i, (indices, _) in enumerate(self._combinations):
                if indices == index:
                    self._current_index = i
                    break

        if hasattr(self, "pbar"):
            self.progress_queue.put_nowait(self._get_progress_dict())

    @property
    def current_point(self) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        """Get the current (indices, values) tuple for the measurement."""
        return self._combinations[self.current_index]

    @property
    def ntotal(self) -> int:
        """Get the total number of measurement points."""
        return len(self._combinations)

    @property
    def status(self) -> MeasurementStatus:
        """Return the runner status as a string.

        The possible statuses are: running, cancelled, failed, and finished.

        Returns:
            MeasurementStatus: Current status of the measurement
        """
        if hasattr(self, "_exception"):
            return MeasurementStatus.FAILED
        elif not self.started.is_set():
            return MeasurementStatus.INIT
        elif self.cancelled.is_set():
            return MeasurementStatus.CANCELLED
        elif self.finished.is_set():
            return MeasurementStatus.FINISHED
        elif not self.running.is_set():
            return MeasurementStatus.PAUSED
        else:
            return MeasurementStatus.RUNNING

    def get_index_by_indices(self, indices: tuple[int, ...]) -> int:
        """Find the linear index for a given tuple of parameter indices.

        Args:
            indices: Tuple of parameter indices to search for

        Returns:
            Linear index matching the indices, or -1 if not found
        """
        for i, (combination_indices, _) in enumerate(self._combinations):
            if combination_indices == indices:
                return i

        self.LOG.error(f"No combination with indices {indices} found")
        return -1

    def pause_unpause(self) -> None:
        """Pause or unpause the runner."""
        if self.running.is_set():
            self.running.clear()
        else:
            self.running.set()
            if not self.started.is_set():
                self.started.set()
        self._ui_update()

    def cancel(self) -> None:
        """Cancel the runner."""
        self.cancelled.set()
        self.finished.set()
        self.running.clear()

    def _get_progress_dict(self) -> ProgressDict:
        """Build a progress dictionary for UI updates.

        Returns:
            ProgressDict containing current progress information
        """
        return {
            **self.pbar.format_dict,
            "indices": self._combinations[
                min(self.current_index, len(self._combinations) - 1)
            ][0],
        }  # type: ignore

    def _revert_progress(self, indices: Union[tuple[int, ...], int]):
        """Revert progress to a previous index (currently unused)."""
        previous_index = getattr(self, "_current_index", 0)
        self.current_index = indices
        self.pbar.unpause()
        self.pbar.update(self.current_index - previous_index)
        self.progress_queue.put_nowait(self._get_progress_dict())

    # ============================================================================
    # DATA STORAGE
    # ============================================================================

    def store_ndarray(
        self, indices: tuple[int, ...], data: tuple[np.ndarray, ...]
    ) -> None:
        """Store measurement results to Zarr archive.

        Creates the dataset on first call, then appends to existing archive.
        Handles both single-variable (DataArray) and multi-variable (Dataset) cases.

        Args:
            indices: Tuple of parameter indices for this measurement
            data: Tuple of numpy arrays to store, one per variable
        """
        assert len(data) == len(self.variables), (
            "Length of variables must match number of returned arrays"
        )

        for var, datum in zip(self.variables, data):
            if len(var.dims) != datum.ndim:
                var.dims = self._default_data_dims(datum)

        if not self._path.exists():
            ds = xr.Dataset(
                {
                    var.name: xr.DataArray(
                        np.empty(
                            (
                                *map(
                                    lambda coord: len(np.array(coord)),
                                    self._param_coords.values(),
                                ),
                                *datum.shape,
                            ),
                            dtype=datum.dtype,
                        )
                        * np.nan,
                        dims=tuple(self._param_coords.keys()) + tuple(var.dims),
                        coords={**self._param_coords, **(var.coords or {})},
                    )
                    for var, datum in zip(self.variables, data)
                }
            )
            self._maybe_da(ds).assign_attrs(**self.metadata).to_zarr(
                self._path,  # type: ignore
                mode="w",
            )

        ds = xr.Dataset(
            {
                var.name: xr.DataArray(
                    datum,
                    dims=var.dims,
                    coords=var.coords or {},
                )
                for var, datum in zip(self.variables, data)
            }
        )

        self._maybe_da(ds).expand_dims(dim=list(self._param_coords.keys())).drop_vars(
            sum([list(var.coords.keys()) for var in self.variables], [])
        ).to_zarr(
            self._path,  # type: ignore
            region={
                dim: slice(index, index + 1)
                for dim, index in zip(self._param_coords.keys(), indices)
            },
        )

        self.last_measurement = ds
        self.last_measurement_queue.put_nowait(self.last_measurement)

        if hasattr(self, "update_plots"):
            self.update_plots()

        self._update_metadata()

        # Zarr attributes are for some reason separate from xarray attributes
        # so we hide the data not used in the result
        zarr.open(str(self._path), mode="a").attrs[_CURRENT_INDEX_KEY] = (
            self.current_index + 1
        )

    def _maybe_da(self, data: xr.Dataset) -> xr.DataArray | xr.Dataset:
        """Convert Dataset to DataArray if only one variable is present.

        Args:
            data: xarray Dataset to potentially convert
        Returns:
            xarray DataArray if only one variable, otherwise original Dataset
        """
        if len(self.variables) == 1:
            return data[self.variables[0].name]
        return data

    def get_measurement_by_indices(
        self, indices: tuple[int, ...]
    ) -> xr.Dataset | xr.DataArray:
        """Retrieve a single measurement from the archive.

        Args:
            indices: Tuple of parameter indices to retrieve

        Returns:
            xarray DataArray or Dataset containing the requested measurement
        """
        if not hasattr(self, "_path"):
            self.LOG.error("Measurement not started yet")
            return xr.Dataset()

        return (xr.open_dataset if len(self.variables) > 1 else xr.open_dataarray)(
            self._path, engine="zarr"
        ).isel(dict(zip(self.param_coords.keys(), indices)))

    def _default_data_dims(self, data: np.ndarray) -> tuple[str, ...]:
        """Generate default dimension names for an array.

        Args:
            data: Numpy array to generate dimensions for

        Returns:
            Tuple of dimension names like ('dim0', 'dim1', ...)
        """
        return tuple((f"dim{i}" for i in range(data.ndim)))

    def _update_metadata(self):
        """Update metadata in the Zarr store.

        Note: This is a workaround for xarray issue #8116. Metadata updates
        need to be manually written to the Zarr .zmetadata file.
        """
        # ! THIS IS A HACK. Fix this once this issue is resolved: https://github.com/pydata/xarray/issues/8116
        if len(self.variables) != 1:
            return
        z = zarr.open_group(self._path, mode="a")
        z = z[self.variables[0].name]
        z.attrs.update(self.metadata)

        with open(self._path / ".zmetadata", "r") as f:
            data = json.load(f)
            data["metadata"][self.variables[0].name + "/.zattrs"].update(self.metadata)

        with open(self._path / ".zmetadata", "w") as f:
            json.dump(data, f)

    @property
    def result(self) -> xr.DataArray | xr.Dataset:
        """The measurement results.

        Call once the measurement has been completed. The results are stored in an
        xarray DataArray or Dataset, which can be accessed through this property.
        The latter structure is returned if there are multiple variables returned by the
        measurement.

        Returns:
            xarray DataArray (single variable) or Dataset (multiple variables)
        """
        ds = xr.open_dataset(self._path, engine="zarr")
        if len(self.variables) == 1:
            return ds[self.variables[0].name]
        return ds

    # ============================================================================
    # VISUALIZATION
    # ============================================================================

    def plot_result(self, *args, **kwargs):
        """Plot the complete measurement results.

        Uses hvplot if available, otherwise falls back to matplotlib plot.

        Args:
            *args: Positional arguments passed to plotting method
            **kwargs: Keyword arguments passed to plotting method

        Returns:
            Plot object
        """
        if hasattr(self.result, "hvplot"):
            return self.result.hvplot(*args, **kwargs)
        else:
            return self.result.plot(*args, **kwargs)  # type: ignore

    def plot_single_step(
        self,
        data: xr.DataArray | xr.Dataset,
    ):
        """Plot a single measurement step.

        Args:
            data: xarray DataArray or Dataset to plot

        Returns:
            Holoviews plot object
        """
        raise NotImplementedError

    def plot_preview(
        self,
        data: xr.DataArray | xr.Dataset,
    ):
        """Plot a preview of the measurement results.

        Args:
            data: xarray DataArray or Dataset to plot

        Returns:
            Holoviews plot object
        """
        raise NotImplementedError

    # ============================================================================
    # UI INTEGRATION
    # ============================================================================

    def _get_filename_suffix(
        self, with_timestamp: bool = True, with_coords: bool = True
    ) -> str:
        """Generate filename suffix based on configuration.

        Args:
            with_timestamp: Whether to include timestamp in suffix
            with_coords: Whether to include parameter ranges in suffix

        Returns:
            Filename suffix string
        """
        autogenerated_filename = (
            f"__{datetime.now().strftime('%d-%m-%YT%H-%M-%S')}"
            if with_timestamp
            else ""
        )

        if with_coords:
            for name, values in self._param_coords.items():
                values = np.asarray(values)
                autogenerated_filename += f"__{name}-{values[0]}-{values[-1]}"

        return autogenerated_filename

    def _live_info(self, *, update_interval: float = 0.1) -> None:
        """Display live information about the runner.

        Sets up notebook widgets for progress display, pause/cancel controls,
        and live plotting. Falls back to tqdm progress bar if notebooks
        are not supported.

        Args:
            update_interval: Seconds between UI updates (default: 0.1)
        """
        live_info_elements = live_info(
            self,
            on_toggle_pause=self.pause_unpause,
            on_cancel=self.cancel,
        )
        self.update_plots = live_plot(self)

        self.LOG.debug("Using notebook live info integration")
        self._ui_update = live_info_elements.ui_update
        self.pbar = live_info_elements.pbar

        self._ui_ctx = logging_redirect_ipywidgets(
            live_info_elements.output, loggers=[self.LOG]
        )
        self._ui_ctx.__enter__()

        def finish_wrapper():
            self.LOG.debug("Finalizing UI...")
            live_info_elements.ui_finish()

        self._ui_finish = finish_wrapper

        # create a thread to periodically update the UI
        def ui_updater():
            idx = 0
            while not self.finished.is_set():
                self._ui_update()
                if self.current_index > idx:
                    idx = self.current_index
                time.sleep(update_interval)
            self._ui_update()
            self._ui_ctx.__exit__(None, None, None)

        self._ui_thread = threading.Thread(target=ui_updater, daemon=True)

    def __del__(self):
        """Clean up when the measurement object is deleted."""
        self.finished.set()
        self._ui_finish()

    def _ipython_display_(self):
        """IPython display hook - start measurement when displayed in Jupyter."""
        self.start()


# ============================================================================
# MODULE-LEVEL HELPER FUNCTIONS
# ============================================================================


def _prepare_coord(coord: Any) -> npt.ArrayLike:
    """Normalize coordinate arrays for parameter ranges.

    Converts coordinates to numpy arrays, handling 0D, 1D, and 2D cases.
    2D coords are converted to structured arrays for use as parameter values.

    Args:
        coord: Input coordinate values (scalar, list, array, or list of tuples)

    Returns:
        Normalized coordinate array

    Raises:
        ValueError: If coordinates have more than 2 dimensions
    """
    result = np.asarray(coord)

    if result.ndim == 0:
        return result.reshape(1)
    elif result.ndim == 1:
        return result
    elif result.ndim == 2:
        struct = np.dtype([("", result.dtype)] * result.shape[1])
        return np.asarray([tuple(c) for c in coord], dtype=struct)
    else:
        raise ValueError("Only 1D and 2D coords are supported for now")
