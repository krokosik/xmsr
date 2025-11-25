import itertools
import json
import logging
import os
import shutil
import sys
import time
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any, Optional, TypedDict, Union

import numpy as np
import numpy.typing as npt
import xarray as xr
import zarr
from tqdm import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect

_CURRENT_INDEX_KEY = "__CURRENT_INDEX__"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


@dataclass
class VariableData:
    name: str
    dims: Collection[str] = field(default_factory=tuple)
    coords: Mapping[str, Any] = field(default_factory=dict)


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

    running = Event()
    progress_queue = Queue[ProgressDict]()
    revert_progress = Queue[tuple[int, ...]]()
    chunk_queue = Queue[xr.DataArray | xr.Dataset]()
    finished = Event()

    _is_single_run = True

    def _revert_progress(self, indices: Union[tuple[int, ...], int]):
        previous_index = getattr(self, "_current_index", 0)
        self.current_index = indices
        self.pbar.unpause()
        self.pbar.update(self.current_index - previous_index)
        self.progress_queue.put_nowait(self._get_progress_dict())

    param_coords: dict[str, Any]

    _param_coords: Mapping[str, Any]

    variables: list[VariableData]

    data_dims: Iterable[str]  # deprecated, use VariableData.dims instead
    data_coords: Mapping[str, Any] = {}  # deprecated, use VariableData.coords instead

    _combinations: list[tuple[tuple[int, ...], tuple[Any, ...]]]
    _last_measurement_shapes: Optional[tuple[tuple[int, ...], ...]]
    _current_index: int
    _path: Path
    LOG: logging.Logger

    metadata: dict[str, Any] = {}
    filename: str
    timestamp: bool = True
    overwrite: bool = False
    with_coords: bool = True
    target_directory: str = os.getcwd()

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
        self.metadata = metadata if metadata is not None else self.metadata
        self.filename = filename if filename is not None else self.__class__.__name__
        self.timestamp = timestamp if timestamp is not None else self.timestamp
        self.overwrite = overwrite if overwrite is not None else self.overwrite
        self.with_coords = with_coords if with_coords is not None else self.with_coords
        self.target_directory = (
            target_directory if target_directory is not None else self.target_directory
        )

        self.LOG = logging.getLogger(self.__class__.__name__)
        self.LOG.setLevel(logging.INFO)

    def prepare(self, metadata: dict[str, Any]):
        """Perform some actions before the measurement starts.

        Args:
            metadata: Serializable data to be stored in the store
                attributes. You can mutate this dictionary to add more data.
        """
        self.LOG.debug("Preparing...")

    @property
    def current_index(self) -> int:
        return self._current_index

    @current_index.setter
    def current_index(self, index: Union[tuple[int, ...], int]):
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
        return self._combinations[self.current_index]

    @property
    def ntotal(self) -> int:
        return len(self._combinations)

    def get_index_by_indices(self, indices: tuple[int, ...]) -> int:
        for i, (combination_indices, _) in enumerate(self._combinations):
            if combination_indices == indices:
                return i

        self.LOG.error(f"No combination with indices {indices} found")
        return -1

    def get_measurement_by_indices(
        self, indices: tuple[int, ...]
    ) -> xr.Dataset | xr.DataArray:
        if not hasattr(self, "_path"):
            self.LOG.error("Measurement not started yet")
            return xr.Dataset()

        return (xr.open_dataset if len(self.variables) > 1 else xr.open_dataarray)(
            self._path, engine="zarr"
        ).isel(dict(zip(self.param_coords.keys(), indices)))

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

    def __init_subclass__(cls):
        if not hasattr(cls, "param_coords"):
            raise ValueError("Measurement subclasses must define 'param_coords'")

        if getattr(cls, "data_dims", None) or getattr(cls, "data_coords", None):
            logging.warning(
                "'data_dims' and 'data_coords' are deprecated, please use 'variables' instead",
            )
            if not hasattr(cls, "variables"):
                cls.variables = [
                    VariableData(
                        "result",
                        getattr(cls, "data_dims", ()),
                        getattr(cls, "data_coords", {}),
                    )
                ]
        elif not hasattr(cls, "variables"):
            logging.warning(
                "Measurement subclasses should define 'variables' to describe the output data",
            )
            cls.variables = [VariableData("result", (), {})]

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

    def _start_measurement(self):
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
        except Exception:
            self.current_index = 0

        if self.overwrite and self.current_index > 0:
            self.LOG.debug(f"Overwriting {self.current_index} measurements...")
            self.current_index = 0
            shutil.rmtree(self._path)

        self.LOG.debug("Starting measurements...")

    def step(self, idx: int | None = None):
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

    def _finalize_measurement(self):
        self.finish(self.metadata)
        self._update_metadata()

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

            self.running.set()
            self.finished.clear()

            with tqdm_logging_redirect(
                tqdm_class=tqdm,
                total=len(self._combinations),
                initial=self.current_index,
                unit="measurement",
            ) as self.pbar:
                self.progress_queue.put_nowait(self._get_progress_dict())

                while not self.finished.is_set():
                    self.running.wait()
                    if self._is_single_run and self.current_index >= self.ntotal:
                        self.finished.set()

                    if self.finished.is_set():
                        break

                    result = self.step()

                    self.store_ndarray(self.current_point[0], result)
                    self.pbar.update(1)
                    self.current_index += 1

                    if self.current_index >= len(self._combinations):
                        self.finished.set()

                    time.sleep(0)  # process events in blocking version

                if self.finished.is_set():
                    self.running.clear()
                    self._finalize_measurement()

        except Exception as e:
            self.LOG.exception(e)
            raise e

    def _get_progress_dict(self) -> ProgressDict:
        return {
            **self.pbar.format_dict,
            "indices": self._combinations[
                min(self.current_index, len(self._combinations) - 1)
            ][0],
        }  # type: ignore

    def store_ndarray(
        self, indices: tuple[int, ...], data: tuple[np.ndarray, ...]
    ) -> None:
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
                        ),
                        dims=tuple(self._param_coords.keys()) + tuple(var.dims),
                        coords={**self._param_coords, **(var.coords or {})},
                    )
                    for var, datum in zip(self.variables, data)
                }
            )
            (
                ds if len(self.variables) > 1 else ds[self.variables[0].name]
            ).assign_attrs(**self.metadata).to_zarr(self._path, mode="w")

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

        (ds if len(self.variables) > 1 else ds[self.variables[0].name]).expand_dims(
            dim=list(self._param_coords.keys())
        ).drop_vars(
            sum([list(var.coords.keys()) for var in self.variables], [])
        ).to_zarr(
            self._path,
            region={
                dim: slice(index, index + 1)
                for dim, index in zip(self._param_coords.keys(), indices)
            },
        )

        self.chunk_queue.put(ds)

        self._update_metadata()

        # Zarr attributes are for some reason separate from xarray attributes
        # so we hide the data not used in the result
        zarr.open(str(self._path), mode="a").attrs[_CURRENT_INDEX_KEY] = (
            self.current_index + 1
        )

    def _default_data_dims(self, data: np.ndarray) -> tuple[str, ...]:
        return tuple((f"dim{i}" for i in range(data.ndim)))

    def _update_metadata(self):
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
        """
        ds = xr.open_dataset(self._path, engine="zarr")
        if len(self.variables) == 1:
            return ds[self.variables[0].name]
        return ds

    def _get_filename_suffix(
        self, with_timestamp: bool = True, with_coords: bool = True
    ) -> str:
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

    def __del__(self):
        if hasattr(self, "pbar") and self.pbar is not None:
            self.pbar.close()

    def plot_result(self, *args, **kwargs):
        try:
            return self.result.plot(*args, **kwargs)
        except Exception:
            self.LOG.error(
                "Default plot_result method failed, please provide a custom plot function"
            )

    def plot_preview(
        self,
        chunk_da: xr.DataArray | xr.Dataset,
        full_da: xr.DataArray | xr.Dataset,
        ax,
    ):
        """Plot a preview of the measurement results.

        When the measurement is run through the GUI, this method will be called
        after each measurement to plot a preview of the results. The plot will be
        displayed in the GUI. If the method is not implemented, no preview will be
        shown.

        Args:
            data: The measurement results for a single coordinate combination. The
                shape of the array is the same as the shape of the measurement and
                the dimensions and coordinates are the same as specified in the
                `data_dims` and `data_coords` class attributes.
            ax: The matplotlib axis to plot on.
        """
        raise NotImplementedError


def _prepare_coord(coord: Any) -> npt.ArrayLike:
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
