import contextlib
import logging
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Event
from typing import Any, Optional, TypedDict, cast

import holoviews as hv
import numpy as np
import xarray as xr
import zarr

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


class Measurement(Thread):
    """Abstract base class for xarray-native parametric measurements.

    Subclasses define measurement schema using xarray templates:
      - `sweep_template`: sweep dimensions and coordinates
      - `result_template`: per-point measurement payload schema

    The `measure` method is called once per sweep point and can return either:
      - xarray DataArray/Dataset matching `result_template`, or
      - numpy/scalar values that can be auto-wrapped to match `result_template`

    Auto-wrap rules:
      - DataArray template: `measure` may return a single ndarray/scalar matching
        template shape.
      - Dataset template: `measure` may return a tuple whose positional entries
        map to `result_template.data_vars` order.
      - Returned shapes must match template variable shapes exactly.
    """

    _is_single_run = True

    sweep_template: xr.Dataset | xr.DataArray
    result_template: xr.Dataset | xr.DataArray

    LOG: logging.Logger

    filename: str
    timestamp: bool = True
    overwrite: bool = False
    zarr_format: int = 3
    with_coords: bool = True
    target_directory: str = os.getcwd()

    live_plot_opts: hv.opts

    def __init__(
        self,
        overwrite: Optional[bool] = None,
        filename: Optional[str] = None,
        timestamp: Optional[bool] = None,
        with_coords: Optional[bool] = None,
        zarr_format: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
        target_directory: Optional[str] = None,
    ):
        super().__init__()
        self.metadata = metadata if metadata is not None else {}
        self.filename = filename if filename is not None else self.__class__.__name__
        self.timestamp = timestamp if timestamp is not None else self.timestamp
        self.overwrite = overwrite if overwrite is not None else self.overwrite
        self.with_coords = with_coords if with_coords is not None else self.with_coords
        self.zarr_format = zarr_format if zarr_format is not None else self.zarr_format
        self.target_directory = (
            target_directory if target_directory is not None else self.target_directory
        )

        if self.zarr_format not in {2, 3}:
            raise ValueError("'zarr_format' must be 2 or 3")
        if (
            self.zarr_format == 3
            and int(zarr.__version__.split(".")[0]) < 3
            and os.environ.get("ZARR_V3_EXPERIMENTAL_API") != "1"
        ):
            raise ValueError(
                "zarr format 3 requires ZARR_V3_EXPERIMENTAL_API=1 with zarr<3"
            )

        self.running = Event()
        self.progress_queue = Queue[ProgressDict]()
        self.revert_progress = Queue[tuple[int, ...]]()
        self.last_measurement = None
        self.last_measurement_queue = Queue[xr.DataArray | xr.Dataset]()
        self.finished = Event()
        self.cancelled = Event()
        self.started = Event()

        self.LOG = logging.getLogger(self.__class__.__name__)
        self.LOG.setLevel(logging.INFO)

    def __init_subclass__(cls):
        if not hasattr(cls, "sweep_template"):
            raise ValueError("Measurement subclasses must define 'sweep_template'")
        if not hasattr(cls, "result_template"):
            raise ValueError("Measurement subclasses must define 'result_template'")

        sweep = cls.sweep_template
        if isinstance(sweep, xr.DataArray):
            sweep = sweep.to_dataset(name=sweep.name or "sweep")
        if not isinstance(sweep, xr.Dataset):
            raise TypeError("'sweep_template' must be an xarray Dataset or DataArray")

        if len(sweep.dims) == 0:
            raise ValueError(
                "'sweep_template' must define at least one sweep dimension"
            )

        result_template = cls.result_template
        if isinstance(result_template, xr.DataArray):
            cls._result_is_dataarray = True
            cls._result_var_name = str(result_template.name or "result")
            cls._result_template_ds = result_template.to_dataset(
                name=cls._result_var_name
            )
        elif isinstance(result_template, xr.Dataset):
            if len(result_template.data_vars) == 0:
                raise ValueError("'result_template' dataset must define data variables")
            cls._result_is_dataarray = False
            cls._result_var_name = ""
            cls._result_template_ds = result_template
        else:
            raise TypeError("'result_template' must be an xarray Dataset or DataArray")

        cls._sweep = sweep
        cls._sweep_dims = tuple(str(dim) for dim in sweep.sizes.keys())
        cls._sweep_sizes = tuple(int(size) for size in sweep.sizes.values())

        coords_ds = cls.sweep_template.assign_coords(cls.result_template.coords)
        cls.coords = (
            coords_ds if isinstance(coords_ds, xr.Dataset) else coords_ds.to_dataset()
        )
        for coord in cls.coords.values():
            if coord.ndim > 1:
                raise ValueError(
                    "Sweep and result coordinates must be 1D. Use derived coordinates for multi-dimensional coordinate logic."
                )

    def prepare(self, metadata: dict):
        self.LOG.debug("Preparing...")

    def measure(
        self,
        values: dict,
        indices: dict[str, int],
        metadata: dict,
    ) -> np.typing.ArrayLike | tuple[np.typing.ArrayLike, ...]:
        """Measure one sweep point.

        Args:
            values: Coordinate values for current sweep indices.
            indices: Integer indices for sweep dimensions and derived coordinates.
            metadata: Mutable run metadata persisted with the result.

        Returns:
            Either xarray objects matching `result_template` or values that can be
            auto-wrapped according to the class auto-wrap rules.
        """
        raise NotImplementedError

    def finish(self, metadata: dict):
        self.LOG.debug("Done")

    def _start_measurement(self):
        self._last_measurement_shapes = None

        filename = self.filename + self._get_filename_suffix(
            self.timestamp, self.with_coords
        )
        self._path = (Path(self.target_directory) / filename).with_suffix(".zarr")
        self.LOG.debug(f"Results will be stored in:\n{filename}")
        self.LOG.debug(f"Full path:\n{os.path.abspath(self._path)}")

        try:
            store_attrs = dict(zarr.open(str(self._path), mode="r").attrs)
            existing_metadata = {
                key: value
                for key, value in store_attrs.items()
                if key != _CURRENT_INDEX_KEY
            }
            existing_metadata.update(self.metadata)
            self.metadata = existing_metadata

            current_index_attr = store_attrs[_CURRENT_INDEX_KEY]
            self.current_index = int(cast(Any, current_index_attr))
            self.LOG.debug(
                f"Found existing data with {self.current_index} measurements, resuming..."
            )
            current_data = self.result
            for i in range(self.current_index):
                self.last_measurement_queue.put_nowait(
                    current_data.isel(self._indices_dict_for_linear_index(i))
                )
        except Exception:
            self.current_index = 0

        if self.overwrite and self.current_index > 0:
            self.LOG.debug(f"Overwriting {self.current_index} measurements...")
            self.current_index = 0
            shutil.rmtree(self._path)

        self.prepare(self.metadata)
        self.LOG.debug("Starting measurements...")

    def _indices_tuple_for_linear_index(self, idx: int) -> tuple[int, ...]:
        return tuple(int(i) for i in np.unravel_index(idx, self._sweep_sizes))

    def _indices_dict_for_linear_index(self, idx: int) -> dict[str, int]:
        return dict(zip(self._sweep_dims, self._indices_tuple_for_linear_index(idx)))

    def _linear_index_for_indices_tuple(self, indices: tuple[int, ...]) -> int:
        return int(np.ravel_multi_index(indices, self._sweep_sizes))

    def _coord_value_for_indices(
        self, coord: xr.DataArray, indices_by_dim: dict[str, int]
    ) -> Any:
        coord_dims = tuple(str(dim) for dim in coord.dims)
        indexers = {
            dim: indices_by_dim[dim] for dim in coord_dims if dim in indices_by_dim
        }
        selected = coord.isel(indexers) if indexers else coord
        value = np.asarray(selected.values)
        if value.ndim == 0:
            return value.item()
        return selected.values

    def _values_and_indices_for_indices_tuple(
        self, indices: tuple[int, ...]
    ) -> tuple[dict[str, Any], dict[str, int]]:
        indices_by_dim = dict(zip(self._sweep_dims, indices))

        values: dict[str, Any] = {}
        indices_dict = {dim: index for dim, index in indices_by_dim.items()}

        for dim, index in indices_by_dim.items():
            coord = self._sweep.coords.get(dim)
            if coord is None:
                values[dim] = index
            else:
                values[dim] = self._coord_value_for_indices(coord, indices_by_dim)

        for coord_name, coord in self._sweep.coords.items():
            name = str(coord_name)
            if name in values:
                continue

            coord_dims = tuple(str(dim) for dim in coord.dims)
            if not coord_dims or any(dim not in indices_by_dim for dim in coord_dims):
                continue

            values[name] = self._coord_value_for_indices(coord, indices_by_dim)

            indices_dict[name] = indices_by_dim[coord_dims[0]]

        return values, indices_dict

    def step(self, idx: int | None = None):
        if idx is None:
            idx = self.current_index

        indices_tuple = self._indices_tuple_for_linear_index(idx)
        values, indices = self._values_and_indices_for_indices_tuple(indices_tuple)

        self.LOG.debug(
            f"Measurement no. {idx} indices: {indices_tuple} values: {tuple(values.values())}"
        )

        result = self.measure(values, indices, self.metadata)
        wrapped = self._coerce_measurement_output(result)

        current_shapes = tuple(arr.shape for arr in wrapped.data_vars.values())
        if self._last_measurement_shapes is None:
            self._last_measurement_shapes = current_shapes
        else:
            assert current_shapes == self._last_measurement_shapes, (
                "All measurements must have the same shapes as 'result_template'"
            )

        return indices_tuple, wrapped

    def run(self) -> None:
        try:
            self._start_measurement()
            self._live_info()

            self._wait_on(self.started)

            if hasattr(self, "_ui_thread"):
                self._ui_thread.start()

            self.running.set()
            self.finished.clear()

            while not self.finished.is_set():
                self._wait_on(self.running)
                if self._is_single_run and self.current_index >= self.ntotal:
                    self.LOG.debug("Single run completed.")
                    self.finished.set()

                if self.finished.is_set():
                    break

                indices_tuple, result = self.step()

                self.store_xarray(indices_tuple, result)
                self.LOG.debug("Stored measurement chunk.")
                self.current_index += 1
                self._ui_update()
                self.LOG.debug("UI updated.")

                self.LOG.debug(f"Current index: {self.current_index}")

                if self.current_index >= self.ntotal:
                    self.finished.set()

                time.sleep(0)

            if self.finished.is_set():
                self.running.clear()
                self._finalize_measurement()

        except Exception as e:
            self.LOG.exception(e)
            self.cancel()
            self._exception = e
            raise e

    def _finalize_measurement(self):
        self.LOG.debug("Finalizing measurement...")
        self.finish(self.metadata)
        self._update_metadata()
        self._ui_finish()

    def _wait_on(self, event: Event):
        if threading.current_thread() is not threading.main_thread():
            event.wait()

    @property
    def current_index(self) -> int:
        return self._current_index

    @current_index.setter
    def current_index(self, index: tuple[int, ...] | int):
        self.LOG.debug(f"Setting current index to {index}")
        if isinstance(index, int):
            self._current_index = index
        else:
            self._current_index = self._linear_index_for_indices_tuple(index)

        if hasattr(self, "pbar"):
            self.progress_queue.put_nowait(self._get_progress_dict())

    @property
    def current_point(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        indices_tuple = self._indices_tuple_for_linear_index(self.current_index)
        values, _ = self._values_and_indices_for_indices_tuple(indices_tuple)
        return indices_tuple, values

    @property
    def ntotal(self) -> int:
        return int(np.prod(self._sweep_sizes))

    @property
    def status(self) -> MeasurementStatus:
        if hasattr(self, "_exception"):
            return MeasurementStatus.FAILED
        if not self.started.is_set():
            return MeasurementStatus.INIT
        if self.cancelled.is_set():
            return MeasurementStatus.CANCELLED
        if self.finished.is_set():
            return MeasurementStatus.FINISHED
        if not self.running.is_set():
            return MeasurementStatus.PAUSED
        return MeasurementStatus.RUNNING

    def get_index_by_indices(self, indices: tuple[int, ...]) -> int:
        if len(indices) != len(self._sweep_dims):
            self.LOG.error(f"Invalid number of indices {indices}")
            return -1
        return self._linear_index_for_indices_tuple(indices)

    def pause_unpause(self) -> None:
        if self.running.is_set():
            self.running.clear()
        else:
            self.running.set()
            if not self.started.is_set():
                self.started.set()
        self._ui_update()

    def wait_for_manual_resume(
        self, message: str, *, clear_running: bool = True
    ) -> None:
        """Pause the measurement and wait for explicit user resume.

        This helper is useful when a parameter must be adjusted manually between
        automated measurement steps.

        Args:
            message: Instruction shown to the operator before pausing.
            clear_running: Whether to force pause before waiting.
        """
        self.LOG.info(message)
        if clear_running:
            self.running.clear()
        self._wait_on(self.running)

    def cancel(self) -> None:
        self.cancelled.set()
        self.finished.set()
        self.running.clear()

    def _get_progress_dict(self) -> ProgressDict:
        clamped = min(self.current_index, max(self.ntotal - 1, 0))
        progress = cast(
            ProgressDict,
            {
                **self.pbar.format_dict,
                "indices": self._indices_tuple_for_linear_index(clamped),
            },
        )
        return progress

    def _revert_progress(self, indices: tuple[int, ...] | int):
        previous_index = getattr(self, "_current_index", 0)
        self.current_index = indices
        self.pbar.unpause()
        self.pbar.update(self.current_index - previous_index)
        self.progress_queue.put_nowait(self._get_progress_dict())

    def _combine_sweep_and_result_templates(self) -> xr.Dataset:
        merged = xr.Dataset()

        for var_name, template_da in self._result_template_ds.data_vars.items():
            shape = self._sweep_sizes + template_da.shape
            dims = self._sweep_dims + template_da.dims
            data = _initial_data(shape, template_da.dtype)

            coords: dict[str, Any] = {}
            coords.update({str(k): v for k, v in self._sweep.coords.items()})
            coords.update({str(k): v for k, v in template_da.coords.items()})

            merged[var_name] = xr.DataArray(
                data,
                dims=dims,
                coords=coords,
                attrs=dict(template_da.attrs),
            )

        merged = merged.assign_coords(
            {str(k): v for k, v in self._sweep.coords.items()}
        )
        merged = merged.assign_attrs(dict(self.metadata))

        return self._apply_derived_xindexes(merged)

    def _apply_derived_xindexes(self, data: xr.Dataset) -> xr.Dataset:
        sweep_dims = set(self._sweep_dims)
        derived_coords = [
            str(name)
            for name, coord in data.coords.items()
            if str(name) not in data.dims
            and coord.dims
            and set(str(dim) for dim in coord.dims).issubset(sweep_dims)
        ]
        for coord_name in derived_coords:
            if coord_name in data.xindexes:
                continue
            try:
                data = data.set_xindex(coord_name)
            except Exception:
                self.LOG.debug(f"Skipping xindex creation for coord '{coord_name}'")
        return data

    def _to_public_result_type(self, ds: xr.Dataset) -> xr.Dataset | xr.DataArray:
        if self._result_is_dataarray:
            return ds[self._result_var_name]
        return ds

    def _zarr_store(self) -> str:
        return str(self._path)

    def _coerce_measurement_output(
        self, value: xr.DataArray | xr.Dataset | Any
    ) -> xr.Dataset:
        if isinstance(value, xr.Dataset):
            _validate_dataset_like_template(value, self._result_template_ds)
            return value

        if isinstance(value, xr.DataArray):
            if not self._result_is_dataarray:
                raise TypeError(
                    "'measure' returned DataArray but 'result_template' is Dataset"
                )
            expected = self._result_template_ds[self._result_var_name]
            candidate = value.rename(self._result_var_name)
            _validate_dataarray_like_template(candidate, expected)
            return candidate.to_dataset(name=self._result_var_name)

        if self._result_is_dataarray:
            expected = self._result_template_ds[self._result_var_name]
            data = np.asarray(value)
            if data.shape != expected.shape:
                raise ValueError(
                    f"Auto-wrapped result shape {data.shape} does not match template shape {expected.shape}"
                )
            da = xr.DataArray(
                data,
                dims=expected.dims,
                coords={k: v for k, v in expected.coords.items()},
                attrs=dict(expected.attrs),
                name=self._result_var_name,
            )
            return da.to_dataset(name=self._result_var_name)

        values = value if isinstance(value, tuple) else (value,)
        expected_vars = list(self._result_template_ds.data_vars.items())
        if len(values) != len(expected_vars):
            raise ValueError(
                f"Expected {len(expected_vars)} return values, got {len(values)}"
            )

        ds = xr.Dataset()
        for raw, (var_name, expected) in zip(values, expected_vars):
            arr = np.asarray(raw)
            if arr.shape != expected.shape:
                raise ValueError(
                    f"Variable '{var_name}' shape {arr.shape} does not match template shape {expected.shape}"
                )
            ds[var_name] = xr.DataArray(
                arr,
                dims=expected.dims,
                coords={k: v for k, v in expected.coords.items()},
                attrs=dict(expected.attrs),
            )

        return ds

    def store_xarray(self, indices: tuple[int, ...], data: xr.Dataset) -> None:
        if not self._path.exists():
            combined = self._combine_sweep_and_result_templates()
            self._to_public_result_type(combined).to_zarr(
                cast(Any, self._zarr_store()),
                mode="w",
                zarr_format=self.zarr_format,
            )

        point_coords = {
            dim: [self._sweep.coords[dim].values[index]]
            for dim, index in zip(self._sweep_dims, indices)
        }

        expanded = data.expand_dims(point_coords)

        non_region_coord_vars = [
            name
            for name, coord in expanded.coords.items()
            if set(coord.dims).isdisjoint(self._sweep_dims)
        ]
        if non_region_coord_vars:
            expanded = expanded.drop_vars(non_region_coord_vars)

        self._to_public_result_type(expanded).to_zarr(
            cast(Any, self._zarr_store()),
            region={
                dim: slice(index, index + 1)
                for dim, index in zip(self._sweep_dims, indices)
            },
            zarr_format=self.zarr_format,
        )

        self.last_measurement = self._to_public_result_type(data)
        self.last_measurement_queue.put_nowait(self.last_measurement)

        if hasattr(self, "update_plots"):
            self.update_plots()

        self._update_metadata()
        zarr.open(str(self._path), mode="a").attrs[_CURRENT_INDEX_KEY] = (
            self.current_index + 1
        )

    def get_measurement_by_indices(
        self, indices: tuple[int, ...]
    ) -> xr.Dataset | xr.DataArray:
        if not hasattr(self, "_path"):
            self.LOG.error("Measurement not started yet")
            return xr.Dataset()

        ds = self._apply_derived_xindexes(
            xr.open_dataset(self._path, engine="zarr", consolidated=False)
        )
        selected = ds.isel(dict(zip(self._sweep_dims, indices)))
        return self._to_public_result_type(selected)

    def _update_metadata(self):
        try:
            zarr.open(str(self._path), mode="a").attrs.update(self.metadata)
        except Exception:
            return

        if not self._result_is_dataarray:
            return

        var_name = self._result_var_name
        try:
            z = zarr.open_group(self._zarr_store(), mode="a")
            z[var_name].attrs.update(self.metadata)
        except Exception:
            return

    @property
    def result(self) -> xr.DataArray | xr.Dataset:
        ds = self._apply_derived_xindexes(
            xr.open_dataset(self._path, engine="zarr", consolidated=False)
        )
        return self._to_public_result_type(ds)

    def plot_result(self, *args, **kwargs):
        if hasattr(self.result, "hvplot"):
            return self.result.hvplot(*args, **kwargs)
        return self.result.plot(*args, **kwargs)

    def plot_single_step(
        self,
        data: xr.DataArray | xr.Dataset,
    ):
        raise NotImplementedError

    def plot_preview(
        self,
        data: xr.DataArray | xr.Dataset,
    ):
        raise NotImplementedError

    def _get_filename_suffix(
        self, with_timestamp: bool = True, with_coords: bool = True
    ) -> str:
        autogenerated_filename = (
            f"__{datetime.now().strftime('%d-%m-%YT%H-%M-%S')}"
            if with_timestamp
            else ""
        )

        if with_coords:
            for name in self._sweep_dims:
                values = np.asarray(self._sweep.coords[name].values)
                autogenerated_filename += f"__{name}-{values[0]}-{values[-1]}"

        return autogenerated_filename

    def _live_info(self, *, update_interval: float = 0.1) -> None:
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
        with contextlib.suppress(Exception):
            self.finished.set()
        with contextlib.suppress(Exception):
            self._ui_finish()

    def _ipython_display_(self):
        self.start()


def _validate_dataarray_like_template(candidate: xr.DataArray, template: xr.DataArray):
    if tuple(candidate.dims) != tuple(template.dims):
        raise ValueError(
            f"Returned DataArray dims {candidate.dims} do not match template dims {template.dims}"
        )
    if candidate.shape != template.shape:
        raise ValueError(
            f"Returned DataArray shape {candidate.shape} does not match template shape {template.shape}"
        )


def _validate_dataset_like_template(candidate: xr.Dataset, template: xr.Dataset):
    if set(candidate.data_vars) != set(template.data_vars):
        raise ValueError("Returned Dataset variables do not match template variables")
    for name in template.data_vars:
        _validate_dataarray_like_template(candidate[name], template[name])


def _initial_data(shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
    if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating):
        data = np.full(shape, np.nan, dtype=dtype)
    elif np.issubdtype(dtype, np.bool_):
        data = np.zeros(shape, dtype=dtype)
    else:
        data = np.zeros(shape, dtype=dtype)
    return data
