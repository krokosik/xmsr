import logging
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from typing import Any, Optional, TypedDict, cast

import holoviews as hv
import numpy as np
import xarray as xr
import zarr

from xmsr.qt_integration import Thread
from xmsr.shared import MeasurementStatus

_CURRENT_INDEX_KEY = "__CURRENT_INDEX__"


class _RestartStep(Exception):
    """Unwind a cooperative operator wait after rewind or cancellation."""


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
        notebook_ui: bool = True,
    ):
        super().__init__()
        self.notebook_ui = notebook_ui
        self._ui_update = lambda: None
        self._ui_finish = lambda: None
        self._progress_started_at: float | None = None
        self._progress_finished_at: float | None = None
        self._progress_initial_index = 0
        self._current_index = 0
        self._control_lock = threading.RLock()
        self._plot_lock = threading.RLock()
        self._control_changed = threading.Condition(self._control_lock)
        self._controls_used = False
        self._valid_until = 0
        self._cursor_revision = 0
        self._selected_index: int | None = None
        self._in_step = False
        self._waiting_manual = False
        self._session_ready = False
        self._active = False
        self._has_run = False
        self._background_run = False
        self._resume_attempt = False
        self._finalized = False
        self._recoverable_error = False
        self._resumed_from_disk = False
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
        self.paused = Event()
        self.paused.set()

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
        if any(size == 0 for size in cls._sweep_sizes):
            raise ValueError("Sweep dimensions must not be empty")

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

        if self._path.exists() and self.overwrite:
            self.LOG.info("Overwriting existing measurement archive")
            shutil.rmtree(self._path)
        if self._path.exists():
            archive = zarr.open_group(str(self._path), mode="r")
            self.zarr_format = archive.metadata.zarr_format
            store_attrs = dict(archive.attrs)
            checkpoint = store_attrs.get(_CURRENT_INDEX_KEY)
            if (
                isinstance(checkpoint, bool)
                or not isinstance(checkpoint, int)
                or not 0 <= checkpoint <= self.ntotal
            ):
                raise ValueError(
                    "Archive has a missing or invalid measurement checkpoint"
                )
            self._validate_archive()
            existing_metadata = {
                key: value
                for key, value in store_attrs.items()
                if key != _CURRENT_INDEX_KEY
            }
            existing_metadata.update(self.metadata)
            self.metadata = existing_metadata

            self._valid_until = checkpoint
            self.current_index = checkpoint
            self._resumed_from_disk = True
            self.LOG.debug(
                f"Found existing data with {self.current_index} measurements, resuming..."
            )
            current_data = self.result
            for i in range(self.current_index):
                self.last_measurement_queue.put_nowait(
                    current_data.isel(self._indices_dict_for_linear_index(i))
                )
            if checkpoint:
                self.last_measurement = current_data.isel(
                    self._indices_dict_for_linear_index(checkpoint - 1)
                ).load()
        else:
            self._valid_until = 0
            self.current_index = 0

        self.prepare(self.metadata)
        self._session_ready = True
        self.LOG.debug("Starting measurements...")

    def _validate_archive(self) -> None:
        """Reject incompatible archives instead of treating them as new runs."""
        with xr.open_dataset(
            self._path, engine="zarr", consolidated=False, mask_and_scale=False
        ) as stored:
            if set(stored.data_vars) != set(self._result_template_ds.data_vars):
                raise ValueError("Archive variables do not match result_template")
            for name, template in self._result_template_ds.data_vars.items():
                variable = stored[name]
                if (
                    variable.dims != self._sweep_dims + template.dims
                    or variable.shape != self._sweep_sizes + template.shape
                    or variable.dtype != template.dtype
                ):
                    raise ValueError(f"Archive schema does not match variable {name!r}")
            for name, coord in self.coords.coords.items():
                if name not in stored.coords or stored[name].dims != coord.dims:
                    raise ValueError(f"Archive coordinate {name!r} does not match")
                try:
                    np.testing.assert_array_equal(stored[name].values, coord.values)
                except AssertionError as error:
                    raise ValueError(
                        f"Archive coordinate {name!r} does not match"
                    ) from error

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

        from .runner_widget import capture_measurement_output

        with capture_measurement_output(getattr(self, "_runner_widget", None)):
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

    def start(self, *args: Any, **kwargs: Any) -> None:
        """Initialize notebook rendering before launching the acquisition thread.

        Args:
            *args: Positional arguments passed to the threading or Qt backend.
            **kwargs: Keyword arguments passed to the backend's start method.
        """
        if self.notebook_ui:
            from .notebook_integration import initialize_notebook

            initialize_notebook()
        super().start(*args, **kwargs)

    def run(self) -> None:
        """Acquire the sweep; background step failures wait for retry or rewind.

        Blocking main-thread execution continues to raise acquisition errors.
        Setup/finalization errors are terminal in both modes.
        """
        with self._control_lock:
            if self._active:
                raise RuntimeError("The measurement session is already running")
            self._active = True
            self._background_run = (
                threading.current_thread() is not threading.main_thread()
            )
            reuse_setup = self._resume_attempt and self._session_ready
            self._resume_attempt = False
            if reuse_setup:
                self._recoverable_error = False
                if hasattr(self, "_exception"):
                    del self._exception
                if not self._background_run:
                    self.running.set()
                    self.started.set()
            elif self._has_run and self.finished.is_set():
                self.finished.clear()
                self.cancelled.clear()
                self.started.clear()
                self.running.clear()
                self._controls_used = False
                self._session_ready = False
                self._finalized = False
                self._selected_index = None
                self.last_measurement = None
                self._resumed_from_disk = False
                if hasattr(self, "_exception"):
                    del self._exception
            self._has_run = True
        try:
            if not reuse_setup:
                self._start_measurement()
                self._live_info()

            if threading.current_thread() is threading.main_thread():
                self.started.set()
            self._wait_on(self.started)
            with self._control_lock:
                if not self.cancelled.is_set():
                    self._progress_started_at = time.monotonic()
                    self._progress_finished_at = None
                    self._progress_initial_index = self.current_index
                    if not self._controls_used:
                        self.running.set()
                        self.paused.clear()
            self._ui_update()

            while (index := self._claim_next_step()) is not None:
                try:
                    revision = self._cursor_revision
                    indices_tuple, result = self.step(index)
                    if (
                        not self.cancelled.is_set()
                        and revision == self._cursor_revision
                    ):
                        self.store_xarray(indices_tuple, result)
                        with self._control_lock:
                            self.current_index = self._valid_until
                except _RestartStep:
                    pass
                except Exception as error:
                    with self._control_lock:
                        self._exception = error
                        self._recoverable_error = True
                        self.running.clear()
                    self.LOG.exception(
                        "Measurement step failed; retry or rewind to continue"
                    )
                    if not self._background_run:
                        raise
                finally:
                    with self._control_lock:
                        self._in_step = False
                        if not self.running.is_set():
                            self.paused.set()
                    self._ui_update()
                time.sleep(0)

            if self.finished.is_set():
                self.running.clear()
                self._finalize_measurement()

        except Exception as e:
            self._exception = e
            self.LOG.exception(e)
            self.running.clear()
            if self._recoverable_error and not self.finished.is_set():
                self._resume_attempt = True
            else:
                self._recoverable_error = False
                self.finished.set()
            raise
        finally:
            self._active = False
            self._in_step = False
            self.paused.set()
            self._progress_finished_at = time.monotonic()
            if self._recoverable_error and not self.finished.is_set():
                self._ui_update()
            else:
                self._ui_finish()

    def _claim_next_step(self) -> int | None:
        """Atomically claim a point only after pause/recovery has been released."""
        with self._control_changed:
            while not self.running.is_set() and not self.finished.is_set():
                self.paused.set()
                self._control_changed.wait(timeout=0.1)
            if self.finished.is_set():
                return None
            if self.current_index >= self.ntotal:
                self.finished.set()
                return None
            self.paused.clear()
            self._in_step = True
            return self.current_index

    def _finalize_measurement(self):
        if self._finalized:
            return
        self._finalized = True
        self._ensure_archive()
        self.LOG.debug("Finalizing measurement...")
        self.finish(self.metadata)
        self._update_metadata()

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
        if self.cancelled.is_set():
            return MeasurementStatus.CANCELLED
        if hasattr(self, "_exception"):
            return MeasurementStatus.FAILED
        if not self.started.is_set():
            return MeasurementStatus.INIT
        if self.finished.is_set():
            return MeasurementStatus.FINISHED
        if not self.running.is_set():
            if self._in_step and not self._waiting_manual:
                return MeasurementStatus.PAUSING
            return MeasurementStatus.PAUSED
        return MeasurementStatus.RUNNING

    @property
    def can_retry(self) -> bool:
        """Whether a failed background step can continue with the current setup."""
        return (
            self._recoverable_error
            and (not self._active or self._background_run)
            and not self._in_step
            and not self.finished.is_set()
            and self.status == MeasurementStatus.FAILED
        )

    @property
    def can_rewind(self) -> bool:
        """Whether the session is quiescent and accepts browsing/rewind commands."""
        return (
            self._session_ready
            and not self.finished.is_set()
            and not self.running.is_set()
            and (not self._in_step or self._waiting_manual)
            and (not self._active or self._background_run or self._waiting_manual)
        )

    def get_index_by_indices(self, indices: tuple[int, ...]) -> int:
        if len(indices) != len(self._sweep_dims):
            self.LOG.error(f"Invalid number of indices {indices}")
            return -1
        return self._linear_index_for_indices_tuple(indices)

    def pause_unpause(self) -> None:
        """Toggle pause for existing programmatic callers."""
        if self.running.is_set():
            self.pause()
        else:
            self.resume()

    def pause(self) -> None:
        """Request a pause before the next measurement step."""
        with self._control_lock:
            if self.status != MeasurementStatus.RUNNING:
                return
            self._controls_used = True
            self.running.clear()
            if not self._in_step:
                self.paused.set()
        self._ui_update()

    def resume(self) -> None:
        """Start or resume acquisition without toggling an existing run."""
        with self._control_lock:
            restart_worker = not self._active and self._resume_attempt
            if self.status == MeasurementStatus.FAILED and self.can_retry:
                del self._exception
                self._recoverable_error = False
                self._reset_progress()
            elif self.status not in {MeasurementStatus.INIT, MeasurementStatus.PAUSED}:
                return
            self._controls_used = True
            self._selected_index = None
            self.paused.clear()
            self.running.set()
            self.started.set()
            self._control_changed.notify_all()
        self._ui_update()
        if restart_worker:
            self.start()

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
        with self._control_lock:
            revision = self._cursor_revision
            if clear_running:
                self.running.clear()
            self._waiting_manual = True
            self.paused.set()
        try:
            self._ui_update()
            if threading.current_thread() is not threading.main_thread():
                with self._control_changed:
                    while (
                        not self.running.is_set()
                        and not self.finished.is_set()
                        and revision == self._cursor_revision
                    ):
                        self._control_changed.wait(timeout=0.1)
            with self._control_lock:
                if self.cancelled.is_set() or revision != self._cursor_revision:
                    raise _RestartStep
        finally:
            with self._control_lock:
                self._waiting_manual = False
                # Preserve blocking execution's existing no-wait behavior.
                if (
                    threading.current_thread() is threading.main_thread()
                    and not self.finished.is_set()
                    and revision == self._cursor_revision
                ):
                    self.running.set()
                if self.running.is_set():
                    self.paused.clear()

    def cancel(self) -> None:
        """Cancel acquisition and wake workers waiting for start or resume."""
        with self._control_lock:
            finalize = not self._active and self._session_ready and not self._finalized
            self._resume_attempt = False
            self.cancelled.set()
            self.finished.set()
            # Wake either wait point; cancellation takes precedence in status/run.
            self.started.set()
            self.running.set()
            self._control_changed.notify_all()
        self._ui_update()
        if finalize:
            try:
                self._finalize_measurement()
            finally:
                self._ui_finish()

    def _get_progress_dict(self) -> ProgressDict:
        clamped = min(self.current_index, max(self.ntotal - 1, 0))
        now = self._progress_finished_at or time.monotonic()
        elapsed = (
            max(0.0, now - self._progress_started_at)
            if self._progress_started_at is not None
            else 0.0
        )
        completed = max(0, self.current_index - self._progress_initial_index)
        rate = completed / elapsed if elapsed > 0 else 0.0
        return {
            "n": self.current_index,
            "total": self.ntotal,
            "percentage": 100 * self.current_index / self.ntotal
            if self.ntotal
            else 0.0,
            "elapsed": elapsed,
            "remaining": max(0, self.ntotal - self.current_index) / rate
            if rate
            else 0.0,
            "rate": rate,
            "unit": "it",
            "unit_scale": False,
            "unit_divisor": 1000,
            "ncols": None,
            "nrows": None,
            "indices": self._indices_tuple_for_linear_index(clamped)
            if self.ntotal
            else tuple(0 for _ in self._sweep_dims),
        }

    def _normalize_point(self, indices: tuple[int, ...] | int) -> int:
        if isinstance(indices, bool):
            raise ValueError("Point indices must be integers")
        if isinstance(indices, int):
            if not 0 <= indices < self.ntotal:
                raise ValueError("Point is outside the sweep")
            return indices
        if len(indices) != len(self._sweep_dims) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        ):
            raise ValueError("Provide one integer index for each sweep dimension")
        try:
            return self._linear_index_for_indices_tuple(indices)
        except ValueError as error:
            raise ValueError("Point is outside the sweep") from error

    @property
    def selected_index(self) -> int:
        """The browsed point, or the latest completed point while following live data."""
        if self._selected_index is not None:
            return self._selected_index
        return max(0, min(self._valid_until - 1, self.ntotal - 1))

    @property
    def display_measurement(self) -> xr.DataArray | xr.Dataset | None:
        """Data for the browsed point without changing the acquisition cursor."""
        if self._selected_index is None and self.last_measurement is not None:
            return self.last_measurement
        if not hasattr(self, "_path") or not self._path.exists():
            return None
        return self.get_measurement_by_indices(
            self._indices_tuple_for_linear_index(self.selected_index)
        )

    def select_point(self, indices: tuple[int, ...] | int | None) -> None:
        """Browse stored data while quiescent, or follow the latest point with None.

        Args:
            indices: A linear point index, sweep index tuple, or None for latest.

        Raises:
            RuntimeError: The runner is acquiring or has ended.
            ValueError: The point is outside the sweep.
        """
        with self._control_lock:
            if not self.can_rewind:
                raise RuntimeError("Wait until the runner is paused before browsing")
            self._selected_index = (
                None if indices is None else self._normalize_point(indices)
            )
        self._refresh_plots()
        self._ui_update()

    def rewind(
        self,
        indices: tuple[int, ...] | int,
        *,
        expected_revision: int | None = None,
        resume: bool = False,
    ) -> None:
        """Persistently invalidate the selected point and the rest of the sweep.

        By default the session stays paused. Resume remeasures it without preparing
        instruments again. Data below the selected point remains valid.

        Args:
            indices: A linear point index or sweep index tuple.
            expected_revision: Optional UI revision to reject stale commands.
            resume: Start acquisition at the selected point after invalidation.

        Raises:
            RuntimeError: The session is not quiescent or the command is stale.
            ValueError: The point is outside the acquired prefix/next point.
        """
        with self._control_lock:
            if not self.can_rewind:
                raise RuntimeError("Wait until the runner is paused before rewinding")
            if (
                expected_revision is not None
                and expected_revision != self._cursor_revision
            ):
                raise RuntimeError("Runner changed; select the point again")
            index = self._normalize_point(indices)
            if index > self.current_index:
                raise ValueError("Cannot rewind to a point ahead of acquisition")
            self._ensure_archive()
            self._persist_checkpoint(index)
            self._valid_until = index
            self.current_index = index
            self._cursor_revision += 1
            self._selected_index = None if resume else index
            self._controls_used = True
            restart_worker = resume and not self._active
            if not self._active:
                self._resume_attempt = True
            self._recoverable_error = False
            if hasattr(self, "_exception"):
                del self._exception
            self._reset_progress()
            for queue in (self.last_measurement_queue, self.progress_queue):
                while True:
                    try:
                        queue.get_nowait()
                    except Empty:
                        break
            self.progress_queue.put_nowait(self._get_progress_dict())
            self.last_measurement = (
                self.get_measurement_by_indices(
                    self._indices_tuple_for_linear_index(index - 1)
                ).load()
                if index
                else None
            )
            self.revert_progress.put_nowait(self._indices_tuple_for_linear_index(index))
            if resume:
                self.running.set()
                self.started.set()
                self.paused.clear()
            self._control_changed.notify_all()
        self.LOG.info("Rewound to point %s; later measurements are invalid", index)
        self._refresh_plots()
        self._ui_update()
        if restart_worker:
            self.start()

    def _reset_progress(self) -> None:
        self._progress_started_at = time.monotonic()
        self._progress_finished_at = None
        self._progress_initial_index = self.current_index

    def _revert_progress(self, indices: tuple[int, ...] | int):
        self.rewind(indices)

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

    def _ensure_archive(self) -> None:
        if not self._path.exists():
            combined = self._combine_sweep_and_result_templates()
            combined.attrs[_CURRENT_INDEX_KEY] = 0
            combined.to_zarr(
                cast(Any, self._zarr_store()),
                mode="w",
                zarr_format=self.zarr_format,
                encoding={
                    name: {
                        "chunks": (1,) * len(self._sweep_dims)
                        + tuple(max(1, size) for size in template.shape)
                    }
                    for name, template in self._result_template_ds.data_vars.items()
                },
                write_empty_chunks=False,
            )

    def _persist_checkpoint(self, index: int) -> None:
        zarr.open_group(str(self._path), mode="a").attrs.update(
            {**self.metadata, _CURRENT_INDEX_KEY: index}
        )

    def store_xarray(self, indices: tuple[int, ...], data: xr.Dataset) -> None:
        self._ensure_archive()
        point_coords = {
            dim: [self._sweep.coords[dim].values[index]]
            for dim, index in zip(self._sweep_dims, indices)
        }

        expanded = data.expand_dims(point_coords)

        # Coordinates are immutable archive schema. Write only payload arrays,
        # avoiding partial rewrites of coordinate chunks shared with valid points.
        expanded = expanded.drop_vars(list(expanded.coords))

        self._to_public_result_type(expanded).to_zarr(
            cast(Any, self._zarr_store()),
            region={
                dim: slice(index, index + 1)
                for dim, index in zip(self._sweep_dims, indices)
            },
            zarr_format=self.zarr_format,
        )

        self._update_metadata()
        checkpoint = self._linear_index_for_indices_tuple(indices) + 1
        self._persist_checkpoint(checkpoint)
        with self._control_lock:
            self._valid_until = checkpoint
            self._cursor_revision += 1
        self.last_measurement = self._to_public_result_type(data)
        self.last_measurement_queue.put_nowait(self.last_measurement)
        self._refresh_plots()

    def _refresh_plots(self) -> None:
        if hasattr(self, "update_plots"):
            with self._plot_lock:
                try:
                    self.update_plots()
                except Exception:
                    self.LOG.exception(
                        "Live plot update failed; acquired data is committed"
                    )

    def get_measurement_by_indices(
        self, indices: tuple[int, ...]
    ) -> xr.Dataset | xr.DataArray:
        if not hasattr(self, "_path"):
            self.LOG.error("Measurement not started yet")
            return xr.Dataset()

        self._normalize_point(indices)
        return self.result.isel(dict(zip(self._sweep_dims, indices)))

    def _update_metadata(self):
        try:
            zarr.open(str(self._path), mode="a").attrs.update(
                {
                    key: value
                    for key, value in self.metadata.items()
                    if key != _CURRENT_INDEX_KEY
                }
            )
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
        """Return results with uncommitted/rewound points masked out as missing.

        Partial integer/boolean results are promoted to represent missing values.
        The on-disk arrays retain their original dtype and raw historical tail.
        """
        checkpoint = zarr.open_group(str(self._path), mode="r").attrs[
            _CURRENT_INDEX_KEY
        ]
        ds = xr.open_dataset(
            self._path, engine="zarr", consolidated=False, mask_and_scale=False
        )
        if checkpoint < self.ntotal:
            valid = xr.DataArray(
                (np.arange(self.ntotal) < checkpoint).reshape(self._sweep_sizes),
                dims=self._sweep_dims,
            )
            ds = ds.where(valid)
        ds = self._apply_derived_xindexes(ds)
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
        if not self.notebook_ui:
            return
        from .notebook_integration import initialize_notebook, live_info, live_plot
        from .runner_widget import logging_redirect_widget

        initialize_notebook()
        live_info_elements = live_info(self)
        self.update_plots = live_plot(self)
        self._refresh_plots()

        self.LOG.debug("Using notebook live info integration")
        self._runner_widget = live_info_elements.widget
        self._ui_update = live_info_elements.ui_update
        self._ui_ctx = logging_redirect_widget(live_info_elements.widget, self.LOG)
        self._ui_ctx.__enter__()
        stop_updates = Event()

        def finish_wrapper():
            stop_updates.set()
            self._ui_thread.join()
            try:
                live_info_elements.ui_finish()
            finally:
                self._ui_ctx.__exit__(None, None, None)

        self._ui_finish = finish_wrapper

        def ui_updater():
            while not stop_updates.wait(update_interval):
                self._ui_update()

        self._ui_thread = threading.Thread(target=ui_updater, daemon=True)
        self._ui_thread.start()

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
