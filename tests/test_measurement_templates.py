import shutil
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from xmsr import Measurement


class _NoUiMixin:
    def _live_info(self, **kwargs):
        class _Pbar:
            format_dict = {
                "n": 0,
                "total": 1,
                "percentage": 0.0,
                "elapsed": 0.0,
                "remaining": 0.0,
                "rate": 0.0,
                "unit": "it",
                "unit_scale": False,
                "unit_divisor": 1000,
                "ncols": None,
                "nrows": None,
            }

            def unpause(self):
                pass

            def update(self, _):
                pass

        self.pbar = _Pbar()
        self._ui_update = lambda: None
        self._ui_finish = lambda: None


class _DatasetAutoWrapMeasurement(_NoUiMixin, Measurement):
    target_directory = tempfile.gettempdir()
    filename = "xmsr_test_dataset_autowrap"
    overwrite = True
    timestamp = False
    with_coords = False

    sweep_template = xr.Dataset().assign_coords(
        x=np.arange(2),
        y=np.arange(3),
        power=("x", np.arange(2) ** 2),
    )
    result_template = xr.Dataset(
        {
            "trace": xr.DataArray(
                np.empty((4,), dtype=float), dims=("t",), coords={"t": np.arange(4)}
            ),
            "scalar": xr.DataArray(np.empty((), dtype=float), dims=()),
        }
    )

    def measure(self, values, indices, metadata):
        return np.arange(4, dtype=float), float(values["x"] + values["y"])


class _DataArrayAutoWrapMeasurement(_NoUiMixin, Measurement):
    target_directory = tempfile.gettempdir()
    filename = "xmsr_test_dataarray_autowrap"
    overwrite = True
    timestamp = False
    with_coords = False

    sweep_template = xr.Dataset().assign_coords(a=np.arange(2), b=np.arange(2))
    result_template = xr.DataArray(
        np.empty((3,), dtype=np.float32),
        dims=("f",),
        coords={"f": np.arange(3)},
        name="signal",
    )

    def measure(self, values, indices, metadata):
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)


class _WrongShapeMeasurement(_NoUiMixin, Measurement):
    target_directory = tempfile.gettempdir()
    filename = "xmsr_test_wrong_shape"
    overwrite = True
    timestamp = False
    with_coords = False

    sweep_template = xr.Dataset().assign_coords(a=np.arange(1))
    result_template = xr.DataArray(
        np.empty((3,), dtype=np.float32), dims=("f",), name="signal"
    )

    def measure(self, values, indices, metadata):
        return np.array([1.0, 2.0], dtype=np.float32)


def _run_blocking(measurement: Measurement):
    measurement.started.set()
    measurement.run()


def _cleanup(measurement: Measurement):
    if hasattr(measurement, "_path"):
        shutil.rmtree(Path(measurement._path), ignore_errors=True)


def test_dataset_template_autowrap_tuple_writes_expected_shapes():
    m = _DatasetAutoWrapMeasurement(overwrite=True)
    try:
        _run_blocking(m)
        result = m.result
        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {"trace", "scalar"}
        assert result["trace"].shape == (2, 3, 4)
        assert result["scalar"].shape == (2, 3)
    finally:
        _cleanup(m)


def test_dataarray_template_autowrap_array_returns_dataarray():
    m = _DataArrayAutoWrapMeasurement(overwrite=True)
    try:
        _run_blocking(m)
        result = m.result
        assert isinstance(result, xr.DataArray)
        assert result.name == "signal"
        assert result.shape == (2, 2, 3)
    finally:
        _cleanup(m)


def test_autowrap_shape_mismatch_raises_value_error():
    m = _WrongShapeMeasurement(overwrite=True)
    try:
        m.started.set()
        try:
            m.run()
            raised = False
        except ValueError:
            raised = True
        assert raised
    finally:
        _cleanup(m)
