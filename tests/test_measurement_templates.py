import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest  # type: ignore[reportMissingImports]
import xarray as xr
import zarr

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


def _supports_zarr_v3() -> bool:
    major = int(zarr.__version__.split(".")[0])
    return major >= 3


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_dataset_template_autowrap_tuple_writes_expected_shapes(zarr_format: int):
    if zarr_format == 3 and not _supports_zarr_v3():
        pytest.skip("zarr format 3 tests require zarr>=3 in this environment")
    m = _DatasetAutoWrapMeasurement(overwrite=True, zarr_format=zarr_format)
    try:
        _run_blocking(m)
        result = m.result
        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {"trace", "scalar"}
        assert result["trace"].shape == (2, 3, 4)
        assert result["scalar"].shape == (2, 3)
    finally:
        _cleanup(m)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_dataarray_template_autowrap_array_returns_dataarray(zarr_format: int):
    if zarr_format == 3 and not _supports_zarr_v3():
        pytest.skip("zarr format 3 tests require zarr>=3 in this environment")
    m = _DataArrayAutoWrapMeasurement(overwrite=True, zarr_format=zarr_format)
    try:
        _run_blocking(m)
        result = m.result
        assert isinstance(result, xr.DataArray)
        assert result.name == "signal"
        assert result.shape == (2, 2, 3)
    finally:
        _cleanup(m)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_autowrap_shape_mismatch_raises_value_error(zarr_format: int):
    if zarr_format == 3 and not _supports_zarr_v3():
        pytest.skip("zarr format 3 tests require zarr>=3 in this environment")
    m = _WrongShapeMeasurement(overwrite=True, zarr_format=zarr_format)
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


class _DerivedCoordsExposureMeasurement(_NoUiMixin, Measurement):
    target_directory = tempfile.gettempdir()
    filename = "xmsr_test_derived_coords"
    overwrite = True
    timestamp = False
    with_coords = False

    x = np.arange(2)
    sweep_template = (
        xr.Dataset()
        .assign_coords(x=x)
        .assign_coords(
            x_gain=("x", 2 * x + 1),
            x_offset=("x", x - 3),
        )
    )
    result_template = xr.DataArray(
        np.empty((2,), dtype=np.float32), dims=("f",), name="signal"
    )

    def measure(self, values, indices, metadata):
        metadata["seen_x_gain"] = values["x_gain"]
        metadata["seen_x_offset"] = values["x_offset"]
        metadata["seen_x_gain_index"] = indices["x_gain"]
        return np.array([1.0, 2.0], dtype=np.float32)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_derived_coords_exposed_in_values_and_indices(zarr_format: int):
    if zarr_format == 3 and not _supports_zarr_v3():
        pytest.skip("zarr format 3 tests require zarr>=3 in this environment")
    m = _DerivedCoordsExposureMeasurement(overwrite=True, zarr_format=zarr_format)
    try:
        _run_blocking(m)
        assert m.metadata["seen_x_gain"] == 3
        assert m.metadata["seen_x_offset"] == -2
        assert m.metadata["seen_x_gain_index"] == 1
        result = m.result
        assert "x_gain" in result.xindexes
        assert "x_offset" in result.xindexes
    finally:
        _cleanup(m)


class _MetadataAppendMeasurement(_DataArrayAutoWrapMeasurement):
    def prepare(self, metadata: dict):
        metadata["started"] = True
        return super().prepare(metadata)

    def measure(self, values, indices, metadata):
        if "measure_count" not in metadata:
            metadata["measure_count"] = 0
        metadata["measure_count"] += 1
        return super().measure(values, indices, metadata)

    def finish(self, metadata: dict):
        metadata["finalized"] = True
        return super().finish(metadata)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_metadata_writes(zarr_format: int):
    if zarr_format == 3 and not _supports_zarr_v3():
        pytest.skip("zarr format 3 tests require zarr>=3 in this environment")
    m = _MetadataAppendMeasurement(overwrite=True, zarr_format=zarr_format)
    try:
        _run_blocking(m)
        assert (
            m.metadata["measure_count"]
            == m.sweep_template.sizes["a"] * m.sweep_template.sizes["b"]
        )
        assert m.metadata["started"] is True
        assert m.metadata["finalized"] is True
    finally:
        _cleanup(m)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_metadata_on_disk(zarr_format: int):
    if zarr_format == 3 and not _supports_zarr_v3():
        pytest.skip("zarr format 3 tests require zarr>=3 in this environment")
    m = _MetadataAppendMeasurement(overwrite=True, zarr_format=zarr_format)
    try:
        _run_blocking(m)
        attrs = m.result.attrs
        assert (
            attrs["measure_count"]
            == m.sweep_template.sizes["a"] * m.sweep_template.sizes["b"]
        )
        assert attrs["started"] is True
        assert attrs["finalized"] is True
    finally:
        _cleanup(m)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_metadata_on_resume(zarr_format: int):
    if zarr_format == 3 and not _supports_zarr_v3():
        pytest.skip("zarr format 3 tests require zarr>=3 in this environment")
    m = _MetadataAppendMeasurement(overwrite=True, zarr_format=zarr_format)
    try:
        _run_blocking(m)
        m = _MetadataAppendMeasurement(overwrite=False, zarr_format=zarr_format)
        _run_blocking(m)

        attrs = m.metadata
        assert (
            attrs["measure_count"]
            == m.sweep_template.sizes["a"] * m.sweep_template.sizes["b"]
        )
        assert attrs["started"] is True
        assert attrs["finalized"] is True
    finally:
        _cleanup(m)
