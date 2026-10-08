import gc
import weakref

import holoviews as hv
import numpy as np
import pytest
import xarray as xr

from xmsr import Measurement, notebook_integration
from xmsr.qt_integration import Thread


class PlottedMeasurement(Measurement):
    sweep_template = xr.Dataset(coords={"point": np.arange(3)})
    result_template = xr.DataArray(
        np.empty(2), dims="frequency", coords={"frequency": [1, 2]}, name="signal"
    )

    def measure(self, values, indices, metadata):
        return np.array([values["point"], values["point"] + 1.0])

    def plot_single_step(self, data):
        return data.hvplot.line()

    def plot_preview(self, data):
        return data.mean("point").hvplot.line()


@pytest.mark.parametrize("notebook_ui", [False, True])
def test_plotting_resources_are_initialized_before_worker_start(
    monkeypatch, notebook_ui
):
    calls = []
    monkeypatch.setattr(
        notebook_integration, "initialize_notebook", lambda: calls.append("resources")
    )
    monkeypatch.setattr(Thread, "start", lambda self: calls.append("worker"))
    measurement = PlottedMeasurement(notebook_ui=notebook_ui)
    measurement._ipython_display_()
    assert calls == (["resources", "worker"] if notebook_ui else ["worker"])


def test_live_maps_survive_garbage_collection_and_reuse_streams(tmp_path, monkeypatch):
    measurement = PlottedMeasurement(target_directory=str(tmp_path), notebook_ui=False)
    measurement._start_measurement()
    maps = []
    monkeypatch.setattr(
        notebook_integration, "display", lambda obj: maps.append(weakref.ref(obj))
    )
    update = notebook_integration.live_plot(measurement)
    measurement.update_plots = update
    indices, first = measurement.step(0)
    measurement.store_xarray(indices, first)
    gc.collect()
    assert len(maps) == 2
    assert all(ref() is not None for ref in maps)

    indices, second = measurement.step(1)
    measurement.store_xarray(indices, second)
    assert len(maps) == 2  # Update existing views instead of redisplaying them.
    np.testing.assert_array_equal(maps[0]()[()].dimension_values("signal"), [1, 2])
    np.testing.assert_array_equal(maps[1]()[()].dimension_values("signal"), [0.5, 1.5])


def test_measurement_updates_both_rendered_bokeh_plots(tmp_path, monkeypatch):
    displayed = []
    plots = []

    def display(*objects):
        for obj in objects:
            if isinstance(obj, hv.DynamicMap):
                plots.append(hv.renderer("bokeh").get_plot(obj))
            else:
                displayed.append(obj)

    monkeypatch.setattr(notebook_integration, "display", display)
    measurement = PlottedMeasurement(target_directory=str(tmp_path))
    try:
        measurement.run()
        assert len(plots) == 2
        np.testing.assert_array_equal(plots[0].handles["source"].data["signal"], [2, 3])
        np.testing.assert_array_equal(plots[1].handles["source"].data["signal"], [1, 2])
    finally:
        for obj in displayed:
            obj.close()
