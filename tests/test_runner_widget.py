import logging
import sys
import threading
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from xmsr import Measurement
from xmsr.runner_widget import RunnerWidget, logging_redirect_widget
from xmsr.shared import MeasurementStatus


class DemoMeasurement(Measurement):
    sweep_template = xr.Dataset(coords={"point": np.arange(3)})
    result_template = xr.DataArray(0.0, name="signal")

    def prepare(self, metadata):
        self.prepared.set()

    def measure(self, values, indices, metadata):
        return float(values["point"])


@pytest.fixture
def measurement(tmp_path: Path) -> DemoMeasurement:
    measurement = DemoMeasurement(target_directory=str(tmp_path), notebook_ui=False)
    measurement.prepared = threading.Event()
    measurement.current_index = 0
    return measurement


def send(widget: RunnerWidget, command: str) -> None:
    widget._on_command(widget, {"command": command}, [])


def test_explicit_commands_are_idempotent_and_terminal_states_are_guarded(measurement):
    widget = RunnerWidget(measurement)
    try:
        send(widget, "resume")
        assert measurement.status == MeasurementStatus.INIT
        send(widget, "start")
        assert widget.state["status"] == "running"
        send(widget, "start")
        assert measurement.running.is_set()
        send(widget, "pause")
        send(widget, "pause")
        assert widget.state["status"] == "paused"
        send(widget, "resume")
        assert widget.state["status"] == "running"
        send(widget, "cancel")
        send(widget, "start")
        assert widget.state["status"] == "cancelled"
        widget._on_command(widget, None, [])
        send(widget, "unknown")
        assert measurement.status == MeasurementStatus.CANCELLED
    finally:
        widget.close()


def test_progress_is_python_owned_and_resume_rate_counts_only_new_steps(
    measurement, monkeypatch
):
    monkeypatch.setattr("xmsr.measurement.time.monotonic", lambda: 14.0)
    measurement._progress_started_at = 10.0
    measurement._progress_initial_index = 1
    measurement.current_index = 2
    progress = measurement._get_progress_dict()
    assert progress["elapsed"] == 4
    assert progress["rate"] == 0.25
    assert progress["remaining"] == 4
    assert progress["indices"] == (2,)
    assert not hasattr(measurement, "pbar")
    measurement._progress_finished_at = 12.0
    assert measurement._get_progress_dict()["elapsed"] == 2


@pytest.mark.parametrize("pause_first", [False, True])
def test_cancel_wakes_measurement_waiting_for_start_or_resume(measurement, pause_first):
    if pause_first:
        measurement.started.set()
        entered = threading.Event()
        paused = threading.Event()
        measurement._ui_update = (
            lambda: paused.set() if not measurement.running.is_set() else None
        )

        def measure(values, indices, metadata):
            entered.set()
            measurement.wait_for_manual_resume("Adjust instrument")
            return 0.0

        measurement.measure = measure
    worker = threading.Thread(target=measurement.run, daemon=True)
    worker.start()
    try:
        assert measurement.prepared.wait(5)
        if pause_first:
            assert entered.wait(5)
            assert paused.wait(5)
        measurement.cancel()
        worker.join(5)
        assert not worker.is_alive()
        assert measurement.status == MeasurementStatus.CANCELLED
    finally:
        measurement.cancel()
        worker.join(5)


def test_bounded_structured_logs_and_handler_restoration(measurement):
    widget = RunnerWidget(measurement, max_logs=2)
    logger = logging.getLogger("xmsr-test-logs")
    console = logging.StreamHandler(sys.stdout)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    added = logging.NullHandler()
    inherited_output = StringIO()
    inherited_handler = logging.StreamHandler(inherited_output)
    logging.root.addHandler(inherited_handler)
    try:
        with pytest.raises(RuntimeError), logging_redirect_widget(widget, logger):
            logger.info("old")
            logger.warning("<instrument>")
            logger.error("new")
            logger.addHandler(added)
            raise RuntimeError("test cleanup")
        widget.publish()
        assert [record["message"] for record in widget.logs] == ["<instrument>", "new"]
        assert widget.logs[0]["level"] == "warning"
        assert console in logger.handlers
        assert added in logger.handlers
        assert logger.propagate
        assert "<instrument>" in inherited_output.getvalue()
        assert "new" in inherited_output.getvalue()
    finally:
        logger.removeHandler(console)
        logger.removeHandler(added)
        logging.root.removeHandler(inherited_handler)
        widget.close()


def test_pause_immediately_after_start_is_not_overwritten_by_worker_startup(
    measurement,
):
    measurement.resume()
    measurement.pause()
    startup_paused = threading.Event()
    measurement._ui_update = (
        lambda: startup_paused.set() if not measurement.running.is_set() else None
    )
    worker = threading.Thread(target=measurement.run, daemon=True)
    worker.start()
    try:
        assert startup_paused.wait(5)
        assert measurement.current_index == 0
        assert measurement.status == MeasurementStatus.PAUSED
        measurement.resume()
        worker.join(5)
        assert not worker.is_alive()
        assert measurement.status == MeasurementStatus.FINISHED
        assert measurement.current_index == 3
    finally:
        if worker.is_alive():
            measurement.cancel()
            worker.join(5)


@pytest.mark.parametrize("fail", [False, True])
def test_notebook_lifecycle_publishes_final_state_and_stops_updates(
    measurement, monkeypatch, fail
):
    from xmsr import notebook_integration

    displayed = []
    monkeypatch.setattr(
        notebook_integration, "display", lambda *items: displayed.extend(items)
    )
    monkeypatch.setattr(notebook_integration, "live_plot", lambda _: lambda: None)
    measurement.notebook_ui = True
    measurement.started.set()
    if fail:

        def measure(*args):
            raise ValueError("Instrument failed")

        measurement.measure = measure
    original_handlers = list(measurement.LOG.handlers)
    try:
        if fail:
            with pytest.raises(ValueError, match="Instrument failed"):
                measurement.run()
        else:
            measurement.run()
        widget = displayed[0]
        assert widget.state["status"] == ("failed" if fail else "finished")
        assert not measurement._ui_thread.is_alive()
        assert measurement.LOG.handlers == original_handlers
        if fail:
            assert widget.state["error"] == "Instrument failed"
            assert any(
                "Instrument failed" in record["message"] for record in widget.logs
            )
        else:
            assert widget.state["completed"] == 3
            assert displayed[1].outputs
        send(widget, "start")
        assert widget.state["status"] == ("failed" if fail else "finished")
    finally:
        for item in displayed:
            item.close()
