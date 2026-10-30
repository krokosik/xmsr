import logging
import sys
import threading
import time
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


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline
        threading.Event().wait(0.01)


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
        if fail:
            assert widget.state["error"] == "Instrument failed"
            assert widget.state["can_retry"]
            assert measurement._ui_thread.is_alive()
            assert not widget._finished
            assert any(
                "Instrument failed" in record["message"] for record in widget.logs
            )
            send(widget, "cancel")
            assert widget.state["status"] == "cancelled"
        else:
            assert widget.state["completed"] == 3
            assert displayed[1].outputs
        assert not measurement._ui_thread.is_alive()
        assert measurement.LOG.handlers == original_handlers
        send(widget, "start")
        assert widget.state["status"] == ("cancelled" if fail else "finished")
    finally:
        for item in displayed:
            item.close()


def test_widget_browses_saved_data_and_rejects_stale_or_unsafe_rewinds(measurement):
    measurement.timestamp = False
    measurement.with_coords = False
    measurement.run()
    loaded = DemoMeasurement(
        target_directory=measurement.target_directory,
        timestamp=False,
        with_coords=False,
        notebook_ui=False,
    )
    loaded.prepared = threading.Event()
    loaded._start_measurement()
    widget = RunnerWidget(loaded)
    entered = threading.Event()
    release = threading.Event()
    acquire = loaded.measure

    def measure(values, indices, metadata):
        entered.set()
        assert release.wait(5)
        return acquire(values, indices, metadata)

    loaded.measure = measure
    try:
        assert widget.state["reopened"]
        assert widget.axes == [
            {"name": "point", "values": ["0", "1", "2"], "units": "", "numeric": True}
        ]
        widget._on_command(widget, {"command": "select", "indices": [1]}, [])
        assert widget.state["selected_indices"] == [1]
        assert widget.state["selection_recorded"]
        assert loaded.display_measurement.item() == 1
        assert loaded.current_index == 3
        widget._on_command(
            widget, {"command": "rewind", "indices": [1], "revision": -1}, []
        )
        assert "changed" in widget.state["command_error"]
        assert loaded.current_index == 3
        widget._on_command(
            widget,
            {"command": "rewind", "indices": [1], "revision": widget.state["revision"]},
            [],
        )
        assert entered.wait(5)
        assert loaded.current_index == 1
        assert loaded.running.is_set()
        assert np.isnan(loaded.result.values[1:]).all()
        widget._on_command(widget, {"command": "select", "indices": [0]}, [])
        assert "paused" in widget.state["command_error"]
        widget._on_command(widget, {"command": []}, [])
    finally:
        release.set()
        loaded.cancel()
        wait_for(lambda: not loaded._active)
        widget.close()


def test_background_error_keeps_widget_alive_and_recovers_without_prepare(
    measurement, monkeypatch
):
    from xmsr import notebook_integration

    displayed = []
    monkeypatch.setattr(
        notebook_integration, "display", lambda *items: displayed.extend(items)
    )
    monkeypatch.setattr(notebook_integration, "live_plot", lambda _: lambda: None)
    measurement.notebook_ui = True
    preparations = []
    acquire = measurement.measure
    prepare = measurement.prepare
    failed = False

    def prepare_once(metadata):
        preparations.append(True)
        prepare(metadata)

    def measure(values, indices, metadata):
        nonlocal failed
        if indices["point"] == 1 and not failed:
            failed = True
            raise RuntimeError("Recoverable timeout")
        return acquire(values, indices, metadata)

    measurement.prepare = prepare_once
    measurement.measure = measure
    thread = threading.Thread(target=measurement.run, daemon=True)
    original_handlers = list(measurement.LOG.handlers)
    thread.start()
    try:
        wait_for(lambda: len(displayed) >= 2)
        widget = displayed[0]
        send(widget, "start")
        wait_for(lambda: widget.state.get("can_retry"))
        assert widget.state["status"] == "failed"
        assert widget.state["can_navigate"]
        assert not widget._finished
        assert measurement._ui_thread.is_alive()
        widget._on_command(
            widget,
            {"command": "rewind", "indices": [0], "revision": widget.state["revision"]},
            [],
        )
        thread.join(5)
        assert not thread.is_alive()
        assert len(preparations) == 1
        assert widget.state["status"] == "finished"
        assert widget.state["error"] == ""
        assert widget._finished
        assert not measurement._ui_thread.is_alive()
        assert measurement.LOG.handlers == original_handlers
    finally:
        measurement.cancel()
        thread.join(5)
        for item in displayed:
            item.close()


def test_measure_prints_reach_widget_and_flush_on_error(measurement):
    widget = RunnerWidget(measurement)
    measurement._runner_widget = widget
    original_stdout, original_stderr = sys.stdout, sys.stderr

    def measure(*args):
        print("voltage:", 1.25)
        print("instrument warning", file=sys.stderr)
        print("partial", end="")
        raise RuntimeError("instrument failed")

    measurement.measure = measure
    try:
        with pytest.raises(RuntimeError, match="instrument failed"):
            measurement.step()
        widget.publish()
        assert [(r["message"], r["level"]) for r in widget.logs] == [
            ("voltage: 1.25", "info"),
            ("instrument warning", "error"),
            ("partial", "info"),
        ]
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr
    finally:
        widget.close()


def test_output_capture_isolates_threads_and_overlapping_contexts(monkeypatch):
    from xmsr._output_capture import capture_output

    console = StringIO()
    monkeypatch.setattr(sys, "stdout", console)
    entered = threading.Barrier(3)
    release = threading.Event()
    messages = [[], []]
    errors = []

    def worker(index):
        try:
            with capture_output(messages[index].append, messages[index].append):
                entered.wait(5)
                assert release.wait(5)
                print(f"worker {index}")
                with capture_output(messages[index].append, messages[index].append):
                    print("nested", end="")
                print("restored", flush=True)
        except BaseException as error:
            errors.append(error)

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for worker_thread in workers:
        worker_thread.start()
    try:
        entered.wait(5)
        print("unrelated notebook output")
    finally:
        release.set()
        for worker_thread in workers:
            worker_thread.join(5)
    assert not errors
    assert all(not worker_thread.is_alive() for worker_thread in workers)
    assert console.getvalue() == "unrelated notebook output\n"
    assert messages == [
        ["worker 0", "nested", "restored"],
        ["worker 1", "nested", "restored"],
    ]
    assert sys.stdout is console


def test_headless_measure_prints_stay_on_console(measurement, capsys):
    def measure(*args):
        print("headless measurement")
        return 0.0

    measurement.measure = measure
    measurement._last_measurement_shapes = None
    measurement.step()
    assert capsys.readouterr().out == "headless measurement\n"
