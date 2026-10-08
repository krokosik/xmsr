import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest
import xarray as xr
import zarr

from xmsr import Measurement
from xmsr.shared import MeasurementStatus


class RecoverableMeasurement(Measurement):
    filename = "recoverable"
    timestamp = False
    with_coords = False
    sweep_template = xr.Dataset(
        coords={
            "gate": [-0.7, 0.2, 1.4],
            "mode": ["low", "high"],
            "gain": ("gate", [-1.4, 0.4, 2.8]),
        }
    )
    result_template = xr.Dataset(
        {"signal": xr.DataArray(np.int64(0)), "ok": xr.DataArray(True)}
    )

    def __init__(self, **kwargs):
        super().__init__(notebook_ui=False, **kwargs)
        self.prepare_calls = 0
        self.finish_calls = 0
        self.version = 0
        self.calls = []
        self.fail_at = None
        self.failed_once = False
        self.prepared = threading.Event()

    def prepare(self, metadata):
        self.prepare_calls += 1
        self.prepared.set()

    def finish(self, metadata):
        self.finish_calls += 1

    def measure(self, values, indices, metadata):
        index = self._linear_index_for_indices_tuple((indices["gate"], indices["mode"]))
        self.calls.append(index)
        if index == self.fail_at and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("Instrument timeout")
        return np.int64(self.version * 100 + index), True


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "Runner did not reach the expected state"
        threading.Event().wait(0.01)


@contextmanager
def worker(measurement):
    thread = threading.Thread(target=measurement.run, daemon=True)
    thread.start()
    try:
        assert measurement.prepared.wait(5)
        yield thread
    finally:
        if thread.is_alive():
            measurement.cancel()
        thread.join(5)
        assert not thread.is_alive()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_retry_keeps_worker_setup_and_checkpoint(zarr_format, tmp_path):
    measurement = RecoverableMeasurement(
        target_directory=str(tmp_path), zarr_format=zarr_format
    )
    measurement.fail_at = 3
    with worker(measurement) as thread:
        measurement.resume()
        wait_for(lambda: measurement.can_retry)
        assert not measurement.finished.is_set()
        assert measurement.current_index == 3
        assert zarr.open_group(str(measurement._path)).attrs["__CURRENT_INDEX__"] == 3
        assert zarr.open_group(str(measurement._path))["signal"].chunks == (1, 1)
        assert np.isnan(measurement.result.signal.values.flat[3])
        assert measurement.prepare_calls == 1
        assert measurement.finish_calls == 0
        measurement.resume()
        thread.join(5)
        assert measurement.status == MeasurementStatus.FINISHED
        assert measurement.calls == [0, 1, 2, 3, 3, 4, 5]
        assert measurement.prepare_calls == 1
        assert measurement.finish_calls == 1
        np.testing.assert_array_equal(
            measurement.result.signal.values.ravel(), np.arange(6)
        )
        assert measurement.result.signal.dtype == np.dtype("int64")
        assert measurement.result.ok.values.all()


def test_browse_then_rewind_after_error_remeasures_the_tail_without_prepare(tmp_path):
    measurement = RecoverableMeasurement(target_directory=str(tmp_path))
    measurement.fail_at = 3
    with worker(measurement) as thread:
        measurement.resume()
        wait_for(lambda: measurement.can_retry)
        measurement.select_point((0, 1))
        assert measurement.current_index == 3
        assert measurement.display_measurement.signal.item() == 1
        revision = measurement._cursor_revision
        with pytest.raises(RuntimeError, match="changed"):
            measurement.rewind(1, expected_revision=revision - 1)
        assert measurement.current_index == 3
        measurement.version = 1
        measurement.rewind((0, 1))
        assert measurement.status == MeasurementStatus.PAUSED
        assert measurement.current_index == 1
        assert np.isnan(measurement.result.signal.values.ravel()[1:]).all()
        measurement.resume()
        thread.join(5)
        assert measurement.calls == [0, 1, 2, 3, 1, 2, 3, 4, 5]
        assert measurement.prepare_calls == 1
        np.testing.assert_array_equal(
            measurement.result.signal.values.ravel(), [0, 101, 102, 103, 104, 105]
        )


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("rewind_to", [0, 3])
def test_rewind_checkpoint_survives_reopening_and_masks_raw_tail(
    tmp_path, zarr_format, rewind_to
):
    original = RecoverableMeasurement(
        target_directory=str(tmp_path), zarr_format=zarr_format
    )
    original.run()
    loaded = RecoverableMeasurement(
        target_directory=str(tmp_path), zarr_format=zarr_format
    )
    loaded._start_measurement()
    assert loaded.current_index == 6
    assert loaded.last_measurement.signal.item() == 5
    loaded.rewind(rewind_to)
    assert loaded.current_index == rewind_to
    np.testing.assert_array_equal(
        loaded.result.signal.values.ravel()[:rewind_to], np.arange(rewind_to)
    )
    assert np.isnan(loaded.result.signal.values.ravel()[rewind_to:]).all()
    raw = zarr.open_group(str(loaded._path), mode="r")
    np.testing.assert_array_equal(raw["signal"][:].ravel(), np.arange(6))
    assert raw.attrs["__CURRENT_INDEX__"] == rewind_to

    reopened = RecoverableMeasurement(
        target_directory=str(tmp_path), zarr_format=zarr_format
    )
    reopened.version = 2
    reopened.run()
    assert reopened.calls == list(range(rewind_to, 6))
    np.testing.assert_array_equal(
        reopened.result.signal.values.ravel(),
        np.concatenate((np.arange(rewind_to), 200 + np.arange(rewind_to, 6))),
    )


def test_pause_acknowledges_the_inflight_step_before_rewind(tmp_path):
    measurement = RecoverableMeasurement(target_directory=str(tmp_path))
    entered = threading.Event()
    release = threading.Event()
    acquire = measurement.measure

    def measure(values, indices, metadata):
        result = acquire(values, indices, metadata)
        if len(measurement.calls) == 3:
            entered.set()
            assert release.wait(5)
        return result

    measurement.measure = measure
    with worker(measurement) as thread:
        try:
            measurement.resume()
            assert entered.wait(5)
            measurement.pause()
            assert measurement.status == MeasurementStatus.PAUSING
            assert not measurement.paused.is_set()
            with pytest.raises(RuntimeError, match="paused"):
                measurement.rewind(1)
            release.set()
            wait_for(lambda: measurement.paused.is_set() and measurement.can_rewind)
            assert measurement.status == MeasurementStatus.PAUSED
            assert measurement.current_index == 3
            measurement.version = 1
            measurement.rewind(1, resume=True)
            thread.join(5)
            assert measurement.prepare_calls == 1
            np.testing.assert_array_equal(
                measurement.result.signal.values.ravel(), [0, 101, 102, 103, 104, 105]
            )
        finally:
            release.set()


@pytest.mark.parametrize("swallow_restart", [False, True])
def test_rewind_unwinds_manual_wait_without_committing_the_old_point(
    tmp_path, swallow_restart
):
    measurement = RecoverableMeasurement(target_directory=str(tmp_path))
    after_wait = threading.Event()
    waited = False
    acquire = measurement.measure

    def measure(values, indices, metadata):
        nonlocal waited
        result = acquire(values, indices, metadata)
        if not waited and len(measurement.calls) == 2:
            waited = True
            try:
                measurement.wait_for_manual_resume("Adjust instrument")
            except Exception:
                if not swallow_restart:
                    raise
            after_wait.set()
        return result

    measurement.measure = measure
    with worker(measurement) as thread:
        measurement.resume()
        wait_for(lambda: measurement._waiting_manual)
        assert measurement.status == MeasurementStatus.PAUSED
        measurement.version = 1
        measurement.rewind(0)
        wait_for(lambda: not measurement._in_step)
        assert after_wait.is_set() == swallow_restart
        assert measurement.current_index == 0
        assert measurement._valid_until == 0
        assert np.isnan(measurement.result.signal).all()
        measurement.resume()
        thread.join(5)
        assert measurement.prepare_calls == 1
        np.testing.assert_array_equal(
            measurement.result.signal.values.ravel(), 100 + np.arange(6)
        )


def test_plot_failure_does_not_undo_committed_acquisition(tmp_path):
    measurement = RecoverableMeasurement(target_directory=str(tmp_path))
    checkpoints = []

    def update_plots():
        checkpoints.append(
            zarr.open_group(str(measurement._path), mode="r").attrs["__CURRENT_INDEX__"]
        )
        raise RuntimeError("Broken plot")

    measurement.update_plots = update_plots
    measurement.run()
    assert checkpoints == list(range(1, 7))
    assert measurement.status == MeasurementStatus.FINISHED
    assert measurement.current_index == 6


@pytest.mark.parametrize("checkpoint", [None, -1, 7, "3", True])
def test_bad_archive_checkpoint_is_not_silently_restarted(tmp_path, checkpoint):
    original = RecoverableMeasurement(target_directory=str(tmp_path))
    original.run()
    root = zarr.open_group(str(original._path), mode="a")
    if checkpoint is None:
        del root.attrs["__CURRENT_INDEX__"]
    else:
        root.attrs["__CURRENT_INDEX__"] = checkpoint
    reopened = RecoverableMeasurement(target_directory=str(tmp_path))
    with pytest.raises(ValueError, match="checkpoint"):
        reopened._start_measurement()
    assert reopened.prepare_calls == 0
    assert reopened.calls == []


def test_changed_sweep_coordinates_are_rejected_on_reopen(tmp_path):
    original = RecoverableMeasurement(target_directory=str(tmp_path))
    original.run()

    class ChangedSweep(RecoverableMeasurement):
        sweep_template = xr.Dataset(
            coords={
                "gate": [-0.7, 0.5, 1.4],
                "mode": ["low", "high"],
                "gain": ("gate", [-1.4, 0.4, 2.8]),
            }
        )

    reopened = ChangedSweep(target_directory=str(tmp_path), filename=original.filename)
    with pytest.raises(ValueError, match="coordinate"):
        reopened._start_measurement()
    assert reopened.prepare_calls == 0


def test_checkpoint_write_failure_leaves_written_point_invalid_until_retry(
    tmp_path, monkeypatch
):
    measurement = RecoverableMeasurement(target_directory=str(tmp_path))
    persist = measurement._persist_checkpoint
    failed = False

    def fail_once(index):
        nonlocal failed
        if index == 3 and not failed:
            failed = True
            raise OSError("Checkpoint write failed")
        persist(index)

    monkeypatch.setattr(measurement, "_persist_checkpoint", fail_once)
    with worker(measurement) as thread:
        measurement.resume()
        wait_for(lambda: measurement.can_retry)
        assert measurement.current_index == 2
        raw = zarr.open_group(str(measurement._path), mode="r")
        assert raw["signal"][:].ravel()[2] == 2
        assert raw.attrs["__CURRENT_INDEX__"] == 2
        assert np.isnan(measurement.result.signal.values.ravel()[2:]).all()
        measurement.resume()
        thread.join(5)
        assert measurement.status == MeasurementStatus.FINISHED
        assert measurement.calls == [0, 1, 2, 2, 3, 4, 5]
        assert measurement.prepare_calls == 1


def test_failed_rewind_does_not_change_memory_or_disk_cursor(tmp_path, monkeypatch):
    original = RecoverableMeasurement(target_directory=str(tmp_path))
    original.run()
    loaded = RecoverableMeasurement(target_directory=str(tmp_path))
    loaded._start_measurement()

    def fail(index):
        raise OSError("Checkpoint write failed")

    monkeypatch.setattr(loaded, "_persist_checkpoint", fail)
    with pytest.raises(OSError, match="Checkpoint"):
        loaded.rewind(0)
    assert loaded.current_index == 6
    assert loaded._valid_until == 6
    np.testing.assert_array_equal(loaded.result.signal.values.ravel(), np.arange(6))


def test_recovery_before_first_committed_point_creates_archive_without_prepare(
    tmp_path,
):
    measurement = RecoverableMeasurement(target_directory=str(tmp_path))
    measurement.fail_at = 0
    with worker(measurement) as thread:
        measurement.resume()
        wait_for(lambda: measurement.can_retry)
        assert not measurement._path.exists()
        measurement.select_point(0)
        assert measurement.display_measurement is None
        measurement.rewind(0)
        assert np.isnan(measurement.display_measurement.signal.item())
        measurement.resume()
        thread.join(5)
        assert measurement.status == MeasurementStatus.FINISHED
        assert measurement.prepare_calls == 1


def test_blocking_execution_keeps_existing_manual_wait_behavior(tmp_path):
    measurement = RecoverableMeasurement(target_directory=str(tmp_path))
    acquire = measurement.measure

    def measure(values, indices, metadata):
        measurement.wait_for_manual_resume(
            "Operator pause requires background execution"
        )
        return acquire(values, indices, metadata)

    measurement.measure = measure
    measurement.run()
    assert measurement.status == MeasurementStatus.FINISHED
    assert measurement.prepare_calls == 1


def test_reopening_uses_the_archives_zarr_format(tmp_path):
    original = RecoverableMeasurement(target_directory=str(tmp_path), zarr_format=2)
    original.run()
    loaded = RecoverableMeasurement(target_directory=str(tmp_path))
    loaded._start_measurement()
    assert loaded.zarr_format == 2
    loaded.rewind(3)
    reopened = RecoverableMeasurement(target_directory=str(tmp_path))
    reopened.run()
    assert reopened.zarr_format == 2
    assert reopened.calls == [3, 4, 5]


def test_sweep_completion_remains_bounded_with_legacy_single_run_flag(tmp_path):
    measurement = RecoverableMeasurement(target_directory=str(tmp_path))
    measurement._is_single_run = False
    measurement.run()
    assert measurement.status == MeasurementStatus.FINISHED
    assert measurement.calls == list(range(6))


def test_blocking_error_can_rewind_and_run_again_without_prepare(tmp_path):
    measurement = RecoverableMeasurement(target_directory=str(tmp_path))
    measurement.fail_at = 3
    with pytest.raises(RuntimeError, match="timeout"):
        measurement.run()
    assert measurement.can_retry and measurement.can_rewind
    measurement.version = 1
    measurement.rewind(1)
    measurement.run()
    assert measurement.prepare_calls == 1
    assert measurement.finish_calls == 1
    assert measurement.status == MeasurementStatus.FINISHED
    np.testing.assert_array_equal(
        measurement.result.signal.values.ravel(), [0, 101, 102, 103, 104, 105]
    )


def test_resume_after_blocking_error_launches_worker_with_existing_setup(tmp_path):
    measurement = RecoverableMeasurement(target_directory=str(tmp_path))
    measurement.fail_at = 0
    with pytest.raises(RuntimeError, match="timeout"):
        measurement.run()
    measurement.resume()
    wait_for(lambda: measurement.finished.is_set() and not measurement._active)
    assert measurement.prepare_calls == 1
    assert measurement.finish_calls == 1
    assert measurement.status == MeasurementStatus.FINISHED


def test_cancel_before_first_point_leaves_a_reopenable_zero_checkpoint(tmp_path):
    measurement = RecoverableMeasurement(target_directory=str(tmp_path))
    with worker(measurement) as thread:
        measurement.cancel()
        thread.join(5)
        assert measurement.status == MeasurementStatus.CANCELLED
        assert (
            zarr.open_group(str(measurement._path), mode="r").attrs["__CURRENT_INDEX__"]
            == 0
        )
    reopened = RecoverableMeasurement(target_directory=str(tmp_path))
    reopened.run()
    assert reopened.calls == list(range(6))


def test_cancel_before_run_is_not_discarded_by_session_initialization(tmp_path):
    measurement = RecoverableMeasurement(target_directory=str(tmp_path))
    measurement.cancel()
    measurement.run()
    assert measurement.status == MeasurementStatus.CANCELLED
    assert measurement.calls == []
