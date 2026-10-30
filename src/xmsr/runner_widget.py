"""Python state/command adapter for the React notebook runner."""

import logging
import sys
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Iterator

import anywidget
import traitlets

from ._output_capture import capture_output
from .shared import MeasurementStatus

if TYPE_CHECKING:
    from .measurement import Measurement

_STATIC = Path(__file__).parent / "static"
_TERMINAL = {
    MeasurementStatus.FINISHED,
    MeasurementStatus.CANCELLED,
}


class RunnerWidget(anywidget.AnyWidget):
    """Expose measurement snapshots and dispatch explicit frontend commands.

    Args:
        measurement (Measurement): Python-owned measurement engine.
        max_logs (int): Maximum number of retained log records.
    """

    _esm = _STATIC / "runner.js"
    _css = _STATIC / "runner.css"
    state = traitlets.Dict().tag(sync=True)
    logs = traitlets.List(traitlets.Dict()).tag(sync=True)
    axes = traitlets.List(traitlets.Dict()).tag(sync=True)

    def __init__(self, measurement: "Measurement", max_logs: int = 500):
        if max_logs < 1:
            raise ValueError("max_logs must be positive")
        self.measurement = measurement
        self._lock = RLock()
        self._records: deque[dict[str, Any]] = deque(maxlen=max_logs)
        self._log_id = 0
        self._finished = False
        self._command_error = ""
        self._snapshot_sequence = 0
        self._published_sequence = 0
        super().__init__()
        self.axes = [
            {
                "name": dim,
                "values": [
                    str(value) for value in measurement._sweep.coords[dim].values
                ],
                "units": str(measurement._sweep.coords[dim].attrs.get("units", "")),
                "numeric": measurement._sweep.coords[dim].dtype.kind in "iuf",
            }
            for dim in measurement._sweep_dims
        ]
        self.on_msg(self._on_command)
        self.publish()

    def publish(self) -> None:
        """Publish one consistent state snapshot and bounded log history."""
        if self._finished:
            return
        measurement = self.measurement
        # Snapshot the engine before acquiring the widget lock. Engine commands
        # can emit logs, which need the widget lock in the opposite direction.
        with measurement._control_lock:
            self._snapshot_sequence += 1
            sequence = self._snapshot_sequence
            path = getattr(measurement, "_path", None)
            progress = measurement._get_progress_dict()
            selected = measurement.selected_index
            selected_indices = measurement._indices_tuple_for_linear_index(selected)
            selected_values, _ = measurement._values_and_indices_for_indices_tuple(
                selected_indices
            )
            error = getattr(measurement, "_exception", None)
            state = {
                "name": measurement.__class__.__name__,
                "status": measurement.status.value,
                "completed": progress["n"],
                "total": progress["total"],
                "elapsed": progress["elapsed"],
                "remaining": progress["remaining"] if progress["rate"] else None,
                "rate": progress["rate"],
                "filename": path.name if path else "",
                "path": str(path.resolve()) if path else "",
                "error": str(error) if error is not None else "",
                "can_navigate": measurement.can_rewind,
                "can_retry": measurement.can_retry,
                "selected_index": selected,
                "selected_indices": list(selected_indices),
                "selection_recorded": selected < measurement._valid_until,
                "selected_values": {
                    key: str(value) for key, value in selected_values.items()
                },
                "revision": measurement._cursor_revision,
                "reopened": measurement._resumed_from_disk,
            }
        with self._lock:
            if self._finished or sequence <= self._published_sequence:
                return
            self._published_sequence = sequence
            self.state = {**state, "command_error": self._command_error}
            self.logs = list(self._records)

    def finish(self) -> None:
        """Publish the final snapshot and detach command handling."""
        self.publish()
        with self._lock:
            self._finished = True
            self.on_msg(self._on_command, remove=True)

    def append_log(self, record: logging.LogRecord) -> None:
        """Buffer a structured log record for the next publication."""
        with self._lock:
            self._log_id += 1
            self._records.append(
                {
                    "id": self._log_id,
                    "level": record.levelname.lower(),
                    "time": logging.Formatter(datefmt="%H:%M:%S").formatTime(
                        record, "%H:%M:%S"
                    ),
                    "message": logging.Formatter().format(record),
                }
            )

    def _on_command(self, widget: Any, content: Any, buffers: Any) -> None:
        if not isinstance(content, dict):
            return
        with self._lock:
            if self._finished or self.measurement.status in _TERMINAL:
                return
            self._command_error = ""
        command = content.get("command")
        if not isinstance(command, str):
            return
        status = self.measurement.status
        # Commands may log or publish. Do not hold the widget lock while calling
        # the engine, which also has an acquisition/control lock.
        try:
            if command == "cancel":
                self.measurement.cancel()
            elif (
                (command == "start" and status == MeasurementStatus.INIT)
                or (command == "pause" and status == MeasurementStatus.RUNNING)
                or (command == "resume" and status == MeasurementStatus.PAUSED)
            ):
                if command == "pause":
                    self.measurement.pause()
                else:
                    self.measurement.resume()
            elif command == "retry" and self.measurement.can_retry:
                self.measurement.resume()
            elif command == "show_latest":
                self.measurement.select_point(None)
            elif command in {"select", "rewind"}:
                indices = content.get("indices")
                if not isinstance(indices, list):
                    raise ValueError(
                        "Provide one integer index for each sweep dimension"
                    )
                if command == "select":
                    self.measurement.select_point(tuple(indices))
                else:
                    revision = content.get("revision")
                    if isinstance(revision, bool) or not isinstance(revision, int):
                        raise ValueError(
                            "A rewind requires the current runner revision"
                        )
                    self.measurement.rewind(
                        tuple(indices), expected_revision=revision, resume=True
                    )
        except (ValueError, RuntimeError, OSError) as error:
            with self._lock:
                self._command_error = str(error)
        self.publish()


class _WidgetLogHandler(logging.Handler):
    def __init__(self, widget: RunnerWidget):
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        self.widget.append_log(record)


class _ExcludeLoggerFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.name == self.name or record.name.startswith(f"{self.name}."))


@contextmanager
def logging_redirect_widget(
    widget: RunnerWidget, logger: logging.Logger
) -> Iterator[None]:
    """Temporarily redirect this logger's console handlers to structured logs.

    File handlers and handlers added during execution are preserved.
    """
    console_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and handler.stream in (sys.stdout, sys.stderr)
    ]
    # Suppress duplicate ancestor-console output without disabling propagation
    # to file handlers configured on the root/parent loggers.
    ancestor_consoles = []
    ancestor = logger
    while ancestor.propagate and ancestor.parent is not None:
        ancestor = ancestor.parent
        ancestor_consoles.extend(
            handler
            for handler in ancestor.handlers
            if isinstance(handler, logging.StreamHandler)
            and handler.stream in (sys.stdout, sys.stderr)
        )
    console_filter = _ExcludeLoggerFilter(logger.name)
    for console_handler in ancestor_consoles:
        console_handler.addFilter(console_filter)
    handler = _WidgetLogHandler(widget)
    for console_handler in console_handlers:
        logger.removeHandler(console_handler)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
        for console_handler in console_handlers:
            logger.addHandler(console_handler)
        for console_handler in ancestor_consoles:
            console_handler.removeFilter(console_filter)


@contextmanager
def capture_measurement_output(widget: RunnerWidget | None) -> Iterator[None]:
    """Send the measurement thread's stdout/stderr to the notebook log box."""
    if widget is None:
        yield
        return

    def emit(message: str, level: int) -> None:
        widget.append_log(
            logging.LogRecord("measurement.output", level, "", 0, message, (), None)
        )

    with capture_output(
        lambda message: emit(message, logging.INFO),
        lambda message: emit(message, logging.ERROR),
    ):
        yield
