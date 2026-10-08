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

from .shared import MeasurementStatus

if TYPE_CHECKING:
    from .measurement import Measurement

_STATIC = Path(__file__).parent / "static"
_TERMINAL = {
    MeasurementStatus.FINISHED,
    MeasurementStatus.CANCELLED,
    MeasurementStatus.FAILED,
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

    def __init__(self, measurement: "Measurement", max_logs: int = 500):
        if max_logs < 1:
            raise ValueError("max_logs must be positive")
        self.measurement = measurement
        self._lock = RLock()
        self._records: deque[dict[str, Any]] = deque(maxlen=max_logs)
        self._log_id = 0
        self._finished = False
        super().__init__()
        self.on_msg(self._on_command)
        self.publish()

    def publish(self) -> None:
        """Publish one consistent state snapshot and bounded log history."""
        with self._lock:
            if self._finished:
                return
            measurement = self.measurement
            path = getattr(measurement, "_path", None)
            progress = measurement._get_progress_dict()
            self.state = {
                "name": measurement.__class__.__name__,
                "status": measurement.status.value,
                "completed": progress["n"],
                "total": progress["total"],
                "elapsed": progress["elapsed"],
                "remaining": progress["remaining"] if progress["rate"] else None,
                "rate": progress["rate"],
                "filename": path.name if path else "",
                "path": str(path.resolve()) if path else "",
                "error": str(measurement._exception)
                if hasattr(measurement, "_exception")
                else "",
            }
            self.logs = list(self._records)

    def finish(self) -> None:
        """Publish the final snapshot and detach command handling."""
        with self._lock:
            self.publish()
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
            command = content.get("command")
            status = self.measurement.status
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
