from __future__ import annotations

import abc
import asyncio
import concurrent.futures as concurrent
import inspect
import time
import traceback
import warnings
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, TypeAlias
from xmsr.measurement import Measurement

from xmsr.notebook_integration import in_ipynb, live_info, live_plot

FutureTypes: TypeAlias = concurrent.Future | asyncio.Future

if TYPE_CHECKING:
    import holoviews

    from ._types import ExecutorTypes


with suppress(ModuleNotFoundError):
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class BaseRunner(metaclass=abc.ABCMeta):
    def __init__(
        self,
        measurement: Measurement,
        *,
        executor: ExecutorTypes | None = None,
        shutdown_executor: bool = False,
        retries: int = 0,
        raise_if_retries_exceeded: bool = True,
    ):
        self.executor = (
            concurrent.ThreadPoolExecutor() if executor is None else executor
        )

        self._pending_tasks: dict[FutureTypes, int] = {}

        # if we instantiate our own executor, then we are also responsible
        # for calling 'shutdown'
        self.shutdown_executor = shutdown_executor or (executor is None)

        self.measurement = measurement

        # Timing
        self.start_time = time.time()
        self.end_time: float | None = None
        self._elapsed_function_time = 0

        # Error handling attributes
        self.retries = retries
        self.raise_if_retries_exceeded = raise_if_retries_exceeded
        self._to_retry: dict[int, int] = {}
        self._tracebacks: dict[int, str] = {}

    def _do_raise(self, e: Exception, pid: int) -> None:
        tb = self._tracebacks[pid]
        indices, values = self.measurement.current_point
        raise RuntimeError(
            "An error occured while evaluating "
            f'"measure(values={values}, indices={indices})". '
            f"See the traceback for details.:\n\n{tb}"
        ) from e

    def _process_futures(
        self,
        done_futs: set[FutureTypes],
    ) -> None:
        for fut in done_futs:
            pid = self._pending_tasks.pop(fut)
            try:
                result = fut.result()
                # total execution time
                t = time.time() - fut.start_time  # type: ignore[union-attr]
            except Exception as e:
                self._tracebacks[pid] = traceback.format_exc()
                self._to_retry[pid] = self._to_retry.get(pid, 0) + 1
                if self._to_retry[pid] > self.retries:
                    self._to_retry.pop(pid)
                    if self.raise_if_retries_exceeded:
                        self._do_raise(e, pid)
            else:
                self._elapsed_function_time += t
                self._to_retry.pop(pid, None)
                self._tracebacks.pop(pid, None)
                indices, _ = self.measurement._combinations[pid]
                self.measurement.store_ndarray(indices, result)
                self.measurement.current_index += 1

    def _get_futures(
        self,
    ) -> list[FutureTypes]:
        start_time = time.time()  # so we can measure execution time
        pid = self.measurement.current_index
        fut = self._submit(pid)
        fut.start_time = start_time  # type: ignore[attr-defined]
        self._pending_tasks[fut] = pid
        return [fut]

    def _remove_unfinished(self) -> list[FutureTypes]:
        # cancel any outstanding tasks
        remaining = list(self._pending_tasks.keys())
        for fut in remaining:
            fut.cancel()
        return remaining

    def _cleanup(self) -> None:
        if self.shutdown_executor:
            self.executor.shutdown(wait=True)
        self.end_time = time.time()

    @property
    def failed(self) -> set[int]:
        """Set of points ids that failed ``runner.retries`` times."""
        return set(self._tracebacks) - set(self._to_retry)

    @abc.abstractmethod
    def elapsed_time(self) -> float:
        """Return the total time elapsed since the runner
        was started.

        Is called in `overhead`.
        """

    def overhead(self) -> float:
        """Overhead of using Adaptive and the executor in percent.

        This is measured as ``100 * (1 - t_function / t_elapsed)``.

        Notes
        -----
        This includes the overhead of the executor that is being used.
        The slower your function is, the lower the overhead will be. The
        learners take ~5-50 ms to suggest a point and sending that point to
        the executor also takes about ~5 ms, so you will benefit from using
        Adaptive whenever executing the function takes longer than 100 ms.
        This of course depends on the type of executor and the type of learner
        but is a rough rule of thumb.
        """
        t_function = self._elapsed_function_time
        if t_function == 0:
            # When no function is done executing, the overhead cannot
            # reliably be determined, so 0 is the best we can do.
            return 0
        t_total = self.elapsed_time()
        return (1 - t_function / t_total) * 100

    @abc.abstractmethod
    def _submit(self, pid: int) -> FutureTypes:
        """Is called in `_get_futures`."""

    def _id_to_point(self, pid: int) -> Any:
        indices, values = self.measurement._combinations[pid]
        return values

    @property
    def tracebacks(self) -> list[tuple[int, str]]:
        return [(self._id_to_point(pid), tb) for pid, tb in self._tracebacks.items()]

    @property
    def to_retry(self) -> list[tuple[int, int]]:
        return [(self._id_to_point(pid), n) for pid, n in self._to_retry.items()]

    @property
    def pending_points(self) -> list[tuple[FutureTypes, Any]]:
        return [
            (fut, self._id_to_point(pid)) for fut, pid in self._pending_tasks.items()
        ]


class AsyncRunner(BaseRunner):

    def __init__(
        self,
        measurement: Measurement,
        *,
        autostart: bool = False,
        executor: ExecutorTypes | None = None,
        shutdown_executor: bool = False,
        ioloop=None,
        retries: int = 0,
        raise_if_retries_exceeded: bool = True,
    ) -> None:
        super().__init__(
            measurement,
            executor=executor,
            shutdown_executor=shutdown_executor,
            retries=retries,
            raise_if_retries_exceeded=raise_if_retries_exceeded,
        )
        self.ioloop = ioloop or asyncio.get_event_loop()
        self.started = asyncio.Event()
        self.running = asyncio.Event()

        # When the learned function is 'async def', we run it
        # directly on the event loop, and not in the executor.
        # The *whole point* of allowing learning of async functions is so that
        # the user can have more fine-grained control over the parallelism.
        if inspect.iscoroutinefunction(measurement.measure):
            if executor:  # user-provided argument
                raise RuntimeError(
                    "Cannot use an executor when learning an async function."
                )
            self.executor.shutdown()  # Make sure we don't shoot ourselves later

        self.measurement._start_measurement()

        self.task = self.ioloop.create_task(self._run())
        if autostart:
            self.pause_unpause()
        if in_ipynb() and not self.ioloop.is_running():
            warnings.warn(
                "The runner has been scheduled, but the asyncio "
                "event loop is not running! If you are "
                "in a Jupyter notebook, remember to run "
                "'adaptive.notebook_extension()'",
                stacklevel=2,
            )

    def _submit(self, pid: int) -> asyncio.Task | asyncio.Future:
        ioloop = self.ioloop
        # if inspect.iscoroutinefunction(self.measurement.measure):
        #     return ioloop.create_task(self.measurement.measure(pid))
        # else:
        return ioloop.run_in_executor(self.executor, self.measurement.step, pid)

    def status(self) -> str:
        """Return the runner status as a string.

        The possible statuses are: running, cancelled, failed, and finished.
        """
        try:
            self.task.result()
        except asyncio.CancelledError:
            return "cancelled"
        except asyncio.InvalidStateError:
            if not self.started.is_set():
                return "initialized"
            elif not self.running.is_set():
                return "paused"
            else:
                return "running"
        except Exception:
            return "failed"
        else:
            return "finished"

    def pause_unpause(self) -> None:
        """Pause or unpause the runner."""
        if self.running.is_set():
            self.running.clear()
        else:
            self.running.set()
            if not self.started.is_set():
                self.started.set()
                self.start_time = time.time()

    def progress(self) -> tuple[int, int]:
        """Return the progress as a tuple of
        (number of completed points, total number of points)."""
        return (self.measurement.current_index, self.measurement.ntotal)

    def cancel(self) -> None:
        """Cancel the runner.

        This is equivalent to calling ``runner.task.cancel()``.
        """
        self.task.cancel()

    def block_until_done(self) -> None:
        if in_ipynb():
            raise RuntimeError(
                "Cannot block the event loop when running in a Jupyter notebook."
                " Use `await runner.task` instead."
            )
        self.ioloop.run_until_complete(self.task)

    def live_plot(
        self,
        *,
        plotter: Callable[[Measurement], holoviews.Element] | None = None,
        update_interval: float = 2.0,
        name: str | None = None,
        normalize: bool = True,
    ) -> holoviews.DynamicMap:
        """Live plotting of the measurement data.

        Parameters
        ----------
        runner : `~adaptive.Runner`
        plotter : function
            A function that takes the measurement as a argument and returns a
            holoviews object. By default ``measurement.plot()`` will be called.
        update_interval : int
            Number of second between the updates of the plot.
        name : hasable
            Name for the `live_plot` task in `adaptive.active_plotting_tasks`.
            By default the name is None and if another task with the same name
            already exists that other `live_plot` is canceled.
        normalize : bool
            Normalize (scale to fit) the frame upon each update.

        Returns
        -------
        dm : `holoviews.core.DynamicMap`
            The plot that automatically updates every `update_interval`.
        """
        return live_plot(
            self,
            plotter=plotter,
            update_interval=update_interval,
            name=name,
            normalize=normalize,
        )

    def live_info(self, *, update_interval: float = 0.1) -> None:
        """Display live information about the runner.

        Returns an interactive ipywidget that can be
        visualized in a Jupyter notebook.
        """
        return live_info(self, update_interval=update_interval)

    async def _run(self) -> None:
        first_completed = asyncio.FIRST_COMPLETED

        await self.started.wait()

        try:
            while self.measurement.current_index < self.measurement.ntotal:
                await self.running.wait()
                futures = self._get_futures()
                done, _ = await asyncio.wait(futures, return_when=first_completed)  # type: ignore[arg-type,type-var]
                self._process_futures(done)

            self.measurement._finalize_measurement()
        finally:
            remaining = self._remove_unfinished()
            if remaining:
                await asyncio.wait(remaining)  # type: ignore[type-var]
            self._cleanup()

    def elapsed_time(self) -> float:
        """Return the total time elapsed since the runner
        was started."""
        if not self.started.is_set():
            return 0
        if self.task.done():
            end_time = self.end_time
            if end_time is None:
                # task was cancelled before it began
                assert self.task.cancelled()
                return 0
        else:
            end_time = time.time()
        return end_time - self.start_time


# Default runner
Runner = AsyncRunner
