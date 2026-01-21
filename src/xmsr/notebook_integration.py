import asyncio
import importlib
import logging
import os.path
import random
import sys
import warnings
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING

from IPython.core.getipython import get_ipython
from tqdm.notebook import tqdm_notebook

from xmsr.shared import LiveInfoElements, MeasurementStatus

if TYPE_CHECKING:
    from xmsr.measurement import Measurement

_holoviews_enabled = False
_ipywidgets_enabled = False
_kernel_do_step = None


def notebook_extension(*, _inline_js=True):
    """Enable ipywidgets, holoviews, and asyncio notebook integration."""
    global _holoviews_enabled, _ipywidgets_enabled, _kernel_do_step

    if not in_ipynb():
        return
    else:
        _kernel_do_step = get_ipython().kernel.do_one_iteration  # type: ignore[name-defined]

    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ModuleNotFoundError:
        warnings.warn(
            "nest_asyncio is not installed; asyncio may not work properly in the notebook.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Load holoviews
    try:
        _holoviews_enabled = False  # After closing a notebook the js is gone
        if not _holoviews_enabled:
            import holoviews

            holoviews.notebook_extension("bokeh", logo=False, inline=_inline_js)
            _holoviews_enabled = True
    except ModuleNotFoundError:
        warnings.warn(
            "holoviews is not installed; plotting is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Load ipywidgets
    try:
        if not _ipywidgets_enabled:
            import ipywidgets  # noqa: F401
            from IPython.display import HTML, display

            _ipywidgets_enabled = True

            # Transparent background for ipywidgets in VSCode Jupyter notebooks
            # display(
            #     HTML(
            #         """
            # <style>
            # /*overwrite hard coded write background by vscode for ipywidges */
            # .cell-output-ipywidget-background {
            #    background-color: transparent !important;
            # }

            # /*set widget foreground text and color of interactive widget to vs dark theme color */
            # :root {
            #     --jp-widgets-color: var(--vscode-editor-foreground);
            #     --jp-widgets-font-size: var(--vscode-editor-font-size);
            # }
            # </style>
            # """
            #     )
            # )
    except ModuleNotFoundError:
        warnings.warn(
            "ipywidgets is not installed; live_info is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )


def ensure_holoviews():
    try:
        return importlib.import_module("holoviews")
    except ModuleNotFoundError:
        raise RuntimeError(
            "holoviews is not installed; plotting is disabled."
        ) from None


def in_ipynb() -> bool:
    try:
        # If we are running in IPython, then `get_ipython()` is always a global
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"  # type: ignore[name-defined]
    except NameError:
        return False


# Fancy displays in the Jupyter notebook

active_plotting_tasks: dict[str, asyncio.Task] = {}


def live_plot(
    runner,
    *,
    step_plotter=None,
    full_plotter=None,
    update_interval=2,
    name=None,
    normalize=True,
):
    """Live plotting of the learner's data.

    Parameters
    ----------
    runner : `~adaptive.Runner`
    plotter : function
        A function that takes the learner as a argument and returns a
        holoviews object. By default ``learner.plot()`` will be called.
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
    if not _holoviews_enabled:
        warnings.warn(
            "Live plotting is not enabled; did you run 'poetry add xmsr[notebook]'?",
            RuntimeWarning,
        )
        return

    import holoviews as hv
    import ipywidgets
    from IPython.display import display

    if name in active_plotting_tasks:
        active_plotting_tasks[name].cancel()

    def plot_generator():
        while True:
            if not plotter:
                yield runner.learner.plot()
            else:
                yield plotter(runner.learner)

    streams = [hv.streams.Stream.define("Next")()]
    dm = hv.DynamicMap(plot_generator(), streams=streams)
    dm.cache_size = 1

    if normalize:
        # XXX: change when https://github.com/pyviz/holoviews/issues/3637
        # is fixed.
        dm = dm.map(lambda obj: obj.opts(framewise=True), hv.Element)

    cancel_button = ipywidgets.Button(
        description="cancel live-plot", layout=ipywidgets.Layout(width="150px")
    )

    # Could have used dm.periodic in the following, but this would either spin
    # off a thread (and measurement is not threadsafe) or block the kernel.

    async def updater():
        event = lambda: hv.streams.Stream.trigger(  # noqa: E731
            dm.streams
        )  # XXX: used to be dm.event()
        # see https://github.com/pyviz/holoviews/issues/3564
        try:
            while not runner.task.done():
                event()
                await asyncio.sleep(update_interval)
            event()  # fire off one last update before we die
        finally:
            if active_plotting_tasks[name] is asyncio.current_task():
                active_plotting_tasks.pop(name, None)
            cancel_button.layout.display = "none"  # remove cancel button

    def cancel(_):
        with suppress(KeyError):
            active_plotting_tasks[name].cancel()

    active_plotting_tasks[name] = runner.ioloop.create_task(updater())
    cancel_button.on_click(cancel)

    display(cancel_button)
    return dm


def should_update(status):
    try:
        # Get the length of the write buffer size
        buffer_size = len(status.comm.kernel.iopub_thread._events)

        # Make sure to only keep all the messages when the notebook
        # is viewed, this means 'buffer_size == 1'. However, when not
        # viewing the notebook the buffer fills up. When this happens
        # we decide to only add messages to it when a certain probability.
        # i.e. we're offline for 12h, with an update_interval of 0.5s,
        # and without the reduced probability, we have buffer_size=86400.
        # With the correction this is np.log(86400) / np.log(1.1) = 119.2
        return 1.1**buffer_size * random.random() < 1
    except Exception:
        # We catch any Exception because we are using a private API.
        return True


def live_info(
    measurement: "Measurement", *, on_toggle_pause=None, on_cancel=None
) -> LiveInfoElements | None:
    """Display live information about the runner.

    Returns an interactive ipywidget that can be
    visualized in a Jupyter notebook.
    """
    if not _ipywidgets_enabled:
        warnings.warn(
            "Live info is not enabled; did you run 'poetry add xmsr[notebook]'?",
            RuntimeWarning,
        )
        return

    import ipywidgets
    from IPython.display import display

    header = ipywidgets.HTML(
        value=f"<h3>{measurement.__class__.__name__}</h3>",
    )

    btn_layout = ipywidgets.Layout(width="100px")

    progress_bar = tqdm_notebook(
        initial=measurement.current_index,
        total=measurement.ntotal,
        display=False,
        ncols="100%",  # type: ignore[arg-type]
        dynamic_ncols=True,
    )

    run = ipywidgets.Button(
        description="Run", layout=btn_layout, disabled=on_toggle_pause is None
    )

    def on_run(_):
        if progress_bar.container.layout.visibility == "hidden":
            progress_bar.container.layout.visibility = "visible"
            progress_bar.displayed = True

        if on_toggle_pause is not None:
            on_toggle_pause()

        if run.description == "Pause":
            run.description = "Run"
        else:
            run.description = "Pause"
            progress_bar.unpause()

    run.on_click(on_run)

    cancel = ipywidgets.Button(
        description="Cancel", layout=btn_layout, disabled=on_cancel is None
    )
    cancel.on_click(lambda _: on_cancel() if on_cancel is not None else None)

    status = ipywidgets.HTML(value=_info_html(measurement))

    output = ipywidgets.Output(layout=ipywidgets.Layout(height="100px"))

    def ui_update():
        status.value = _info_html(measurement)
        progress_bar.update(measurement.current_index - progress_bar.n)
        run.description = "Run" if measurement.status == "paused" else "Pause"

    def ui_finish():
        progress_bar.close()
        status.value = _info_html(measurement)
        cancel.layout.display = "none"
        output.append_display_data(measurement.result)
        output.layout.height = None

    if measurement.status != MeasurementStatus.RUNNING:
        progress_bar.container.layout.visibility = "hidden"
    else:
        run.description = "Pause"
        progress_bar.displayed = True

    display(
        ipywidgets.VBox(
            (
                header,
                ipywidgets.HBox((run, cancel)),
                status,
                progress_bar.container,
                output,
            )
        )
    )

    return LiveInfoElements(
        ui_update=ui_update,
        ui_finish=ui_finish,
        output=output,
        run_btn=run,
        cancel_btn=cancel,
        pbar=progress_bar,
    )


def _table_row(i, key, value):
    """Style the rows of a table. Based on the default Jupyterlab table style."""
    style = "text-align: right; padding: 0.5em 2em; line-height: 1.0;"
    if i % 2 == 1:
        style += " background: var(--md-grey-100);"
    return f'<tr><th style="{style}">{key}</th><th style="{style}">{value}</th></tr>'


def _info_html(measurement: "Measurement") -> str:
    status = measurement.status

    color = {
        MeasurementStatus.INIT: "#808080",
        MeasurementStatus.PAUSED: "#00ffff",
        MeasurementStatus.CANCELLED: "#ffa500",
        MeasurementStatus.FAILED: "#ff4444",
        MeasurementStatus.RUNNING: "#4488ff",
        MeasurementStatus.FINISHED: "#00ff00",
    }[status]

    # overhead = runner.overhead()
    # red_level = max(0, min(int(255 * overhead / 100), 255))
    # overhead_color = f"#{red_level:02x}{255 - red_level:02x}{0:02x}"

    info = [
        ("status", f'<font color="{color}">{status.value}</font>'),
        # ("elapsed time", datetime.timedelta(seconds=runner.elapsed_time())),
        # ("overhead", f'<font color="{overhead_color}">{overhead:.2f}%</font>'),
    ]
    with suppress(Exception):
        info.append(("filename", str(measurement._path.name)))
        info.append(
            (
                "clipboard",
                f'<button style="display: inline-block;" onclick="navigator.clipboard.writeText(\'{measurement._path.name}\')">Copy Filename</button>'
                + f'<button style="display: inline-block; margin-left: 10px;" onclick="navigator.clipboard.writeText(\'{os.path.abspath(measurement._path)}\')">Copy Path</button>',
            )
        )

    table = "\n".join(_table_row(i, k, v) for i, (k, v) in enumerate(info))

    return f"""
        <table style="width: 100%;">
        {table}
        </table>
    """


class _OutputWidgetHandler(logging.Handler):
    """Custom logging handler sending logs to an output widget"""

    def __init__(self, out, *args, **kwargs):
        super(_OutputWidgetHandler, self).__init__(*args, **kwargs)
        self.out = out

    def emit(self, record):
        """Overload of logging.Handler method"""
        formatted_record = self.format(record)
        new_output = {
            "name": "stdout",
            "output_type": "stream",
            "text": formatted_record + "\n",
        }
        self.out.outputs = (new_output,) + self.out.outputs

    def clear_logs(self):
        """Clear the current logs"""
        self.out.clear_output()


def _is_console_logging_handler(handler):
    return isinstance(handler, logging.StreamHandler) and handler.stream in {
        sys.stdout,
        sys.stderr,
    }


def _get_first_found_console_logging_handler(handlers):
    for handler in handlers:
        if _is_console_logging_handler(handler):
            return handler


@contextmanager
def logging_redirect_ipywidgets(
    output_widget,
    loggers=None,
):
    """
    Context manager redirecting console logging to `output_widget.append_display_data()`, leaving
    other logging handlers (e.g. log files) unaffected.
    """
    if loggers is None:
        loggers = [logging.root]
    original_handlers_list = [logger.handlers for logger in loggers]
    try:
        for logger in loggers:
            out_handler = _OutputWidgetHandler(output_widget)
            orig_handler = _get_first_found_console_logging_handler(logger.handlers)
            if orig_handler is not None:
                out_handler.setFormatter(orig_handler.formatter)
                out_handler.stream = orig_handler.stream
            logger.handlers = [
                handler
                for handler in logger.handlers
                if not _is_console_logging_handler(handler)
            ] + [out_handler]
        yield
    finally:
        for logger, original_handlers in zip(loggers, original_handlers_list):
            logger.handlers = original_handlers
