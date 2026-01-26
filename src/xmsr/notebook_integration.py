import logging
import os.path
import sys
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Callable

import holoviews as hv
import hvplot.xarray  # noqa: F401
import ipywidgets as widgets
import xarray as xr
from holoviews.streams import Pipe
from IPython.display import HTML, display
from tqdm.notebook import tqdm_notebook

from xmsr.shared import LiveInfoElements, LivePlotElements, MeasurementStatus

if TYPE_CHECKING:
    from xmsr.measurement import Measurement

hv.notebook_extension("bokeh", logo=False, inline=True)

# Transparent background for ipywidgets in VSCode Jupyter notebooks
display(
    HTML(
        """
<style>
/*overwrite hard coded write background by vscode for ipywidges */
.cell-output-ipywidget-background {
   background-color: transparent !important;
}

/*set widget foreground text and color of interactive widget to vs dark theme color */
:root {
    /* Layout and background colors */
    --jp-layout-color0: var(--vscode-editor-background);
    --jp-layout-color1: var(--vscode-editorWidget-background);
    --jp-layout-color2: var(--vscode-input-background);
    --jp-layout-color3: var(--vscode-dropdown-background);

    /* Inverse layout colors */
    --jp-inverse-layout-color0: var(--vscode-button-background);
    --jp-inverse-layout-color1: var(--vscode-button-hoverBackground);

    /* Font and text colors */
    --jp-content-font-color0: var(--vscode-editor-foreground);
    --jp-content-font-color1: var(--vscode-descriptionForeground);
    --jp-content-font-color2: var(--vscode-textSeparator-foreground);
    --jp-content-font-color3: var(--vscode-textLink-foreground);
    --jp-ui-font-color0: var(--vscode-foreground);
    --jp-ui-font-color1: var(--vscode-descriptionForeground);

    /* Border and accent colors */
    --jp-border-color1: var(--vscode-focusBorder);
    --jp-border-color2: var(--vscode-editorWidget-border);
    --jp-brand-color0: var(--vscode-button-background);
    --jp-brand-color1: var(--vscode-button-hoverBackground);
    --jp-brand-color2: var(--vscode-badge-background);
    --jp-accent-color0: var(--vscode-progressBar-background);
    --jp-accent-color1: var(--vscode-textLink-activeForeground);
    --jp-accent-color2: var(--vscode-list-highlightForeground);

    /* Status colors */
    --jp-warn-color0: var(--vscode-editorWarning-foreground);
    --jp-error-color0: var(--vscode-editorError-foreground);
    --jp-success-color0: var(--vscode-charts-green);
    --jp-info-color0: var(--vscode-editorInfo-foreground);

    /* Cell and editor integration */
    --jp-cell-padding: 4px;
    --jp-cell-editor-background: var(--vscode-editor-background);
    --jp-cell-editor-border-color: var(--vscode-focusBorder);
    --jp-cell-editor-background-edit: var(--vscode-editor-selectionBackground);
    --jp-cell-editor-border-color-edit: var(--vscode-focusBorder);
    --jp-cell-prompt-width: 0px;

    /* Fonts and sizing */
    --jp-ui-font-scale-factor: 1;
    --jp-ui-font-size0: var(--vscode-font-size);
    --jp-ui-font-size1: var(--vscode-editor-font-size);
    --jp-code-font-size: var(--vscode-editor-font-size);
    --jp-code-line-height: 1.5;
    --jp-code-padding: 4px;
    --jp-code-font-family: var(--vscode-editor-font-family);

    /* Widget-specific overrides */
    --jp-widgets-color: var(--vscode-editor-foreground);
    --jp-widgets-font-size: var(--vscode-editor-font-size);
    --jp-widgets-label-color: var(--vscode-editor-foreground);
    --jp-widgets-readout-color: var(--vscode-descriptionForeground);
    --jp-widgets-slider-handle-border-color: var(--vscode-focusBorder);
    --jp-widgets-slider-handle-background-color: var(--vscode-button-background);
    --jp-widgets-slider-active-handle-color: var(--vscode-button-hoverBackground);
}
</style>
"""
    )
)


def live_plot(measurement: "Measurement") -> Callable[[], None]:
    try:
        measurement.plot_single_step(xr.DataArray(0))  # test if plotter works
    except NotImplementedError:
        use_step_plot = False
    except Exception:
        use_step_plot = True
    else:
        use_step_plot = True

    try:
        measurement.plot_preview(xr.DataArray(0))
    except NotImplementedError:
        use_full_plot = False
    except Exception:
        use_full_plot = True
    else:
        use_full_plot = True

    step_pipe = None
    step_dmap = None

    full_pipe = None
    full_dmap = None

    to_display = []

    def update_plots():
        nonlocal step_pipe, step_dmap, full_pipe, full_dmap

        step_da = measurement.last_measurement
        if use_step_plot and step_da is not None:
            if step_pipe is None:
                step_pipe = Pipe(data=step_da)
                step_dmap = hv.DynamicMap(
                    measurement.plot_single_step, streams=[step_pipe]
                )
                to_display.append(step_dmap)
            elif step_da is not None and step_pipe is not None:
                step_pipe.send(step_da)

        full_da = measurement.result
        if use_full_plot and full_da is not None:
            if full_pipe is None:
                full_pipe = Pipe(data=full_da)
                full_dmap = hv.DynamicMap(measurement.plot_preview, streams=[full_pipe])
                to_display.append(full_dmap)
            elif full_da is not None and full_pipe is not None:
                full_pipe.send(full_da)

        if len(to_display) > 0:
            layout = hv.Layout(to_display).cols(1)
            if hasattr(measurement, "live_plot_opts"):
                layout = layout.opts(measurement.live_plot_opts)
            display(layout)
            to_display.clear()

    return update_plots


def live_info(
    measurement: "Measurement", *, on_toggle_pause=None, on_cancel=None
) -> LiveInfoElements:
    """Display live information about the runner.

    Returns an interactive ipywidget that can be
    visualized in a Jupyter notebook.
    """

    header = widgets.HTML(
        value=f'<h3 style="margin: 0.5em 0">{measurement.__class__.__name__}</h3>',
    )

    btn_layout = widgets.Layout(width="100px")

    progress_bar = tqdm_notebook(
        initial=measurement.current_index,
        total=measurement.ntotal,
        display=False,
        ncols="100%",  # type: ignore[arg-type]
        dynamic_ncols=True,
    )

    run = widgets.Button(
        description="Start", layout=btn_layout, disabled=on_toggle_pause is None
    )

    cancel = widgets.Button(icon="stop", layout=btn_layout, disabled=True)
    cancel.on_click(lambda _: on_cancel() if on_cancel is not None else None)

    def run_update():
        status = measurement.status

        if status == MeasurementStatus.PAUSED:
            run.icon = "play"
        elif run.icon != "pause" and status == MeasurementStatus.RUNNING:
            run.icon = "pause"
            progress_bar.unpause()
        else:
            run.disabled = True
            cancel.disabled = True

    def on_run(_):
        if not progress_bar.displayed:
            progress_bar.container.layout.visibility = "visible"
            progress_bar.displayed = True
            run.description = ""
            cancel.disabled = False

        if on_toggle_pause is not None:
            on_toggle_pause()

        run_update()

    run.on_click(on_run)

    status = widgets.HTML(value=_info_html(measurement))

    output = widgets.Output(layout=widgets.Layout(height="100px"))

    def ui_update():
        status.value = _info_html(measurement)
        progress_bar.update(measurement.current_index - progress_bar.n)
        run_update()

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
        widgets.VBox(
            (
                widgets.HBox(
                    (header, widgets.HBox((run, cancel))),
                    layout=widgets.Layout(
                        justify_content="space-between",
                        width="100%",
                        align_items="flex-end",
                    ),
                ),
                status,
                progress_bar.container,
                output,
            ),
            layout=widgets.Layout(max_width="700px"),
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
    style = "text-align: right; padding: 0.5em; line-height: 1.0;"
    if i % 2 == 1:
        style += " background: var(--jp-layout-color1);"
    return f'<tr><th style="{style}">{key}</th><th style="{style}">{value}</th></tr>'


def _info_html(measurement: "Measurement") -> str:
    status = measurement.status

    color = {
        MeasurementStatus.INIT: "var(--jp-ui-font-color0)",
        MeasurementStatus.PAUSED: "var(--jp-ui-font-color1)",
        MeasurementStatus.CANCELLED: "var(--jp-warn-color0)",
        MeasurementStatus.FAILED: "var(--jp-error-color0)",
        MeasurementStatus.RUNNING: "var(--jp-info-color0)",
        MeasurementStatus.FINISHED: "var(--jp-success-color0)",
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
                f'<button class="lm-Widget jupyter-widgets jupyter-button" onclick="navigator.clipboard.writeText(\'{measurement._path.name}\')">Filename</button>'
                + f'<button class="lm-Widget jupyter-widgets jupyter-button" style="margin-right: -0.5em;" onclick="navigator.clipboard.writeText(\'{os.path.abspath(measurement._path)}\')">Path</button>',
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
                out_handler.stream = orig_handler.stream  # type: ignore[assignment]
            logger.handlers = [
                handler
                for handler in logger.handlers
                if not _is_console_logging_handler(handler)
            ] + [out_handler]
        yield
    finally:
        for logger, original_handlers in zip(loggers, original_handlers_list):
            logger.handlers = original_handlers
