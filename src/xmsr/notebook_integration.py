"""Notebook display integration for runner controls and existing live plots."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import holoviews as hv
import hvplot.xarray  # noqa: F401
import ipywidgets as widgets
import xarray as xr
from holoviews.streams import Pipe
from IPython import get_ipython
from IPython.display import display

from .runner_widget import RunnerWidget
from .shared import MeasurementStatus

if TYPE_CHECKING:
    from .measurement import Measurement

_NOTEBOOK_INITIALIZED = False


def initialize_notebook() -> None:
    """Load plotting resources in the notebook cell before starting a worker."""
    global _NOTEBOOK_INITIALIZED
    if _NOTEBOOK_INITIALIZED:
        return
    try:
        hv.notebook_extension("bokeh", logo=False, inline=True)
    except Exception:
        hv.extension("bokeh", logo=False)
    _NOTEBOOK_INITIALIZED = get_ipython() is not None


@dataclass
class LiveInfoElements:
    """Notebook view and its lifecycle callbacks."""

    widget: RunnerWidget
    ui_update: Callable[[], None]
    ui_finish: Callable[[], None]


def live_info(measurement: "Measurement") -> LiveInfoElements:
    """Display React controls with a separate rich result output."""
    widget = RunnerWidget(measurement)
    output = widgets.Output()
    display(widget, output)

    def ui_finish() -> None:
        widget.finish()
        if (
            measurement.current_index > 0
            and measurement.status != MeasurementStatus.FAILED
        ):
            output.append_display_data(measurement.result)

    return LiveInfoElements(widget, widget.publish, ui_finish)


def live_plot(measurement: "Measurement") -> Callable[[], None]:
    """Create the existing HoloViews live plot updater."""
    initialize_notebook()

    try:
        measurement.plot_single_step(xr.DataArray(0))
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
    full_buffer = None
    full_dmap = None

    def update_plots() -> None:
        # Keep the maps alive: Pipe stream subscribers and notebook display
        # hooks do not guarantee ownership of the DynamicMap after display().
        nonlocal step_pipe, step_dmap, full_buffer, full_dmap
        step_da = measurement.display_measurement
        if use_step_plot and step_da is not None:
            if step_pipe is None:
                step_pipe = Pipe(data=step_da)
                step_dmap = hv.DynamicMap(
                    measurement.plot_single_step, streams=[step_pipe]
                )
                display(step_dmap)
            else:
                step_pipe.send(step_da)

        full_da = None
        if hasattr(measurement, "_path") and measurement._path.exists():
            try:
                full_da = measurement.result
            except Exception:
                full_da = None

        if use_full_plot and full_da is not None:
            if full_buffer is None:
                full_buffer = Pipe(data=full_da)
                full_dmap = hv.DynamicMap(
                    measurement.plot_preview, streams=[full_buffer]
                )
                display(full_dmap)
            else:
                full_buffer.send(full_da)

    return update_plots
