from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

import ipywidgets as widgets
import tqdm.notebook


class MeasurementStatus(StrEnum):
    INIT = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class LiveInfoElements(object):
    pbar: tqdm.notebook.tqdm
    run_btn: widgets.Button
    cancel_btn: widgets.Button
    output: widgets.Output
    ui_update: Callable[[], None]
    ui_finish: Callable[[], None]


@dataclass
class LivePlotElements(object):
    update_step: Callable[[], None]
    update_full: Callable[[], None]
