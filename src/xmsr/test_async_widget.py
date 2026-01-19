# %%
# from IPython import get_ipython

# scratchpad for evaluating async Python and ipywidgets code

# ip = get_ipython()
# if ip is not None:
#     ip.run_line_magic("gui", "asyncio")

from IPython.display import display
import asyncio


def wait_for_change(widget, value):
    future = asyncio.Future()

    def getvalue(change):
        # make the new value available
        future.set_result(change.new)
        widget.unobserve(getvalue, value)

    widget.observe(getvalue, value)
    return future


from ipywidgets import IntSlider, Output

slider = IntSlider()
out = Output()


async def f():
    for i in range(10):
        out.append_stdout("did work " + str(i) + "\n")
        x = await wait_for_change(slider, "value")
        out.append_stdout("async function continued with value " + str(x) + "\n")


asyncio.ensure_future(f())

display(out)
slider
# %%
import threading
from IPython.display import display
import ipywidgets as widgets
import time

progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0)


def work(progress):
    total = 100
    for i in range(total):
        time.sleep(0.2)
        progress.value = float(i + 1) / total


thread = threading.Thread(target=work, args=(progress,))
display(progress)
thread.start()

# %%
play = widgets.Play(
    value=50,
    min=0,
    max=100,
    step=1,
    interval=500,
    description="Press play",
    disabled=True,
)
slider = widgets.IntSlider()
widgets.jslink((play, "value"), (slider, "value"))
widgets.HBox([play, slider])
# %%
from queue import Queue
from tqdm.notebook import tqdm_notebook

run_btn = widgets.ToggleButton(description="Run")
cancel_btn = widgets.Button(description="Cancel")
progress_bar = tqdm_notebook(total=100, display=False)

running = threading.Event()

def work(progress):
    total = 100
    for i in range(total):
        running.wait()
        time.sleep(0.2)
        progress_bar.update(1)

    progress_bar.close()

def toggle_run(value):
    if value:
        run_btn.description = "Paused"
        running.clear()
    else:
        run_btn.description = "Run"
        running.set()

def cancel_clicked(b):
    run_btn.disabled = True
    cancel_btn.disabled = True
    cancel_btn.description = "Cancelled..."
    running.clear()
    progress_bar.close()

w = widgets.HBox(
    [
        run_btn,
        cancel_btn,
        progress_bar.container,
    ]
)
progress_bar.displayed = True

thread = threading.Thread(target=work, args=(progress_bar,))
display(w)
thread.start()
running.set()
run_btn.observe(lambda change: toggle_run(change["new"]), names="value")
cancel_btn.on_click(cancel_clicked)

# %%
