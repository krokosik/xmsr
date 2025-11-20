from IPython.display import display
from ipywidgets import Button
from xmsr.measurement import Measurement

class NotebookRunner(object):
    def __init__(self, measurement: Measurement):
        self.m = measurement

    def run_widget(self):
        run_btn = Button(
            description=f"Run {self.__class__.__name__}",
            layout={"width": "200px"},
        )

        display(run_btn)

        def run_measurement():
            self.m.run()
            self.m.plot_result()

        run_btn.on_click(lambda _: run_measurement())
