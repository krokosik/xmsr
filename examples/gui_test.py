import numpy as np
from time import sleep
from xmsr import Measurement, VariableData
from PyQt5.QtWidgets import QApplication

t = np.pi * np.linspace(-1, 1, 100)

app = QApplication([])


class MeasurementWithPreview(Measurement):
    target_directory = "~/kod/xmsr/tmp"
    param_coords = {
        "amplitude": range(5, 10),
        "frequency": [1, 1 / 3, 1 / 2],
        # "signal/noise": ["signal", "noise"],
    }
    variables = [
        VariableData("signal", ["time"], {"time": t}),
        VariableData("noise", ["time"], {"time": t}),
    ]
    # data_dims = ["time"]
    # data_coords = {
    #     "time": (np.pi * np.linspace(-1, 1, 100)),
    # }
    with_coords = False

    def measure(self, values, indices, metadata):
        sleep(0.1)
        signal = np.random.randn(len(t)) + values["amplitude"] * np.sin(
            values["frequency"] * t
        )
        noise = np.random.randn(len(t))

        return signal, noise

    def plot_preview(self, chunk_da, full_da, ax):
        chunk_da["signal"].plot.line(x="time", ax=ax, ylim=(-10, 10))


MeasurementWithPreview.gui

app.exec_()