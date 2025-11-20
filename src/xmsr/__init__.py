# from .gui import _Measurement as Measurement
# from .gui import _MeasurementGUI
from .measurement import VariableData, Measurement

# @classmethod
# @property
# def _measurement_gui(cls):
#     if not hasattr(cls, "_gui"):
#         setattr(cls, "_gui", _MeasurementGUI(cls))
#     cls._gui.show()

# try:
#     setattr(Measurement, "gui", _measurement_gui)
# except AttributeError:
#     pass

__all__ = ["Measurement", "VariableData"]