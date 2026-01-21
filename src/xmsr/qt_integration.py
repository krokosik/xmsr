from qtpy import QtBindingsNotFoundError

_qt_available = False

try:
    from qtpy import QT_VERSION

    _qt_available = True
except QtBindingsNotFoundError:
    pass

if _qt_available:
    from qtpy.QtCore import QThread as Thread
else:
    from threading import Thread

__all__ = ["Thread"]
