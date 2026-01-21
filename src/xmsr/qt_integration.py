_qt_available = False

try:
    import qtpy

    _qt_available = True
except Exception:
    pass

if _qt_available:
    from qtpy.QtCore import QThread as Thread
else:
    from threading import Thread

__all__ = ["Thread"]
