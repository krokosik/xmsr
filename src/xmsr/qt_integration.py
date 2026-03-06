_qt_available = False

try:
    import qtpy  # noqa: F401

    _qt_available = True
except Exception:
    pass

if _qt_available:
    from qtpy.QtCore import QThread as Thread  # type: ignore[attr-defined]
else:
    from threading import Thread

__all__ = ["Thread"]
