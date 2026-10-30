"""Scoped, thread-aware routing of Python text streams."""

import sys
from contextlib import contextmanager
from threading import RLock, get_ident
from typing import Any, Callable, Iterator

_LOCK = RLock()


class _LineBuffer:
    def __init__(self, emit: Callable[[str], None]):
        self.emit = emit
        self.pending = ""

    def write(self, text: str) -> int:
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.emit(line.rstrip("\r"))
        return len(text)

    def flush(self) -> None:
        if self.pending:
            text, self.pending = self.pending, ""
            self.emit(text)


class _ThreadStream:
    def __init__(self, original: Any):
        self.original = original
        self.targets: dict[int, _LineBuffer] = {}

    def write(self, text: str) -> int:
        target = self.targets.get(get_ident(), self.original)
        return target.write(text)

    def flush(self) -> None:
        target = self.targets.get(get_ident(), self.original)
        target.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


@contextmanager
def capture_output(
    stdout: Callable[[str], None], stderr: Callable[[str], None]
) -> Iterator[None]:
    """Capture text from the calling thread, restoring streams even on failure.

    Other threads retain their original output destination. Nested captures and
    concurrent measurement threads each have independent line buffers.
    """
    thread_id = get_ident()
    installed = []
    with _LOCK:
        for name, emit in (("stdout", stdout), ("stderr", stderr)):
            stream = getattr(sys, name)
            if not isinstance(stream, _ThreadStream):
                stream = _ThreadStream(stream)
                setattr(sys, name, stream)
            previous = stream.targets.get(thread_id)
            buffer = _LineBuffer(emit)
            stream.targets[thread_id] = buffer
            installed.append((name, stream, previous, buffer))
    try:
        yield
    finally:
        try:
            for _, _, _, buffer in installed:
                buffer.flush()
        finally:
            with _LOCK:
                for name, stream, previous, _ in installed:
                    if previous is None:
                        stream.targets.pop(thread_id)
                    else:
                        stream.targets[thread_id] = previous
                    if not stream.targets and getattr(sys, name) is stream:
                        setattr(sys, name, stream.original)
