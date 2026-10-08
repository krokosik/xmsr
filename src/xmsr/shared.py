from enum import StrEnum


class MeasurementStatus(StrEnum):
    INIT = "initialized"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"
