from enum import StrEnum


class MeasurementStatus(StrEnum):
    INIT = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"
