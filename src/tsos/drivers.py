from __future__ import annotations

from typing import Protocol


class ExecutionDriver(Protocol):
    def instructions_allowed_this_tick(self, process_id: int, kernel_tick: int) -> int: ...


class ClockedExecutionDriver:
    def __init__(self, instructions_per_tick: int = 1):
        if instructions_per_tick < 0:
            raise ValueError("instructions_per_tick must be non-negative")
        self.instructions_per_tick = instructions_per_tick

    def instructions_allowed_this_tick(self, process_id: int, kernel_tick: int) -> int:
        return self.instructions_per_tick


class WaveThresholdExecutionDriver:
    """Experimental timing driver. It changes when instructions run, not VM semantics."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def instructions_allowed_this_tick(self, process_id: int, kernel_tick: int) -> int:
        return 1 if kernel_tick >= 0 else 0
