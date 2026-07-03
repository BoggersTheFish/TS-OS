from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .kernel import Kernel, Process


class ExecutionDriver(Protocol):
    def instructions_allowed_this_tick(self, kernel: "Kernel", process: "Process") -> int: ...


class ClockedExecutionDriver:
    def __init__(self, instructions_per_tick: int = 1):
        if instructions_per_tick < 0:
            raise ValueError("instructions_per_tick must be non-negative")
        self.instructions_per_tick = instructions_per_tick

    def instructions_allowed_this_tick(self, kernel: "Kernel", process: "Process") -> int:
        return self.instructions_per_tick


class WaveThresholdExecutionDriver:
    """Experimental timing driver. It changes when instructions run, not VM semantics."""

    def __init__(self, threshold: float = 0.5, instructions_per_tick: int = 1):
        self.threshold = threshold
        self.instructions_per_tick = instructions_per_tick

    def instructions_allowed_this_tick(self, kernel: "Kernel", process: "Process") -> int:
        sample = kernel.sample_process_field(process.pid)
        return self.instructions_per_tick if sample >= self.threshold else 0
