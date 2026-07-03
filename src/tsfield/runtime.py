from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class BoundaryMode(str, Enum):
    NEUMANN = "NEUMANN"
    PERIODIC = "PERIODIC"


@dataclass(frozen=True)
class FieldConfiguration:
    dimensions: int = 3
    resolution: int = 16
    extent: float = 6.0
    timestep: float = 0.02
    diffusion: float = 0.1
    reaction_beta: float = 1.0
    boundary: BoundaryMode = BoundaryMode.NEUMANN
    clip_min: float = -2.0
    clip_max: float = 2.0

    @property
    def spacing(self) -> float:
        return self.extent / (self.resolution - 1)


class FieldRuntime:
    def __init__(self, config: FieldConfiguration):
        self.config = config
        shape = (config.resolution,) * config.dimensions
        self.phi = np.zeros(shape, dtype=np.float64)
        self.step_counter = 0

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        dx2 = self.config.spacing**2
        lap = np.zeros_like(field)
        if self.config.boundary == BoundaryMode.PERIODIC:
            for axis in range(field.ndim):
                lap += np.roll(field, 1, axis=axis) + np.roll(field, -1, axis=axis) - 2 * field
            return lap / dx2
        padded = np.pad(field, 1, mode="edge")
        center = (slice(1, -1),) * field.ndim
        for axis in range(field.ndim):
            plus = list(center)
            minus = list(center)
            plus[axis] = slice(2, None)
            minus[axis] = slice(None, -2)
            lap += padded[tuple(plus)] + padded[tuple(minus)] - 2 * field
        return lap / dx2

    def step(self, source: np.ndarray | None = None) -> dict[str, object]:
        if source is None:
            source = np.zeros_like(self.phi)
        lap = self.laplacian(self.phi)
        reaction = self.phi - self.phi**3
        self.phi = self.phi + self.config.timestep * (
            self.config.diffusion * lap + self.config.reaction_beta * reaction + source
        )
        self.phi = np.clip(self.phi, self.config.clip_min, self.config.clip_max)
        receipt = {
            "runtime": "tsfield-v0",
            "step": self.step_counter,
            "boundary": self.config.boundary.value,
            "min": float(np.min(self.phi)),
            "max": float(np.max(self.phi)),
            "mean": float(np.mean(self.phi)),
        }
        self.step_counter += 1
        return receipt
