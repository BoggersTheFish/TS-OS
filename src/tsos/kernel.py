from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from bogvm.formats import BogPkg
from bogvm.machine import BOGMachine
from bogvm.receipts import sha256_hex
from tsfield import FieldConfiguration, FieldRuntime

from .drivers import ClockedExecutionDriver, ExecutionDriver, WaveThresholdExecutionDriver

UNOWNED = -1


class ProcessStatus(str, Enum):
    CREATED = "CREATED"
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    SUSPENDED = "SUSPENDED"
    HALTED = "HALTED"
    FAULTED = "FAULTED"
    TERMINATED = "TERMINATED"


@dataclass
class Process:
    pid: int
    position: tuple[float, float, float]
    priority: int
    amplitude: float
    phase: float
    machine: BOGMachine
    status: ProcessStatus = ProcessStatus.CREATED
    territory_size: int = 0
    output_events: list[int] = field(default_factory=list)

    def is_eligible(self) -> bool:
        return self.status in {ProcessStatus.READY, ProcessStatus.RUNNING}

    def sync_from_machine(self) -> None:
        if self.machine.fault is not None:
            self.status = ProcessStatus.FAULTED
        elif self.machine.halted:
            self.status = ProcessStatus.HALTED

    @property
    def last_receipt_hash(self) -> str | None:
        return self.machine.receipts[-1].receipt_hash if self.machine.receipts else None


@dataclass(frozen=True)
class IPCRecord:
    process_a: int
    process_b: int
    boundary_measure: int
    field_tension: int
    sample_count: int
    runtime_tick: int
    valid: bool
    source_state_hash: str

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()


class WeightedSpatialAllocator:
    def allocate(self, kernel: "Kernel") -> None:
        active = [
            p
            for p in kernel.processes.values()
            if p.status not in {ProcessStatus.SUSPENDED, ProcessStatus.TERMINATED}
            and p.amplitude > 0
        ]
        if not active:
            kernel.voxel_ownership.fill(UNOWNED)
            for p in kernel.processes.values():
                p.territory_size = 0
            return
        pids = list(kernel.processes)
        dists = np.zeros((len(pids), kernel.N, kernel.N, kernel.N), dtype=np.float64)
        for idx, pid in enumerate(pids):
            p = kernel.processes[pid]
            if p.status in {ProcessStatus.TERMINATED, ProcessStatus.SUSPENDED} or p.amplitude <= 0:
                dists[idx].fill(1e12)
                continue
            dist = np.sqrt(
                (kernel.X - p.position[0]) ** 2
                + (kernel.Y - p.position[1]) ** 2
                + (kernel.Z - p.position[2]) ** 2
            )
            dists[idx] = dist / max(float(p.amplitude), 1e-9)
        kernel.voxel_ownership = np.array(pids, dtype=np.int32)[np.argmin(dists, axis=0)]
        for pid, p in kernel.processes.items():
            p.territory_size = int(np.sum(kernel.voxel_ownership == pid)) if p.status != ProcessStatus.SUSPENDED else 0


class Kernel:
    version = "tsos-kernel-v0"

    def __init__(
        self,
        N: int = 12,
        driver: ExecutionDriver | None = None,
        field_enabled: bool = False,
        field_runtime: FieldRuntime | None = None,
    ):
        self.N = N
        self.x = np.linspace(-3, 3, N)
        self.dx = float(self.x[1] - self.x[0]) if N > 1 else 1.0
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.x, self.x, indexing="ij")
        self.Phi: np.ndarray = np.zeros((N, N, N), dtype=np.float64)
        self.voxel_ownership: np.ndarray = np.full((N, N, N), UNOWNED, dtype=np.int32)
        self.processes: dict[int, Process] = {}
        self.driver = driver or ClockedExecutionDriver()
        self.allocator = WeightedSpatialAllocator()
        self.ipc_records: dict[tuple[int, int], IPCRecord] = {}
        self.runtime_tick = 0
        self.field_runtime = field_runtime
        if self.field_runtime is None and field_enabled:
            self.field_runtime = FieldRuntime(FieldConfiguration(resolution=N))
        self.field_enabled = self.field_runtime is not None
        self.kernel_receipts: list[dict[str, Any]] = []

    def boot_package(self, package: BogPkg) -> None:
        for definition in package.processes:
            machine = BOGMachine()
            machine.load_program(definition.executable.program())
            self.processes[definition.pid] = Process(
                definition.pid,
                definition.position,
                definition.priority,
                definition.amplitude,
                definition.phase,
                machine,
                ProcessStatus.READY,
            )
        self.allocator.allocate(self)

    def suspend(self, pid: int) -> None:
        self.processes[pid].status = ProcessStatus.SUSPENDED

    def resume(self, pid: int) -> None:
        p = self.processes[pid]
        if p.status == ProcessStatus.SUSPENDED:
            p.status = ProcessStatus.READY

    def kill(self, pid: int) -> None:
        self.processes[pid].status = ProcessStatus.TERMINATED

    def _nearest_index(self, position: tuple[float, float, float]) -> tuple[int, int, int]:
        return (
            int(np.argmin(np.abs(self.x - position[0]))),
            int(np.argmin(np.abs(self.x - position[1]))),
            int(np.argmin(np.abs(self.x - position[2]))),
        )

    def sample_process_field(self, pid: int) -> float:
        if self.field_runtime is None:
            return 0.0
        process = self.processes[pid]
        return float(self.field_runtime.phi[self._nearest_index(process.position)])

    def _build_field_source(self) -> np.ndarray:
        assert self.field_runtime is not None
        source = np.zeros_like(self.field_runtime.phi)
        for process in self.processes.values():
            if process.status in {ProcessStatus.SUSPENDED, ProcessStatus.TERMINATED}:
                continue
            idx = self._nearest_index(process.position)
            source[idx] += process.amplitude * np.cos(process.phase)
        return source

    def _step_field(self) -> dict[str, object] | None:
        if self.field_runtime is None:
            return None
        receipt = self.field_runtime.step(self._build_field_source())
        self.Phi = self.field_runtime.phi.copy()
        return receipt

    def _resolve_ipc(self) -> None:
        self.ipc_records = {}
        active = [p for p in self.processes.values() if p.status not in {ProcessStatus.SUSPENDED, ProcessStatus.TERMINATED}]
        if len(active) < 2:
            return
        owners = self.voxel_ownership
        pairs: dict[tuple[int, int], int] = {}
        for axis in range(3):
            a = np.take(owners, range(owners.shape[axis] - 1), axis=axis)
            b = np.take(owners, range(1, owners.shape[axis]), axis=axis)
            mask = (a != b) & (a != UNOWNED) & (b != UNOWNED)
            for x, y in zip(a[mask].flat, b[mask].flat):
                left, right = sorted((int(x), int(y)))
                key: tuple[int, int] = (left, right)
                pairs[key] = pairs.get(key, 0) + 1
        source_hash = sha256_hex(
            {
                "tick": self.runtime_tick,
                "owners": int(np.sum(owners)),
                "field_step": None
                if self.field_runtime is None
                else self.field_runtime.step_counter,
            }
        )
        for pair, count in pairs.items():
            proc_a, proc_b = pair
            field_tension = 0
            if self.field_runtime is not None:
                field_tension = int(round(float(np.mean(np.abs(self.Phi))) * 100_000_000))
            self.ipc_records[pair] = IPCRecord(
                proc_a,
                proc_b,
                count,
                field_tension,
                count,
                self.runtime_tick,
                True,
                source_hash,
            )

    def tick(self) -> dict[str, Any]:
        executed: list[dict[str, Any]] = []
        outputs: list[dict[str, int]] = []
        field_receipt = self._step_field()
        field_samples: dict[str, float] = {
            str(pid): self.sample_process_field(pid)
            for pid, process in sorted(self.processes.items())
            if process.status not in {ProcessStatus.TERMINATED}
        }
        for pid in sorted(self.processes):
            p = self.processes[pid]
            if not p.is_eligible():
                continue
            p.status = ProcessStatus.RUNNING
            allowed = self.driver.instructions_allowed_this_tick(self, p)
            for _ in range(allowed):
                if p.machine.halted or p.machine.fault is not None:
                    break
                receipt = p.machine.step()
                executed.append({"pid": pid, "receipt_hash": receipt.receipt_hash, "pc_after": receipt.pc_after})
                if receipt.output is not None:
                    p.output_events.append(receipt.output)
                    outputs.append({"pid": pid, "value": receipt.output})
            p.sync_from_machine()
            if p.status == ProcessStatus.RUNNING:
                p.status = ProcessStatus.READY
        self.allocator.allocate(self)
        self._resolve_ipc()
        kernel_receipt: dict[str, Any] = {
            "kernel_version": self.version,
            "runtime_tick": self.runtime_tick,
            "executed": executed,
            "outputs": outputs,
            "process_status": {str(pid): p.status.value for pid, p in sorted(self.processes.items())},
            "ipc_pairs": [list(k) for k in sorted(self.ipc_records)],
            "field": field_receipt,
            "field_samples": field_samples,
        }
        kernel_receipt["receipt_hash"] = sha256_hex(kernel_receipt)
        self.kernel_receipts.append(kernel_receipt)
        self.runtime_tick += 1
        return kernel_receipt

    def run(self, max_ticks: int) -> None:
        for _ in range(max_ticks):
            if all(p.status in {ProcessStatus.HALTED, ProcessStatus.FAULTED, ProcessStatus.TERMINATED, ProcessStatus.SUSPENDED} for p in self.processes.values()):
                break
            self.tick()

    def state(self) -> dict[str, Any]:
        if isinstance(self.driver, WaveThresholdExecutionDriver):
            driver_state = {
                "kind": "wave_threshold",
                "threshold": self.driver.threshold,
                "instructions_per_tick": self.driver.instructions_per_tick,
            }
        elif isinstance(self.driver, ClockedExecutionDriver):
            driver_state = {
                "kind": "clocked",
                "instructions_per_tick": self.driver.instructions_per_tick,
            }
        else:
            driver_state = {"kind": "unknown"}
        return {
            "runtime_tick": self.runtime_tick,
            "field_enabled": self.field_enabled,
            "field": None if self.field_runtime is None else self.field_runtime.snapshot(),
            "driver": driver_state,
            "processes": {
                str(pid): {
                    "status": p.status.value,
                    "position": list(p.position),
                    "priority": p.priority,
                    "amplitude": p.amplitude,
                    "territory_size": p.territory_size,
                    "machine": p.machine.snapshot(),
                    "last_receipt": p.last_receipt_hash,
                    "outputs": list(p.output_events),
                }
                for pid, p in sorted(self.processes.items())
            },
            "ipc_records": [r.to_json() for r in self.ipc_records.values()],
        }
