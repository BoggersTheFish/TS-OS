from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assembler import Program
from .faults import FormatValidationError
from .instruction import Instruction
from .isa import MACHINE_VERSION
from .receipts import canonical_json_bytes, sha256_hex, verify_receipt_chain


BOGEXE_FORMAT = "bogexe"
BOGPKG_FORMAT = "bogpkg"
BOGSTATE_FORMAT = "bogstate"
BOGGRAPH_FORMAT = "boggraph"
BOGTRACE_FORMAT = "bogtrace"
SCHEMA_VERSION = 0


def _strict_keys(obj: dict[str, Any], required: set[str]) -> None:
    if set(obj) != required:
        missing = required - set(obj)
        unknown = set(obj) - required
        raise FormatValidationError(f"schema keys mismatch missing={sorted(missing)} unknown={sorted(unknown)}")


def write_canonical_json(path: str | Path, obj: Any) -> None:
    Path(path).write_bytes(canonical_json_bytes(obj) + b"\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class BogExe:
    machine_version: str
    instructions: tuple[Instruction, ...]
    entry_point: int = 0
    symbols: dict[str, int] | None = None
    program_hash: str | None = None

    @classmethod
    def from_program(cls, program: Program) -> "BogExe":
        return cls(MACHINE_VERSION, program.instructions, program.entry_point, program.symbols, program.program_hash)

    def program(self) -> Program:
        return Program(self.instructions, self.entry_point, self.symbols)

    def to_json(self) -> dict[str, Any]:
        program = self.program()
        return {
            "format": BOGEXE_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "machine_version": self.machine_version,
            "entry_point": self.entry_point,
            "instructions": [i.to_json() for i in self.instructions],
            "symbols": self.symbols or {},
            "program_hash": self.program_hash or program.program_hash,
        }

    @classmethod
    def from_json(cls, obj: Any) -> "BogExe":
        if not isinstance(obj, dict):
            raise FormatValidationError("bogexe must be an object")
        _strict_keys(obj, {"format", "schema_version", "machine_version", "entry_point", "instructions", "symbols", "program_hash"})
        if obj["format"] != BOGEXE_FORMAT or obj["schema_version"] != SCHEMA_VERSION:
            raise FormatValidationError("unsupported bogexe format or version")
        if obj["machine_version"] != MACHINE_VERSION:
            raise FormatValidationError("unsupported machine version")
        instructions = tuple(Instruction.from_json(i) for i in obj["instructions"])
        exe = cls(obj["machine_version"], instructions, int(obj["entry_point"]), dict(obj["symbols"]), obj["program_hash"])
        if exe.program().program_hash != obj["program_hash"]:
            raise FormatValidationError("program hash mismatch")
        return exe


@dataclass(frozen=True)
class ProcessDefinition:
    pid: int
    executable: BogExe
    position: tuple[float, float, float]
    priority: int = 1
    amplitude: float = 1.0
    phase: float = 0.0
    requested_resources: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "executable": self.executable.to_json(),
            "position": list(self.position),
            "priority": self.priority,
            "amplitude": self.amplitude,
            "phase": self.phase,
            "requested_resources": self.requested_resources or {},
        }

    @classmethod
    def from_json(cls, obj: Any) -> "ProcessDefinition":
        if not isinstance(obj, dict):
            raise FormatValidationError("process definition must be an object")
        _strict_keys(obj, {"pid", "executable", "position", "priority", "amplitude", "phase", "requested_resources"})
        pos = obj["position"]
        if not isinstance(pos, list) or len(pos) != 3:
            raise FormatValidationError("position must be a 3-vector")
        return cls(
            int(obj["pid"]),
            BogExe.from_json(obj["executable"]),
            (float(pos[0]), float(pos[1]), float(pos[2])),
            int(obj["priority"]),
            float(obj["amplitude"]),
            float(obj["phase"]),
            dict(obj["requested_resources"]),
        )


@dataclass(frozen=True)
class BogPkg:
    package_id: str
    processes: tuple[ProcessDefinition, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "format": BOGPKG_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "package_id": self.package_id,
            "processes": [p.to_json() for p in self.processes],
        }

    @classmethod
    def from_json(cls, obj: Any) -> "BogPkg":
        if not isinstance(obj, dict):
            raise FormatValidationError("bogpkg must be an object")
        _strict_keys(obj, {"format", "schema_version", "package_id", "processes"})
        if obj["format"] != BOGPKG_FORMAT or obj["schema_version"] != SCHEMA_VERSION:
            raise FormatValidationError("unsupported bogpkg format or version")
        return cls(str(obj["package_id"]), tuple(ProcessDefinition.from_json(p) for p in obj["processes"]))


def trace_to_json(receipts: list[Any]) -> dict[str, Any]:
    payload = [r.to_json() if hasattr(r, "to_json") else r for r in receipts]
    return {
        "format": BOGTRACE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "receipt_count": len(payload),
        "receipts": payload,
        "valid": verify_receipt_chain(payload),
    }


def validate_trace_json(obj: Any) -> bool:
    if not isinstance(obj, dict):
        raise FormatValidationError("bogtrace must be an object")
    _strict_keys(obj, {"format", "schema_version", "receipt_count", "receipts", "valid"})
    if obj["format"] != BOGTRACE_FORMAT or obj["schema_version"] != SCHEMA_VERSION:
        raise FormatValidationError("unsupported bogtrace format or version")
    if obj["receipt_count"] != len(obj["receipts"]):
        raise FormatValidationError("receipt count mismatch")
    return verify_receipt_chain(obj["receipts"])


def bogstate_json(kernel_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": BOGSTATE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "kernel_version": "tsos-kernel-v0",
        "state": kernel_state,
        "state_hash": sha256_hex(kernel_state),
    }
