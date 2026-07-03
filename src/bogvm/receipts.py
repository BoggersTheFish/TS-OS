from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


def canonical_json_bytes(obj: Any) -> bytes:
    import json

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


@dataclass(frozen=True)
class ExecutionReceipt:
    program_hash: str
    machine_version: str
    step: int
    pc_before: int
    pc_after: int
    instruction: dict[str, Any]
    register_changes: dict[str, list[int]]
    memory_changes: dict[str, list[int]]
    flags_before: dict[str, bool]
    flags_after: dict[str, bool]
    output: int | None
    halted: bool
    fault: dict[str, str] | None
    previous_hash: str
    receipt_hash: str

    def signing_payload(self) -> dict[str, Any]:
        obj = self.to_json()
        obj.pop("receipt_hash")
        return obj

    def to_json(self) -> dict[str, Any]:
        return {
            "program_hash": self.program_hash,
            "machine_version": self.machine_version,
            "step": self.step,
            "pc_before": self.pc_before,
            "pc_after": self.pc_after,
            "instruction": self.instruction,
            "register_changes": self.register_changes,
            "memory_changes": self.memory_changes,
            "flags_before": self.flags_before,
            "flags_after": self.flags_after,
            "output": self.output,
            "halted": self.halted,
            "fault": self.fault,
            "previous_hash": self.previous_hash,
            "receipt_hash": self.receipt_hash,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ExecutionReceipt":
        return cls(**payload)


def make_receipt(payload: dict[str, Any]) -> ExecutionReceipt:
    payload = dict(payload)
    payload["receipt_hash"] = sha256_hex(payload)
    return ExecutionReceipt(**payload)


def verify_receipt_chain(receipts: list[ExecutionReceipt] | list[dict[str, Any]]) -> bool:
    previous = "0" * 64
    for item in receipts:
        receipt = item if isinstance(item, ExecutionReceipt) else ExecutionReceipt.from_payload(item)
        if receipt.previous_hash != previous:
            return False
        if sha256_hex(receipt.signing_payload()) != receipt.receipt_hash:
            return False
        previous = receipt.receipt_hash
    return True
