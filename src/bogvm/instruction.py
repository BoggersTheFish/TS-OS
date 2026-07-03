from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .isa import OPERAND_COUNTS, Opcode, checked_register


@dataclass(frozen=True)
class Instruction:
    opcode: Opcode
    operands: tuple[Any, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"opcode": self.opcode.value, "operands": list(self.operands)}

    @classmethod
    def from_json(cls, obj: Any) -> "Instruction":
        if not isinstance(obj, dict) or set(obj) != {"opcode", "operands"}:
            raise ValueError("instruction must contain exactly opcode and operands")
        opcode = Opcode(obj["opcode"])
        operands = tuple(obj["operands"])
        if len(operands) != OPERAND_COUNTS[opcode]:
            raise ValueError(f"{opcode.value} expects {OPERAND_COUNTS[opcode]} operands")
        for i, operand in enumerate(operands):
            if opcode in {Opcode.LOADI, Opcode.LOAD, Opcode.STORE, Opcode.EMIT, Opcode.MOV, Opcode.CMP}:
                if i == 0 or (opcode in {Opcode.MOV, Opcode.CMP} and i == 1):
                    checked_register(str(operand))
            if opcode in {Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV}:
                checked_register(str(operand))
        return cls(opcode, operands)
