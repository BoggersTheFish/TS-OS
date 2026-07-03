from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .faults import AssemblyError
from .instruction import Instruction
from .isa import OPERAND_COUNTS, Opcode, checked_register
from .receipts import sha256_hex

TOKEN_SPLIT = re.compile(r"[\s,]+")


@dataclass(frozen=True)
class Program:
    instructions: tuple[Instruction, ...]
    entry_point: int = 0
    symbols: dict[str, int] | None = None

    def to_instruction_json(self) -> list[dict[str, Any]]:
        return [i.to_json() for i in self.instructions]

    @property
    def program_hash(self) -> str:
        return sha256_hex({"entry_point": self.entry_point, "instructions": self.to_instruction_json()})


def _parse_operand(opcode: Opcode, token: str, labels: dict[str, int], operand_index: int) -> Any:
    if opcode in {Opcode.JMP, Opcode.JZ, Opcode.JNZ}:
        if token in labels:
            return labels[token]
        try:
            return int(token, 0)
        except ValueError as exc:
            raise AssemblyError(f"unresolved label {token!r}") from exc
    if opcode in {Opcode.LOADI} and operand_index == 1:
        return int(token, 0)
    if opcode in {Opcode.LOAD, Opcode.STORE} and operand_index == 1:
        return int(token, 0)
    checked_register(token)
    return token


def assemble_text(text: str) -> Program:
    labels: dict[str, int] = {}
    rows: list[tuple[int, list[str]]] = []
    pc = 0
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.split(";", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            label = line[:-1].strip()
            if not label or label in labels:
                raise AssemblyError(f"invalid or duplicate label on line {line_no}")
            labels[label] = pc
            continue
        parts = [p for p in TOKEN_SPLIT.split(line) if p]
        rows.append((line_no, parts))
        pc += 1

    instructions: list[Instruction] = []
    for line_no, parts in rows:
        try:
            opcode = Opcode(parts[0].upper())
        except ValueError as exc:
            raise AssemblyError(f"unknown opcode on line {line_no}: {parts[0]}") from exc
        expected = OPERAND_COUNTS[opcode]
        operands = parts[1:]
        if len(operands) != expected:
            raise AssemblyError(f"{opcode.value} expects {expected} operands on line {line_no}")
        parsed = tuple(_parse_operand(opcode, op, labels, i) for i, op in enumerate(operands))
        instructions.append(Instruction(opcode, parsed))
    return Program(tuple(instructions), symbols=dict(sorted(labels.items())))
