from __future__ import annotations

from enum import Enum

MACHINE_VERSION = "bogvm-isa-v0"
WORD_BITS = 32
WORD_MASK = (1 << WORD_BITS) - 1
SIGNED_MIN = -(1 << (WORD_BITS - 1))
SIGNED_MAX = (1 << (WORD_BITS - 1)) - 1
REGISTER_COUNT = 4
MEMORY_SIZE = 64


class Opcode(str, Enum):
    NOP = "NOP"
    LOADI = "LOADI"
    MOV = "MOV"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    LOAD = "LOAD"
    STORE = "STORE"
    CMP = "CMP"
    JMP = "JMP"
    JZ = "JZ"
    JNZ = "JNZ"
    EMIT = "EMIT"
    HALT = "HALT"


OPERAND_COUNTS = {
    Opcode.NOP: 0,
    Opcode.LOADI: 2,
    Opcode.MOV: 2,
    Opcode.ADD: 3,
    Opcode.SUB: 3,
    Opcode.MUL: 3,
    Opcode.DIV: 3,
    Opcode.LOAD: 2,
    Opcode.STORE: 2,
    Opcode.CMP: 2,
    Opcode.JMP: 1,
    Opcode.JZ: 1,
    Opcode.JNZ: 1,
    Opcode.EMIT: 1,
    Opcode.HALT: 0,
}


def to_word(value: int) -> int:
    value &= WORD_MASK
    if value > SIGNED_MAX:
        return value - (1 << WORD_BITS)
    return value


def checked_register(name: str) -> int:
    if not isinstance(name, str) or not name.startswith("R"):
        raise ValueError(f"invalid register {name!r}")
    try:
        idx = int(name[1:])
    except ValueError as exc:
        raise ValueError(f"invalid register {name!r}") from exc
    if idx < 0 or idx >= REGISTER_COUNT:
        raise ValueError(f"invalid register {name!r}")
    return idx
