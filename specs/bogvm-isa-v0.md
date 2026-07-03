# BOGVM ISA v0

Machine version: `bogvm-isa-v0`.

BOGVM v0 is a deterministic integer virtual machine. Canonical semantics do not depend on topology, pulse timing, neighbour ordering, graphics, or floating-point field state.

## State

- Registers: `R0` through `R3`, signed 32-bit two's-complement integers.
- Memory: 64 signed 32-bit integer cells, addressed `0..63`.
- Programme counter: zero-based instruction index.
- Flag: `zero`.
- Status: runnable, halted, or faulted.
- Step counter: starts at zero and increments after every executed instruction or deterministic fault receipt.

## Integer Semantics

Arithmetic results wrap to signed 32-bit two's-complement range. Division truncates toward zero. Division by zero faults. Memory and jump bounds faults are deterministic.

## Instructions

- `NOP`
- `LOADI dst, imm`
- `MOV dst, src`
- `ADD dst, a, b`
- `SUB dst, a, b`
- `MUL dst, a, b`
- `DIV dst, a, b`
- `LOAD dst, address`
- `STORE src, address`
- `CMP a, b`
- `JMP target`
- `JZ target`
- `JNZ target`
- `EMIT src`
- `HALT`

Invalid opcodes, malformed operands, invalid registers, unresolved labels, and wrong operand counts are rejected by assembly or strict object validation before execution.

## Receipts

Each executed instruction produces one receipt from the actual transition: PC before/after, decoded instruction, register changes, memory changes, flags before/after, output event, halt/fault state, previous receipt hash, and current receipt hash.
