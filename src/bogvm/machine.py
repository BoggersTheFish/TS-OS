from __future__ import annotations

from dataclasses import dataclass, field

from .assembler import Program
from .faults import BOGFault
from .instruction import Instruction
from .isa import MACHINE_VERSION, MEMORY_SIZE, REGISTER_COUNT, Opcode, checked_register, to_word
from .receipts import ExecutionReceipt, make_receipt


@dataclass(frozen=True)
class MachineConfig:
    register_count: int = REGISTER_COUNT
    memory_size: int = MEMORY_SIZE
    machine_version: str = MACHINE_VERSION


@dataclass
class BOGMachine:
    config: MachineConfig = field(default_factory=MachineConfig)
    registers: list[int] = field(default_factory=lambda: [0] * REGISTER_COUNT)
    memory: list[int] = field(default_factory=lambda: [0] * MEMORY_SIZE)
    pc: int = 0
    zero_flag: bool = False
    halted: bool = False
    fault: BOGFault | None = None
    step_counter: int = 0
    program: Program | None = None
    receipts: list[ExecutionReceipt] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)

    def load_program(self, program: Program) -> None:
        self.program = program
        self.pc = program.entry_point
        self.halted = False
        self.fault = None
        self.step_counter = 0
        self.receipts.clear()
        self.outputs.clear()

    @property
    def program_hash(self) -> str:
        if self.program is None:
            raise RuntimeError("no program loaded")
        return self.program.program_hash

    def snapshot(self) -> dict[str, object]:
        return {
            "registers": list(self.registers),
            "memory": list(self.memory),
            "pc": self.pc,
            "zero_flag": self.zero_flag,
            "halted": self.halted,
            "fault": None if self.fault is None else {"code": self.fault.code, "message": self.fault.message},
            "step_counter": self.step_counter,
            "outputs": list(self.outputs),
        }

    def _reg(self, name: object) -> int:
        idx = checked_register(str(name))
        if idx >= len(self.registers):
            raise BOGFault("INVALID_REGISTER", f"{name} outside configured register file")
        return idx

    def _mem(self, address: object) -> int:
        addr = int(str(address))
        if addr < 0 or addr >= len(self.memory):
            raise BOGFault("MEMORY_BOUNDS", f"address {addr} outside memory")
        return addr

    def _jump(self, target: object) -> int:
        assert self.program is not None
        pc = int(str(target))
        if pc < 0 or pc >= len(self.program.instructions):
            raise BOGFault("PC_BOUNDS", f"jump target {pc} outside program")
        return pc

    def step(self) -> ExecutionReceipt:
        if self.program is None:
            raise RuntimeError("no program loaded")
        if self.halted or self.fault is not None:
            raise RuntimeError("machine is not runnable")
        pc_before = self.pc
        flags_before = {"zero": self.zero_flag}
        regs_before = list(self.registers)
        mem_before = list(self.memory)
        output: int | None = None
        instruction: Instruction
        try:
            if self.pc < 0 or self.pc >= len(self.program.instructions):
                raise BOGFault("PC_BOUNDS", f"pc {self.pc} outside program")
            instruction = self.program.instructions[self.pc]
            self.pc += 1
            op = instruction.opcode
            a = instruction.operands
            if op is Opcode.NOP:
                pass
            elif op is Opcode.LOADI:
                self.registers[self._reg(a[0])] = to_word(int(a[1]))
            elif op is Opcode.MOV:
                self.registers[self._reg(a[0])] = self.registers[self._reg(a[1])]
            elif op is Opcode.ADD:
                self.registers[self._reg(a[0])] = to_word(self.registers[self._reg(a[1])] + self.registers[self._reg(a[2])])
            elif op is Opcode.SUB:
                self.registers[self._reg(a[0])] = to_word(self.registers[self._reg(a[1])] - self.registers[self._reg(a[2])])
            elif op is Opcode.MUL:
                self.registers[self._reg(a[0])] = to_word(self.registers[self._reg(a[1])] * self.registers[self._reg(a[2])])
            elif op is Opcode.DIV:
                denom = self.registers[self._reg(a[2])]
                if denom == 0:
                    raise BOGFault("DIVIDE_BY_ZERO", "integer division by zero")
                self.registers[self._reg(a[0])] = to_word(int(self.registers[self._reg(a[1])] / denom))
            elif op is Opcode.LOAD:
                self.registers[self._reg(a[0])] = self.memory[self._mem(a[1])]
            elif op is Opcode.STORE:
                self.memory[self._mem(a[1])] = self.registers[self._reg(a[0])]
            elif op is Opcode.CMP:
                self.zero_flag = self.registers[self._reg(a[0])] == self.registers[self._reg(a[1])]
            elif op is Opcode.JMP:
                self.pc = self._jump(a[0])
            elif op is Opcode.JZ:
                if self.zero_flag:
                    self.pc = self._jump(a[0])
            elif op is Opcode.JNZ:
                if not self.zero_flag:
                    self.pc = self._jump(a[0])
            elif op is Opcode.EMIT:
                output = self.registers[self._reg(a[0])]
                self.outputs.append(output)
            elif op is Opcode.HALT:
                self.halted = True
        except BOGFault as exc:
            self.fault = exc
            instruction = self.program.instructions[pc_before] if 0 <= pc_before < len(self.program.instructions) else Instruction(Opcode.NOP)

        reg_changes = {
            f"R{i}": [before, after]
            for i, (before, after) in enumerate(zip(regs_before, self.registers))
            if before != after
        }
        mem_changes = {
            str(i): [before, after]
            for i, (before, after) in enumerate(zip(mem_before, self.memory))
            if before != after
        }
        previous_hash = self.receipts[-1].receipt_hash if self.receipts else "0" * 64
        receipt = make_receipt(
            {
                "program_hash": self.program_hash,
                "machine_version": self.config.machine_version,
                "step": self.step_counter,
                "pc_before": pc_before,
                "pc_after": self.pc,
                "instruction": instruction.to_json(),
                "register_changes": reg_changes,
                "memory_changes": mem_changes,
                "flags_before": flags_before,
                "flags_after": {"zero": self.zero_flag},
                "output": output,
                "halted": self.halted,
                "fault": None if self.fault is None else {"code": self.fault.code, "message": self.fault.message},
                "previous_hash": previous_hash,
            }
        )
        self.step_counter += 1
        self.receipts.append(receipt)
        return receipt

    def run(self, max_steps: int) -> list[ExecutionReceipt]:
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        made: list[ExecutionReceipt] = []
        for _ in range(max_steps):
            if self.halted or self.fault is not None:
                break
            made.append(self.step())
        if not self.halted and self.fault is None and len(made) == max_steps:
            self.fault = BOGFault("INSTRUCTION_BUDGET_EXHAUSTED", f"budget {max_steps} exhausted")
        return made
