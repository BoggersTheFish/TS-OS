import pytest

from bogvm import BOGMachine, assemble_text
from bogvm.assembler import AssemblyError
from bogvm.receipts import verify_receipt_chain


def run(asm: str, steps: int = 20) -> BOGMachine:
    machine = BOGMachine()
    machine.load_program(assemble_text(asm))
    machine.run(steps)
    return machine


def test_assemble_valid_program_and_stable_hash():
    a = assemble_text("LOADI R0, 13\nHALT\n")
    b = assemble_text("LOADI R0, 13\nHALT\n")
    assert a.program_hash == b.program_hash


@pytest.mark.parametrize("asm", ["LOADI R9, 1", "ADD R0, R1", "NOPE"])
def test_reject_malformed_program(asm):
    with pytest.raises((AssemblyError, ValueError)):
        assemble_text(asm)


def test_loadi_mov_arithmetic_output_and_halt_receipts():
    machine = run(
        """
        LOADI R0, 13
        LOADI R1, 10
        ADD R2, R0, R1
        MOV R3, R2
        EMIT R3
        HALT
        """
    )
    assert machine.registers == [13, 10, 23, 23]
    assert machine.outputs == [23]
    assert machine.halted
    assert verify_receipt_chain(machine.receipts)
    assert machine.receipts[2].register_changes == {"R2": [0, 23]}


def test_sub_mul_div_memory_load_store():
    machine = run(
        """
        LOADI R0, 9
        LOADI R1, 3
        SUB R2, R0, R1
        MUL R2, R2, R1
        DIV R2, R2, R1
        STORE R2, 4
        LOAD R3, 4
        HALT
        """
    )
    assert machine.memory[4] == 6
    assert machine.registers[3] == 6


def test_divide_by_zero_fault():
    machine = run("LOADI R0, 1\nDIV R2, R0, R1\n", 10)
    assert machine.fault is not None
    assert machine.fault.code == "DIVIDE_BY_ZERO"
    assert verify_receipt_chain(machine.receipts)


def test_conditional_jump():
    machine = run(
        """
        LOADI R0, 1
        LOADI R1, 1
        CMP R0, R1
        JZ equal
        LOADI R2, 99
        equal:
        LOADI R2, 7
        HALT
        """
    )
    assert machine.registers[2] == 7


def test_instruction_budget_exhaustion():
    machine = BOGMachine()
    machine.load_program(assemble_text("NOP\nJMP 0\n"))
    machine.run(3)
    assert machine.fault is not None
    assert machine.fault.code == "INSTRUCTION_BUDGET_EXHAUSTED"


def test_same_input_same_final_state_and_hashes():
    asm = "LOADI R0, 13\nLOADI R1, 10\nADD R2, R0, R1\nEMIT R2\nHALT\n"
    a = run(asm)
    b = run(asm)
    assert a.snapshot() == b.snapshot()
    assert [r.receipt_hash for r in a.receipts] == [r.receipt_hash for r in b.receipts]
