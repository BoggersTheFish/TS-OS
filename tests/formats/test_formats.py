import copy

import pytest

from bogvm.assembler import assemble_text
from bogvm.formats import BogExe, BogPkg, FormatValidationError, ProcessDefinition, trace_to_json, validate_trace_json
from bogvm.machine import BOGMachine


def exe():
    return BogExe.from_program(assemble_text("LOADI R0, 1\nHALT\n"))


def test_valid_bogexe_roundtrip():
    obj = exe().to_json()
    assert BogExe.from_json(obj).program().program_hash == obj["program_hash"]


def test_invalid_version_missing_field_and_unknown_opcode():
    obj = exe().to_json()
    obj["schema_version"] = 99
    with pytest.raises(FormatValidationError):
        BogExe.from_json(obj)
    obj = exe().to_json()
    del obj["symbols"]
    with pytest.raises(FormatValidationError):
        BogExe.from_json(obj)
    obj = exe().to_json()
    obj["instructions"][0]["opcode"] = "BOOM"
    with pytest.raises(ValueError):
        BogExe.from_json(obj)


def test_invalid_register_and_tampered_program_hash():
    obj = exe().to_json()
    obj["instructions"][0]["operands"][0] = "R9"
    with pytest.raises(ValueError):
        BogExe.from_json(obj)
    obj = exe().to_json()
    obj["program_hash"] = "0" * 64
    with pytest.raises(FormatValidationError):
        BogExe.from_json(obj)


def test_valid_bogpkg_and_invalid_embedded_executable():
    package = BogPkg("pkg", (ProcessDefinition(0, exe(), (0, 0, 0)),)).to_json()
    assert BogPkg.from_json(package).package_id == "pkg"
    bad = copy.deepcopy(package)
    bad["processes"][0]["executable"]["program_hash"] = "bad"
    with pytest.raises(FormatValidationError):
        BogPkg.from_json(bad)


def test_tampered_receipt_chain():
    machine = BOGMachine()
    machine.load_program(assemble_text("LOADI R0, 1\nHALT\n"))
    machine.run(10)
    trace = trace_to_json(machine.receipts)
    assert validate_trace_json(trace)
    trace["receipts"][0]["pc_after"] = 99
    assert not validate_trace_json(trace)
