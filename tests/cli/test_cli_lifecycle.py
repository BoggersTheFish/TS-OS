from __future__ import annotations

from pathlib import Path

from tscli.main import main
from tsos.persistence import load_bogstate


PACKAGE = Path(__file__).resolve().parents[2] / "examples" / "addition_process.bogpkg"


def test_cli_suspend_persists_across_invocations(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert main(["boot", str(PACKAGE)]) == 0
    assert main(["step", "1"]) == 0
    _, kernel = load_bogstate(".tsos/session.bogstate")
    assert kernel.processes[0].machine.pc == 1

    assert main(["suspend", "0"]) == 0
    assert main(["step", "3"]) == 0
    _, kernel = load_bogstate(".tsos/session.bogstate")
    assert kernel.processes[0].status.value == "SUSPENDED"
    assert kernel.processes[0].machine.pc == 1

    assert main(["ps"]) == 0
    assert "SUSPENDED" in capsys.readouterr().out


def test_cli_resume_continues_from_persisted_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert main(["boot", str(PACKAGE)]) == 0
    assert main(["step", "1"]) == 0
    assert main(["suspend", "0"]) == 0
    assert main(["step", "2"]) == 0
    assert main(["resume", "0"]) == 0
    assert main(["run"]) == 0

    _, kernel = load_bogstate(".tsos/session.bogstate")
    process = kernel.processes[0]
    assert process.status.value == "HALTED"
    assert process.machine.registers[2] == 23
    assert process.output_events == [23]


def test_cli_kill_persists_and_prevents_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert main(["boot", str(PACKAGE)]) == 0
    assert main(["kill", "0"]) == 0
    assert main(["run"]) == 0

    _, kernel = load_bogstate(".tsos/session.bogstate")
    process = kernel.processes[0]
    assert process.status.value == "TERMINATED"
    assert process.machine.step_counter == 0
    assert process.output_events == []
