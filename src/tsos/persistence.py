from __future__ import annotations

from pathlib import Path
from typing import Any

from bogvm.faults import BOGFault
from bogvm.formats import bogstate_json, read_json
from bogvm.receipts import ExecutionReceipt

from .bootloader import boot_package
from .kernel import Kernel, ProcessStatus


def kernel_to_bogstate(kernel: Kernel, package_path: str) -> dict[str, Any]:
    state = kernel.state()
    state["package_path"] = package_path
    for pid, process in kernel.processes.items():
        state["processes"][str(pid)]["machine"]["receipts"] = [
            receipt.to_json() for receipt in process.machine.receipts
        ]
    state["kernel_receipts"] = list(kernel.kernel_receipts)
    return bogstate_json(state)


def kernel_from_bogstate(obj: dict[str, Any]) -> tuple[str, Kernel]:
    if obj.get("format") != "bogstate" or obj.get("schema_version") != 0:
        raise ValueError("unsupported bogstate")
    state = obj["state"]
    package_path = str(state["package_path"])
    kernel = boot_package(package_path)
    kernel.runtime_tick = int(state["runtime_tick"])
    kernel.kernel_receipts = list(state.get("kernel_receipts", []))

    for pid_text, process_state in state["processes"].items():
        pid = int(pid_text)
        process = kernel.processes[pid]
        process.status = ProcessStatus(process_state["status"])
        process.territory_size = int(process_state.get("territory_size", 0))
        process.output_events = [int(v) for v in process_state.get("outputs", [])]

        machine_state = process_state["machine"]
        machine = process.machine
        machine.registers = [int(v) for v in machine_state["registers"]]
        machine.memory = [int(v) for v in machine_state["memory"]]
        machine.pc = int(machine_state["pc"])
        machine.zero_flag = bool(machine_state["zero_flag"])
        machine.halted = bool(machine_state["halted"])
        fault = machine_state["fault"]
        machine.fault = None if fault is None else BOGFault(str(fault["code"]), str(fault["message"]))
        machine.step_counter = int(machine_state["step_counter"])
        machine.outputs = [int(v) for v in machine_state.get("outputs", [])]
        machine.receipts = [
            ExecutionReceipt.from_payload(receipt) for receipt in machine_state.get("receipts", [])
        ]

    kernel.allocator.allocate(kernel)
    return package_path, kernel


def save_bogstate(path: str | Path, kernel: Kernel, package_path: str) -> None:
    from bogvm.formats import write_canonical_json

    write_canonical_json(path, kernel_to_bogstate(kernel, package_path))


def load_bogstate(path: str | Path) -> tuple[str, Kernel]:
    return kernel_from_bogstate(read_json(path))


def package_to_kernel(path: str | Path) -> Kernel:
    return boot_package(path, Kernel())
