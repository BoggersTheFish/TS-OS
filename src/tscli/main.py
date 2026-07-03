from __future__ import annotations

import argparse
import json
from pathlib import Path

from bogvm.formats import read_json, trace_to_json, validate_trace_json, write_canonical_json
from bogvm.receipts import verify_receipt_chain
from tsos.bootloader import boot_package
from tsos.persistence import load_bogstate, save_bogstate

SESSION = Path(".tsos/session.bogstate")
TRACE_DIR = Path(".tsos/traces")


def _save_session(package: str, kernel) -> None:
    SESSION.parent.mkdir(exist_ok=True)
    save_bogstate(SESSION, kernel, package)


def _load_session():
    if not SESSION.exists():
        raise SystemExit("no booted package; run: ts boot <package>")
    return load_bogstate(SESSION)


def _write_traces(kernel) -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    for pid, process in kernel.processes.items():
        write_canonical_json(TRACE_DIR / f"pid-{pid}.bogtrace", trace_to_json(process.machine.receipts))


def cmd_boot(args) -> int:
    kernel = boot_package(args.package)
    _save_session(args.package, kernel)
    _write_traces(kernel)
    print(f"booted: {args.package}")
    print(f"processes: {len(kernel.processes)}")
    return 0


def cmd_ps(args) -> int:
    _, kernel = _load_session()
    for pid, p in kernel.processes.items():
        print(f"{pid}\t{p.status.value}\tpc={p.machine.pc}\tsteps={p.machine.step_counter}\toutputs={p.output_events}")
    return 0


def cmd_step(args) -> int:
    package, kernel = _load_session()
    for _ in range(args.count):
        kernel.tick()
    _save_session(package, kernel)
    _write_traces(kernel)
    print(f"runtime_tick: {kernel.runtime_tick}")
    for pid, p in kernel.processes.items():
        if p.output_events:
            print(f"pid {pid} output: {p.output_events[-1]}")
        print(f"pid {pid} state: {p.status.value}")
    return 0


def cmd_run(args) -> int:
    package, kernel = _load_session()
    kernel.run(args.max_steps)
    _save_session(package, kernel)
    _write_traces(kernel)
    for pid, p in kernel.processes.items():
        for value in p.output_events:
            print(f"output: {value}")
        print(f"pid: {pid}")
        print(f"state: {p.status.value}")
        print(f"registers: {p.machine.registers}")
        print(f"trace: {TRACE_DIR / f'pid-{pid}.bogtrace'}")
        print(f"trace verification: {'PASS' if verify_receipt_chain(p.machine.receipts) else 'FAIL'}")
    return 0


def cmd_inspect(args) -> int:
    _, kernel = _load_session()
    p = kernel.processes[int(args.pid)]
    print(json.dumps(kernel.state()["processes"][str(p.pid)], indent=2, sort_keys=True))
    return 0


def cmd_registers(args) -> int:
    _, kernel = _load_session()
    print(kernel.processes[int(args.pid)].machine.registers)
    return 0


def cmd_memory(args) -> int:
    _, kernel = _load_session()
    print(kernel.processes[int(args.pid)].machine.memory)
    return 0


def cmd_receipts(args) -> int:
    _, kernel = _load_session()
    for receipt in kernel.processes[int(args.pid)].machine.receipts:
        print(json.dumps(receipt.to_json(), sort_keys=True))
    return 0


def cmd_verify_trace(args) -> int:
    valid = validate_trace_json(read_json(args.path))
    print("PASS" if valid else "FAIL")
    return 0 if valid else 1


def cmd_status_change(args, action: str) -> int:
    package, kernel = _load_session()
    getattr(kernel, action)(int(args.pid))
    _save_session(package, kernel)
    print(f"pid {args.pid}: {action.upper()}")
    return 0


def cmd_field_status(args) -> int:
    _, kernel = _load_session()
    print(f"field_enabled: {kernel.field_enabled}")
    print(f"grid: {kernel.N}x{kernel.N}x{kernel.N}")
    return 0


def cmd_graph_status(args) -> int:
    print("graph: unavailable")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ts")
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("boot")
    p.add_argument("package")
    p.set_defaults(func=cmd_boot)
    sub.add_parser("ps").set_defaults(func=cmd_ps)
    p = sub.add_parser("step")
    p.add_argument("count", nargs="?", type=int, default=1)
    p.set_defaults(func=cmd_step)
    p = sub.add_parser("run")
    p.add_argument("max_steps", nargs="?", type=int, default=100)
    p.set_defaults(func=cmd_run)
    for name, func in [("inspect", cmd_inspect), ("registers", cmd_registers), ("memory", cmd_memory), ("receipts", cmd_receipts)]:
        p = sub.add_parser(name)
        p.add_argument("pid")
        p.set_defaults(func=func)
    p = sub.add_parser("verify-trace")
    p.add_argument("path")
    p.set_defaults(func=cmd_verify_trace)
    for name in ["suspend", "resume", "kill"]:
        p = sub.add_parser(name)
        p.add_argument("pid")
        p.set_defaults(func=lambda args, n=name: cmd_status_change(args, n))
    sub.add_parser("field-status").set_defaults(func=cmd_field_status)
    sub.add_parser("graph-status").set_defaults(func=cmd_graph_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
