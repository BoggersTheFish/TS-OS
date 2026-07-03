# TS-OS

TS-OS is now anchored by a deterministic BOGVM runtime path:

`.bogpkg -> .bogexe validation -> bootloader -> Process -> BOGVM step -> registers/memory -> kernel state -> receipt chain -> CLI`

## Implemented Now

- Strict canonical JSON `.bogexe`, `.bogpkg`, `.bogstate`, `.boggraph`, and `.bogtrace` format helpers.
- Text assembler for BOGVM ISA v0.
- Deterministic integer BOGVM core with four registers, fixed memory, faults, outputs, halt state, and SHA-256 receipt chains.
- Kernel `tick()` that actually executes process VM instructions through a clocked execution driver.
- Process statuses: `CREATED`, `READY`, `RUNNING`, `WAITING`, `SUSPENDED`, `HALTED`, `FAULTED`, `TERMINATED`.
- CLI commands for boot, run, step, inspect, registers, memory, receipts, trace verification, suspend, resume, kill, field status, and graph status.
- Shared Allen-Cahn field runtime with explicit `NEUMANN` and `PERIODIC` boundary modes.
- Topology candidate typing using measured gradient and Hessian signatures.

## Experimental / Legacy

Archived under `legacy/experiments/`:

- wave-triggered execution;
- prior PyGame and PyQt desktop demonstrations;
- phase-derived process allocation experiments;
- prior cube/topology scripts with geometry-specific assumptions;
- BOGVIS and natural binary display experiments.

These are preserved as research history, not canonical runtime implementation.

## Quick Start

```bash
git clone https://github.com/BoggersTheFish/TS-OS.git
cd TS-OS
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ts boot examples/addition_process.bogpkg
ts run
```

Expected result:

```text
output: 23
pid: 0
state: HALTED
registers: [13, 10, 23, 0]
trace verification: PASS
```

## Development Checks

```bash
pytest
ruff check src tests
mypy src
```

GUI dependencies are optional:

```bash
pip install -e ".[gui]"
```

## Specifications

- `specs/bogvm-isa-v0.md`
- `specs/bog-object-formats-v0.md`
- `specs/determinism-v0.md`
