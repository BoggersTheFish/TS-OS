# Milestone Report

## Original State

Baseline clone was branch `main` at `d3715020046c9e54de01b478901b4a3ddc764055`. The repo contained related prototypes but no integrated execution path. `Process.program` was stored but not executed, `.bogpk` had no strict versioned schema, field loops were duplicated, and `BOGVM-0.py` was a standalone pulse-timing demo.

## Failures Discovered

- `python3 BOGVM-0.py` failed before dependency installation because NumPy was missing.
- `python3 ts_os_runtime.py` failed before dependency installation because NumPy was missing.
- `python3 -m pytest` failed because pytest was not installed and no test suite existed.
- README commands used machine-specific absolute paths.
- `BOGVM-0.py` referenced absent/gitignored `ts_v2_3d_graph.json`.

## Architecture Changed

- Added canonical `src/bogvm`, `src/tsos`, `src/tscli`, `src/tsfield`, and `src/tstopology`.
- Added deterministic BOGVM ISA v0, assembler, machine, faults, receipts, and strict formats.
- Integrated BOGVM into `Process` and kernel `tick()`.
- Added CLI-visible boot/run/inspect/register/memory/receipt/verify path.
- Added canonical field runtime and typed topology classifier.
- Moved old one-off scripts and prior-guided experiments under `legacy/experiments/`.

## Files Added

- `pyproject.toml`
- `specs/*.md`
- `docs/baseline_audit.md`
- `docs/architecture.md`
- `docs/milestone_report.md`
- `src/**`
- `tests/**`
- `examples/add_13_10.bogasm`
- `examples/add_13_10.bogexe`
- `examples/addition_process.bogpkg`
- `.github/workflows/ci.yml`

## Files Moved

Old root scripts, `core/`, `utils/`, `experiments/`, generated graph/output files, and the legacy `.bogpk` seed were moved under `legacy/experiments/`.

## Test Counts

Current local result after CLI lifecycle and field-authority tests: `32 passed`.

## Exact Commands Executed

```bash
git clone https://github.com/BoggersTheFish/TS-OS.git TS-OS-runtime-milestone
git branch --show-current
git rev-parse HEAD
python3 --version
python3 BOGVM-0.py
python3 ts_os_runtime.py
python3 -m pytest
python3 -m compileall .
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/mypy src
.venv/bin/ts boot examples/addition_process.bogpkg
.venv/bin/ts run
.venv/bin/ts verify-trace .tsos/traces/pid-0.bogtrace
.venv/bin/ts boot examples/addition_process.bogpkg --driver wave --threshold 0.5
.venv/bin/ts step 1
.venv/bin/ts ps
.venv/bin/ts boot examples/addition_process.bogpkg --driver wave --threshold 0.01
.venv/bin/ts run
```

## Addition Transcript

```text
booted: examples/addition_process.bogpkg
processes: 1
output: 23
pid: 0
state: HALTED
registers: [13, 10, 23, 0]
trace: .tsos/traces/pid-0.bogtrace
trace verification: PASS
PASS
```

Final register state: `[13, 10, 23, 0]`.

Receipt-chain verification result: `PASS`.

## Follow-up Field Authority Patch

After the v0.1 milestone, the CLI replay-only session bug was fixed with `.bogstate` persistence. A first field-authority patch now connects `FieldRuntime` to `Kernel.tick()` and makes `WaveThresholdExecutionDriver` consume measured field samples at process positions.

Observed behavior:

```text
ts boot examples/addition_process.bogpkg --driver wave --threshold 0.5
ts step 1
ts ps
0 READY pc=0 steps=0 outputs=[]

ts boot examples/addition_process.bogpkg --driver wave --threshold 0.01
ts run
output: 23
state: HALTED
trace verification: PASS
```

The high threshold blocks VM execution; the low threshold permits execution from measured field state.

## Remaining Limitations

- The canonical VM remains an ordinary integer VM; field authority currently controls timing, not instruction semantics.
- GUI frontends are archived and not yet adapted to the new CLI/API surface.
- Field runtime is centralized and measured by the kernel, but cross-platform bitwise floating reproducibility is not claimed.
- Topology tests cover Hessian typing and serialization stability, not full saddle-manifold extraction.

## Next Recommended Milestone

Adapt the Pygame/PyQt frontends to consume the canonical `tsos.Kernel` and `tsfield.FieldRuntime`, then add optional headless GUI smoke tests without changing VM semantics.
