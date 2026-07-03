# Baseline Audit

Repository cloned from `https://github.com/BoggersTheFish/TS-OS`.

- Default branch: `main`
- Baseline commit: `d3715020046c9e54de01b478901b4a3ddc764055`
- Python checked: `Python 3.12.3`
- Git checked: `git version 2.43.0`

## Baseline Files

The initial repo contained top-level prototype scripts including `BOGVM-0.py`, `kernel.py`, `bootloader.py`, `shell.py`, `ts_os_runtime.py`, `ts_shell.py`, `ts_de.py`, `core/*`, `utils/*`, `experiments/*`, `competing_processes.bogpk`, `output/constraint_graph.json`, and `ts_v1_3d_graph.json`.

## Commands Attempted

- `python3 BOGVM-0.py`: broken in the base interpreter, `ModuleNotFoundError: No module named 'numpy'`.
- `python3 ts_os_runtime.py`: broken in the base interpreter, `ModuleNotFoundError: No module named 'numpy'`.
- `python3 -m pytest`: broken in the base interpreter, `No module named pytest`.
- `python3 -m compileall .`: confirmed source files compiled.

## Findings

Confirmed working:

- Source files were syntactically valid under `compileall`.

Partially working:

- `requirements.txt` listed NumPy, SciPy, and Matplotlib but omitted Pygame, PyQt6, pytest, packaging metadata, and CLI entry points.
- `bootloader.py` parsed `.bogpk` process seeds but did not validate a typed executable schema.
- `kernel.py` calculated weighted spatial allocation and IPC-like buffers but did not execute process programmes.

Broken:

- README commands used absolute `/home/boggersthefish/...` paths and failed in a clean base interpreter.
- `BOGVM-0.py` depended on `ts_v2_3d_graph.json`, which was gitignored and absent in the clean clone.
- `.bogpk` was used as an unversioned process seed, not a strict package format.

Untested:

- Pygame and PyQt frontends.
- Field evolution numerical claims.
- Topology graph extraction claims.

Concept-only or experimental:

- Wave-triggered instruction execution.
- Fully emergent topology/cube discovery.
- BOGVIS and natural binary display.
- Phase-derived process allocation.
