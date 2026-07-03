import json
import os
from kernel import Kernel, Process, SCALE

def load_package(filepath: str, kernel: Kernel) -> None:
    """
    Reads a .bogpk process package file (JSON format) and registers
    the processes into the TS-OS Kernel.
    """
    print(f"[Bootloader] Ingesting process package: {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Process package file {filepath} not found.")

    with open(filepath, 'r') as f:
        pkg = json.load(f)

    for p_data in pkg.get('processes', []):
        pid = int(p_data['pid'])
        pos = [float(c) for c in p_data['pos']]
        # Convert float amplitude & phase to integer fixed-point
        amplitude = int(round(p_data['amplitude'] * SCALE))
        phase = int(round(p_data['phase'] * SCALE))
        program = p_data.get('program', [])

        p = Process(pid, pos, amplitude, phase, program)
        kernel.register_process(p)

    print("[Bootloader] System Initialization state established successfully.")
