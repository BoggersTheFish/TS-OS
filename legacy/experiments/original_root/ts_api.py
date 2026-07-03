import numpy as np
from kernel import Kernel, Process, SCALE

class TS_API:
    """
    Developer API Abstraction Layer for the TS-OS Kernel.
    Allows user-space apps to manage processes without writing differential equations.
    """
    def __init__(self, kernel: Kernel, lock):
        self.kernel = kernel
        self.lock = lock

    def spawn_process(self, pos, amplitude, phase=0.0, program=None):
        """
        Spawns a new process source in the 3D grid substrate.
        :param pos: (x, y, z) list of floats
        :param amplitude: float representing process priority/size
        :param phase: float representing starting wave phase
        :param program: list of BOGVM-0 program instructions
        :return: new process ID (int)
        """
        if program is None:
            program = []
            
        amp_scaled = int(round(amplitude * SCALE))
        phase_scaled = int(round(phase * SCALE))
        
        with self.lock:
            # Find next unique PID
            new_pid = max(self.kernel.processes.keys()) + 1 if self.kernel.processes else 0
            p = Process(new_pid, pos, amp_scaled, phase_scaled, program)
            self.kernel.register_process(p)
            # Re-calculate spatial divisions
            self.kernel.update_space_multiplexing()
            
        return new_pid

    def kill_process(self, pid):
        """
        Suspends/kills a process by setting its priority (amplitude) to 0.
        """
        with self.lock:
            if pid in self.kernel.processes:
                self.kernel.processes[pid].amplitude = 0
                self.kernel.update_space_multiplexing()
                return True
        return False

    def get_registry(self):
        """
        Returns a serialized list of currently registered processes.
        """
        with self.lock:
            registry = []
            for pid, p in self.kernel.processes.items():
                registry.append({
                    "pid": p.id,
                    "pos": [float(p.pos[0]), float(p.pos[1]), float(p.pos[2])],
                    "amplitude": float(p.amplitude / SCALE),
                    "phase": float(p.phase / SCALE),
                    "registers": list(p.registers),
                    "memory": list(p.memory),
                    "territory": int(p.territory_size),
                    "state": "RUNNING" if p.amplitude > 0 else "SUSPENDED",
                    "halted": bool(p.halted)
                })
        return registry
