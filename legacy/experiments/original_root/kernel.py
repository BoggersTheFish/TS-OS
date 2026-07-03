import numpy as np

SCALE = 100_000_000

class Process:
    """
    Represents an active wave-state process running on the TS-OS Kernel.
    Each process has its own location, priority (amplitude), and virtual machine registers.
    """
    def __init__(self, pid, pos, amplitude, phase, program):
        self.id = pid
        self.pos = np.array(pos, dtype=np.float64)  # (x, y, z)
        self.amplitude = int(amplitude)            # Scaled fixed-point
        self.phase = int(phase)                    # Scaled fixed-point
        self.program = program                      # List of (step, node_offset, energy_float)
        
        # BOGVM-0 registers and memory
        self.registers = [0, 0, 0, 0]
        self.memory = [0] * 16
        self.refractory = 0
        self.halted = False
        self.territory_size = 0

    def load_val(self, reg, val):
        self.registers[reg] = val

    def add(self):
        self.registers[2] = self.registers[0] + self.registers[1]

    def print_state(self):
        print(f"    [Process {self.id}] Registers: R={self.registers} | Territory: {self.territory_size} voxels")

class Kernel:
    """
    TS-OS Kernel
    Manages the 3D voxel space, partitions it among active processes,
    resolves IPC memory zones (Saddle Tension Buffers), and performs resource arbitration.
    """
    def __init__(self, N=50):
        self.N = N
        self.x = np.linspace(-3, 3, N)
        self.dx = self.x[1] - self.x[0]
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.x, self.x, indexing='ij')
        
        # Persisted 3D Field Substrate
        self.Phi = np.zeros((N, N, N), dtype=np.float64)
        
        self.processes = {}
        self.voxel_ownership = -np.ones((N, N, N), dtype=np.int32)
        self.ipc_buffer = {}  # (pid_a, pid_b) -> value (shared memory)

    def register_process(self, p: Process):
        self.processes[p.id] = p
        print(f"[Kernel] Registered Process {p.id} at {p.pos} | Priority (Amp): {p.amplitude / SCALE:.2f}")

    def update_space_multiplexing(self):
        """
        Dynamically calculates the territorial boundary for each process.
        Uses a weighted Voronoi calculation: voxel belongs to process p minimizing ||x - x_p|| / amp_p.
        """
        if not self.processes:
            return

        pids = list(self.processes.keys())
        num_p = len(pids)
        
        # Calculate distances to all process sources
        dists = np.zeros((num_p, self.N, self.N, self.N))
        for idx, pid in enumerate(pids):
            p = self.processes[pid]
            if p.amplitude <= 0:
                dists[idx] = 1e9  # Dead or suspended processes have infinite distance metric
                continue
                
            dist_sq = (self.X - p.pos[0])**2 + (self.Y - p.pos[1])**2 + (self.Z - p.pos[2])**2
            dist = np.sqrt(dist_sq)
            
            # Weighted Voronoi distance metric: dist / (amplitude_float)
            amp_float = max(p.amplitude / SCALE, 1e-5)
            dists[idx] = dist / amp_float

        # Assign ownership to the process with minimum distance metric
        self.voxel_ownership = np.argmin(dists, axis=0)
        
        # Map indices back to actual process IDs
        ownership_map = np.array(pids)
        self.voxel_ownership = ownership_map[self.voxel_ownership]

        # Update territory sizes for all processes
        for pid in pids:
            p = self.processes[pid]
            if p.amplitude <= 0:
                p.territory_size = 0
            else:
                p.territory_size = int(np.sum(self.voxel_ownership == pid))

    def resolve_ipc_buffers(self):
        """
        Inter-Process Communication (IPC): Saddle Tension Buffer.
        Identifies boundary voxels between processes A and B.
        Boundary occurs where the distance metrics for two processes are very close.
        """
        pids = list(self.processes.keys())
        if len(pids) < 2:
            return

        # Calculate distances again for boundary check
        dists = []
        active_pids = []
        for pid in pids:
            p = self.processes[pid]
            if p.amplitude > 0:
                dist_sq = (self.X - p.pos[0])**2 + (self.Y - p.pos[1])**2 + (self.Z - p.pos[2])**2
                dist = np.sqrt(dist_sq)
                amp_float = p.amplitude / SCALE
                dists.append(dist / amp_float)
                active_pids.append(pid)

        if len(dists) < 2:
            return

        dists = np.array(dists)  # Shape: (num_active, N, N, N)
        sorted_dists = np.sort(dists, axis=0)
        
        # Boundary threshold: difference in metric is less than 0.15
        diff = sorted_dists[1] - sorted_dists[0]
        boundary_mask = diff < 0.15
        
        # Find which processes share the boundary
        # For each boundary voxel, find the top 2 closest processes
        closest_indices = np.argsort(dists, axis=0)
        
        # Vectorized identification of boundary owners
        p0 = np.array(active_pids)[closest_indices[0]]
        p1 = np.array(active_pids)[closest_indices[1]]
        
        # Find unique boundary pairs
        boundary_pairs = np.unique(np.stack([p0[boundary_mask], p1[boundary_mask]], axis=-1), axis=0)
        
        for pair in boundary_pairs:
            pid_a, pid_b = int(pair[0]), int(pair[1])
            if pid_a == pid_b:
                continue
            pair_key = tuple(sorted([pid_a, pid_b]))
            
            # Segment boundary mask for this specific pair
            pair_mask = boundary_mask & (((p0 == pid_a) & (p1 == pid_b)) | ((p0 == pid_b) & (p1 == pid_a)))
            
            # The IPC value is the average field value (tension) in this boundary segment
            if np.any(pair_mask):
                ipc_val = np.mean(self.Phi[pair_mask])
                # Quantize/scale IPC value to fixed-point integer
                self.ipc_buffer[pair_key] = int(round(ipc_val * SCALE))

    def rebalance(self):
        """
        Resource Arbitration:
        If a process's territory size is crushed below a minimum volume (e.g. 800 voxels)
        due to pressure from a high-priority process, apply destructive interference
        by flatlining its amplitude (effectively suspending or killing it).
        """
        min_volume = 800
        for pid, p in self.processes.items():
            if p.amplitude > 0 and p.territory_size < min_volume:
                print(f"[Kernel] Arbitration: Process {p.id} territory ({p.territory_size} voxels) fell below minimum limit ({min_volume}).")
                print(f"[Kernel] Applying destructive interference. Process {p.id} SUSPENDED.")
                p.amplitude = 0
                p.territory_size = 0
