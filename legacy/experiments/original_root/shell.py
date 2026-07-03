import sys
import cmd
import time
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kernel import Kernel, Process, SCALE
from bootloader import load_package

class TSShell(cmd.Cmd):
    intro = (
        "\n=========================================================================\n"
        "       TS-OS KERNEL SHELL v1.0 (Deterministic Cognitive Substrate)\n"
        "=========================================================================\n"
        "Type help or ? to list commands. Type exit or quit to terminate the OS.\n"
    )
    prompt = "ts-os> "

    def __init__(self, kernel: Kernel, lock: threading.Lock):
        super().__init__()
        self.kernel = kernel
        self.lock = lock

    def do_ts_status(self, arg):
        """
        Query process registry status.
        Usage: ts_status
        """
        with self.lock:
            processes = list(self.kernel.processes.values())
        
        if not processes:
            print("No active processes registered.")
            return

        print("\nPID | Priority (Amp) | Territory (Voxels) | State")
        print("------------------------------------------------------")
        for p in processes:
            state = "RUNNING" if p.amplitude > 0 else "SUSPENDED"
            priority = p.amplitude / SCALE
            print(f"{p.id:3d} | {priority:14.2f} | {p.territory_size:18d} | {state}")
        print("")

    def do_ts_spawn(self, arg):
        """
        Spawn a new wave-state process dynamically.
        Usage: ts_spawn [amplitude] [x,y,z]
        Example: ts_spawn 2.5 0.5,-0.5,0.0
        """
        try:
            parts = arg.split()
            if len(parts) < 2:
                print("Error: Missing arguments. Usage: ts_spawn [amplitude] [x,y,z]")
                return

            amplitude_val = float(parts[0])
            coords_str = parts[1]
            coords = [float(c) for c in coords_str.split(',')]
            if len(coords) != 3:
                raise ValueError("Coordinates must be a triplet x,y,z")

            # Scale amplitude to integer fixed-point
            amp_scaled = int(round(amplitude_val * SCALE))

            with self.lock:
                # Generate new unique PID
                new_pid = max(self.kernel.processes.keys()) + 1 if self.kernel.processes else 0
                p = Process(new_pid, coords, amp_scaled, 0, [])
                self.kernel.register_process(p)
                # Force immediate boundary re-calculation
                self.kernel.update_space_multiplexing()

            print(f"[Shell] Successfully spawned Process {new_pid} at {coords} with Priority {amplitude_val:.2f}")
        except Exception as e:
            print(f"Error spawning process: {e}")

    def do_ts_inspect(self, arg):
        """
        Probe field amplitude, curvature, and process ownership at specific coordinates.
        Usage: ts_inspect [x,y,z]
        Example: ts_inspect -0.5,0.0,0.0
        """
        try:
            if not arg:
                print("Error: Missing coordinates. Usage: ts_inspect [x,y,z]")
                return
            
            coords = [float(c) for c in arg.split(',')]
            if len(coords) != 3:
                raise ValueError("Coordinates must be a triplet x,y,z")

            px, py, pz = coords[0], coords[1], coords[2]

            with self.lock:
                # Resolve closest voxel indices
                ix = np.argmin(np.abs(self.kernel.x - px))
                iy = np.argmin(np.abs(self.kernel.x - py))
                iz = np.argmin(np.abs(self.kernel.x - pz))

                # Boundary safety check
                if ix == 0 or ix == self.kernel.N-1 or iy == 0 or iy == self.kernel.N-1 or iz == 0 or iz == self.kernel.N-1:
                    print("Error: Coordinates are on grid boundary. Central differences cannot be evaluated.")
                    return

                # Get ownership and field value
                owner_id = self.kernel.voxel_ownership[ix, iy, iz]
                phi_val = self.kernel.Phi[ix, iy, iz]

                # Compute local Hessian on-the-fly using central differences
                h = self.kernel.dx
                phi_xx = (self.kernel.Phi[ix+1, iy, iz] - 2 * phi_val + self.kernel.Phi[ix-1, iy, iz]) / (h**2)
                phi_yy = (self.kernel.Phi[ix, iy+1, iz] - 2 * phi_val + self.kernel.Phi[ix, iy-1, iz]) / (h**2)
                phi_zz = (self.kernel.Phi[ix, iy, iz+1] - 2 * phi_val + self.kernel.Phi[ix, iy, iz-1]) / (h**2)

                phi_xy = (self.kernel.Phi[ix+1, iy+1, iz] - self.kernel.Phi[ix+1, iy-1, iz] - 
                          self.kernel.Phi[ix-1, iy+1, iz] + self.kernel.Phi[ix-1, iy-1, iz]) / (4 * h**2)
                phi_xz = (self.kernel.Phi[ix+1, iy, iz+1] - self.kernel.Phi[ix+1, iy, iz-1] - 
                          self.kernel.Phi[ix-1, iy, iz+1] + self.kernel.Phi[ix-1, iy, iz-1]) / (4 * h**2)
                phi_yz = (self.kernel.Phi[ix, iy+1, iz+1] - self.kernel.Phi[ix, iy+1, iz-1] - 
                          self.kernel.Phi[ix, iy-1, iz+1] + self.kernel.Phi[ix, iy-1, iz-1]) / (4 * h**2)

                H = np.array([
                    [phi_xx, phi_xy, phi_xz],
                    [phi_xy, phi_yy, phi_yz],
                    [phi_xz, phi_yz, phi_zz]
                ])

                # Solve eigenvalues
                eigvals = np.linalg.eigvalsh(H)

            # Display Probe Report
            print(f"\n--- PROBE REPORT at spatial coords: ({self.kernel.x[ix]:.4f}, {self.kernel.x[iy]:.4f}, {self.kernel.x[iz]:.4f}) ---")
            print(f"  Voxel Index: [{ix}, {iy}, {iz}]")
            print(f"  Field Amplitude (Phi): {phi_val:.6f}")
            print(f"  Curvature Eigenvalues: lambda_1 = {eigvals[0]:.4f}, lambda_2 = {eigvals[1]:.4f}, lambda_3 = {eigvals[2]:.4f}")
            print(f"  Territorial Owner: Process {owner_id} (State: {'RUNNING' if self.kernel.processes[owner_id].amplitude > 0 else 'SUSPENDED'})")
            print("")
        except Exception as e:
            print(f"Error probing coordinates: {e}")

    def do_ts_render(self, arg):
        """
        Dumps the current 3D space multiplexing territory slice map to a visualization file.
        Usage: ts_render
        """
        print("[Shell] Generating 3D territory slice rendering...")
        with self.lock:
            # Re-read grid state
            N = self.kernel.N
            voxel_ownership = self.kernel.voxel_ownership.copy()
            processes = list(self.kernel.processes.values())
            x_mesh = self.kernel.x.copy()

        try:
            plt.figure(figsize=(8, 8), facecolor='#111111')
            ax = plt.axes()
            
            slice_idx = N // 2
            slice_data = voxel_ownership[:, :, slice_idx]
            
            # Map unique process PIDs to colors
            im = plt.imshow(slice_data, extent=[-3, 3, -3, 3], origin='lower', cmap='plasma')
            plt.colorbar(im, label='Process ID Owner')
            
            for p in processes:
                color = 'white' if p.amplitude > 0 else 'red'
                marker = 'o' if p.amplitude > 0 else 'x'
                label = f"Process {p.id} (Active)" if p.amplitude > 0 else f"Process {p.id} (Suspended)"
                plt.scatter(p.pos[1], p.pos[0], color=color, s=150, edgecolors='black', marker=marker, label=label)
                
            plt.title(f"TS-OS Space Multiplexing slice (Z = {x_mesh[slice_idx]:.2f})", color='white', fontsize=13, pad=10)
            plt.xlabel('Y dimension', color='white')
            plt.ylabel('X dimension', color='white')
            ax.tick_params(colors='white')
            plt.legend(facecolor='#222222', edgecolor='white', labelcolor='white')
            
            output_plot = "ts_os_territory.png"
            plt.savefig(output_plot, dpi=150, bbox_inches='tight', facecolor='#111111')
            plt.close()
            print(f"[Shell] Render complete. Saved visualization to {output_plot}")
        except Exception as e:
            print(f"Error rendering space visualization: {e}")

    def do_exit(self, arg):
        """
        Exit the TS-OS Kernel Shell.
        """
        print("[Shell] Terminating Kernel thread...")
        return True

    def do_quit(self, arg):
        """
        Exit the TS-OS Kernel Shell.
        """
        return self.do_exit(arg)

# Asynchronous Background substrate evolution worker
def background_substrate_worker(kernel: Kernel, lock: threading.Lock, stop_event: threading.Event):
    N = kernel.N
    k = 2.0 * np.pi / 2.0
    dt = 0.02
    D = 0.1

    def laplacian_3d(Z_field):
        return -6 * Z_field \
               + np.roll(Z_field, 1, axis=0) + np.roll(Z_field, -1, axis=0) \
               + np.roll(Z_field, 1, axis=1) + np.roll(Z_field, -1, axis=1) \
               + np.roll(Z_field, 1, axis=2) + np.roll(Z_field, -1, axis=2)

    while not stop_event.is_set():
        with lock:
            # 1. Compute process wave sources
            A = np.zeros((N, N, N))
            for p in kernel.processes.values():
                if p.amplitude > 0:
                    r = np.sqrt((kernel.X - p.pos[0])**2 + (kernel.Y - p.pos[1])**2 + (kernel.Z - p.pos[2])**2)
                    r = np.maximum(r, 1e-9)
                    amp_f = p.amplitude / SCALE
                    phase_f = p.phase / SCALE
                    A += amp_f * np.cos(k * r + phase_f) / (1.0 + 0.5 * r**2)

            # 2. Integrate Substrate
            lap = laplacian_3d(kernel.Phi)
            kernel.Phi += dt * (D * lap + (kernel.Phi - kernel.Phi**3) + 1.0 * A)
            kernel.Phi = np.clip(kernel.Phi, -2.0, 2.0)

            # 3. Update spatial ownership mapping
            kernel.update_space_multiplexing()

            # 4. Update IPC boundary buffer
            kernel.resolve_ipc_buffers()

            # 5. Run arbitration
            kernel.rebalance()

        # Relax to prevent lock starvation & CPU hogging
        time.sleep(0.01)

def boot_system():
    # Setup kernel
    N = 50
    kernel = Kernel(N)

    # Ingest baseline competing processes seed package
    seed_file = "competing_processes.bogpk"
    load_package(seed_file, kernel)

    # Thread synchronization primitives
    lock = threading.Lock()
    stop_event = threading.Event()

    # Start background process execution worker
    worker_thread = threading.Thread(
        target=background_substrate_worker,
        args=(kernel, lock, stop_event),
        daemon=True
    )
    worker_thread.start()

    # Run Shell Loop in main thread
    try:
        TSShell(kernel, lock).cmdloop()
    finally:
        # Signal worker thread to exit
        stop_event.set()
        worker_thread.join(timeout=2.0)
        print("[TS-OS] System shutdown complete.")

if __name__ == '__main__':
    boot_system()
