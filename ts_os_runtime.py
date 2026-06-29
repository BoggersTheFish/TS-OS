import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kernel import Kernel, SCALE
from bootloader import load_package

def run_runtime():
    print("=========================================================================")
    print("[TS-OS] BOOTING TS-OS KERNEL CORE...")
    print("=========================================================================")
    
    # Initialize Kernel
    N = 50
    kernel = Kernel(N)
    
    # Load competing processes from package seed
    package_file = "competing_processes.bogpk"
    load_package(package_file, kernel)

    # Compile mesh coordinates for wave propagation
    k = 2.0 * np.pi / 2.0  # Wave frequency

    # Evolve parameters
    dt = 0.02
    D = 0.1
    steps = 50

    def laplacian_3d(Z_field):
        return -6 * Z_field \
               + np.roll(Z_field, 1, axis=0) + np.roll(Z_field, -1, axis=0) \
               + np.roll(Z_field, 1, axis=1) + np.roll(Z_field, -1, axis=1) \
               + np.roll(Z_field, 1, axis=2) + np.roll(Z_field, -1, axis=2)

    # Initial boundary and territory mapping
    kernel.update_space_multiplexing()
    print("\n[TS-OS] Initial Space-Multiplexing Territories:")
    for pid, p in kernel.processes.items():
        print(f"  Process {pid}: {p.territory_size} voxels")

    print("\n[TS-OS] Entering Runtime Execution Loop...")
    for step in range(1, steps + 1):
        # 1. Calculate active wave sources based on current process amplitudes/phases
        A = np.zeros((N, N, N))
        for pid, p in kernel.processes.items():
            if p.amplitude > 0:
                # Distance to process source
                r = np.sqrt((kernel.X - p.pos[0])**2 + (kernel.Y - p.pos[1])**2 + (kernel.Z - p.pos[2])**2)
                r = np.maximum(r, 1e-9)
                amp_f = p.amplitude / SCALE
                phase_f = p.phase / SCALE
                # decaying isotropic wave source
                A += amp_f * np.cos(k * r + phase_f) / (1.0 + 0.5 * r**2)

        # 2. Evolve continuous field substrate
        lap = laplacian_3d(kernel.Phi)
        kernel.Phi += dt * (D * lap + (kernel.Phi - kernel.Phi**3) + 1.0 * A)
        kernel.Phi = np.clip(kernel.Phi, -2.0, 2.0)

        # 3. Update space-multiplexing and territory ownership
        kernel.update_space_multiplexing()

        # 4. Resolve Saddle IPC buffers
        kernel.resolve_ipc_buffers()

        # 5. Kernel arbitration check
        kernel.rebalance()

        # Log status every 10 steps
        if step % 10 == 0 or step == 1:
            active_pids = [pid for pid, p in kernel.processes.items() if p.amplitude > 0]
            print(f"  Step {step:2d} | Active PIDs: {active_pids}")
            for pid, p in kernel.processes.items():
                status = "RUNNING" if p.amplitude > 0 else "SUSPENDED"
                print(f"    Process {pid} ({status}): Territory = {p.territory_size} voxels")
            print(f"    IPC Buffers: {kernel.ipc_buffer}")

    print("\n[TS-OS] Execution cycle completed.")
    print("=========================================================================")
    print("[TS-OS] Final System State:")
    for pid, p in kernel.processes.items():
        status = "RUNNING" if p.amplitude > 0 else "SUSPENDED"
        print(f"  Process {pid} ({status}): Territory = {p.territory_size} voxels")
    print(f"  IPC Memory: {kernel.ipc_buffer}")
    print("=========================================================================")

    # 6. Save a 2D slice visual of the space multiplexing
    plt.figure(figsize=(8, 8), facecolor='#111111')
    ax = plt.axes()
    
    # Take a 2D slice at Z midpoint (z = N // 2)
    slice_idx = N // 2
    slice_data = kernel.voxel_ownership[:, :, slice_idx]
    
    # Plot the slice
    im = plt.imshow(slice_data, extent=[-3, 3, -3, 3], origin='lower', cmap='plasma')
    plt.colorbar(im, label='Process ID Owner', ticks=[0, 1])
    
    # Plot process source positions
    for pid, p in kernel.processes.items():
        color = 'white' if p.amplitude > 0 else 'red'
        marker = 'o' if p.amplitude > 0 else 'x'
        label = f"Process {pid} (Active)" if p.amplitude > 0 else f"Process {pid} (Suspended)"
        plt.scatter(p.pos[1], p.pos[0], color=color, s=150, edgecolors='black', marker=marker, label=label)
        
    plt.title(f"TS-OS Space Multiplexing slice (Z = {kernel.x[slice_idx]:.2f})", color='white', fontsize=13, pad=10)
    plt.xlabel('Y dimension', color='white')
    plt.ylabel('X dimension', color='white')
    ax.tick_params(colors='white')
    plt.legend(facecolor='#222222', edgecolor='white', labelcolor='white')
    
    output_plot = "ts_os_territory.png"
    plt.savefig(output_plot, dpi=150, bbox_inches='tight', facecolor='#111111')
    print(f"[TS-OS] Saved territorial slice map to {output_plot}")
    plt.close()

if __name__ == '__main__':
    run_runtime()
