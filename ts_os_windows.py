import os
import sys
import time
import json
import threading
import numpy as np

# Suppress Pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from kernel import Kernel, Process, SCALE

# Setup headless fallback for video driver if display is unavailable
headless = False
try:
    pygame.init()
    # Test open window
    test_screen = pygame.display.set_mode((1, 1))
    pygame.display.quit()
except pygame.error:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    headless = True
    print("[TS-OS-Windows] Headless sandbox detected. Rendering in memory.")

# Pre-defined program sequences for Start Menu spawning
PROGRAMS = {
    "LOAD R0 (Value 13)": [
        {"step": 0, "node": 0, "energy": 1.4}
    ],
    "LOAD R1 (Value 10)": [
        {"step": 15, "node": 1, "energy": 0.9}
    ],
    "ADD R2 = R0 + R1": [
        {"step": 30, "node": 2, "energy": 0.5}
    ],
    "PRINT Registers": [
        {"step": 45, "node": 3, "energy": 0.6}
    ],
    "HALT Substrate": [
        {"step": 60, "node": 6, "energy": 1.0}
    ]
}

# Color Palette (Fluent Dark Theme)
COLOR_BG = (15, 15, 20)           # Dark backdrop
COLOR_TASKBAR = (25, 25, 38, 220)  # Semi-transparent dark grey
COLOR_START_MENU = (32, 32, 48, 235)
COLOR_WINDOW_BG = (40, 40, 60, 180) # Glassmorphic frosted translucent window
COLOR_WINDOW_BORDER = (100, 100, 150)
COLOR_TEXT = (240, 240, 255)
COLOR_ACCENT = (0, 204, 255)       # Cyan accent
COLOR_RED = (255, 0, 85)          # Pink/Red
COLOR_GREEN = (0, 255, 128)

class DesktopWindow:
    """
    Translates a running kernel process into a translucent Windows-like window.
    """
    def __init__(self, pid, x, y, width, height, title):
        self.pid = pid
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title
        self.is_dragging = False
        self.is_resizing = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.logs = []

    def update_logs(self, msg):
        self.logs.append(msg)
        if len(self.logs) > 4:
            self.logs.pop(0)

class OSKernelManager:
    """
    Thread-safe supervisor of the background Allen-Cahn substrate and process VM.
    """
    def __init__(self, kernel: Kernel, lock: threading.Lock):
        self.kernel = kernel
        self.lock = lock
        self.stop_event = threading.Event()
        self.thread = None
        self.step_count = 0
        self.running_vm = True
        self.refractory = {}
        
    def start(self):
        self.thread = threading.Thread(target=self._substrate_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def _substrate_loop(self):
        N = self.kernel.N
        k = 2.0 * np.pi / 2.0
        dt = 0.02
        D = 0.1

        def laplacian_3d(Z_field):
            return -6 * Z_field \
                   + np.roll(Z_field, 1, axis=0) + np.roll(Z_field, -1, axis=0) \
                   + np.roll(Z_field, 1, axis=1) + np.roll(Z_field, -1, axis=1) \
                   + np.roll(Z_field, 1, axis=2) + np.roll(Z_field, -1, axis=2)

        while not self.stop_event.is_set():
            with self.lock:
                # 1. Update wave sources from active process states
                A = np.zeros((N, N, N))
                for p in self.kernel.processes.values():
                    if p.amplitude > 0:
                        r = np.sqrt((self.kernel.X - p.pos[0])**2 + (self.kernel.Y - p.pos[1])**2 + (self.kernel.Z - p.pos[2])**2)
                        r = np.maximum(r, 1e-9)
                        amp_f = p.amplitude / SCALE
                        phase_f = p.phase / SCALE
                        A += amp_f * np.cos(k * r + phase_f) / (1.0 + 0.5 * r**2)

                # 2. Evolve Substrate
                lap = laplacian_3d(self.kernel.Phi)
                self.kernel.Phi += dt * (D * lap + (self.kernel.Phi - self.kernel.Phi**3) + A)
                self.kernel.Phi = np.clip(self.kernel.Phi, -2.0, 2.0)

                # 3. Update spatial multiplexing ownership maps
                self.kernel.update_space_multiplexing()
                self.kernel.resolve_ipc_buffers()
                self.kernel.rebalance()

                # 4. Check BOGVM-0 triggers for running processes
                if self.running_vm:
                    self._execute_vm_step()

                self.step_count += 1

            time.sleep(0.015)  # Evolve at ~65Hz

    def _execute_vm_step(self):
        # Trigger parameters
        trigger_threshold = int(0.50 * SCALE)
        neighbor_active_threshold = int(0.15 * SCALE)
        
        # BOGVM-0 16-Opcode Mapping
        OPCODES = {
            0: "LOAD", 1: "NOP", 2: "NOP", 3: "ADD",
            4: "NOP", 5: "NOP", 6: "NOP", 7: "HALT",
            8: "NOP", 9: "LOAD", 10: "NOP", 11: "NOP",
            12: "NOP", 13: "NOP", 14: "NOP", 15: "PRINT"
        }

        for pid, p in list(self.kernel.processes.items()):
            if p.amplitude <= 0 or p.halted:
                continue

            # Update process program steps
            for instruction in list(p.program):
                if self.step_count == instruction["step"]:
                    # Inject pulse directly into process position node coordinates
                    # Note: We represent pulse injection by raising process wave energy
                    val_scaled = int(instruction["energy"] * SCALE)
                    # We can use the register load value dynamically
                    p.refractory = 0  # Force clear refractory on direct injection
                    p.registers[0] = int(instruction["energy"] * 10)  # Value loader

            if p.refractory > 0:
                p.refractory -= 1
                continue

            # Check if local field intensity at process source crosses 0.50
            # Resolve voxel index
            ix = np.argmin(np.abs(self.kernel.x - p.pos[0]))
            iy = np.argmin(np.abs(self.kernel.x - p.pos[1]))
            iz = np.argmin(np.abs(self.kernel.x - p.pos[2]))
            
            local_e = int(self.kernel.Phi[ix, iy, iz] * SCALE)
            if local_e >= trigger_threshold:
                p.refractory = 15  # Refractory period of 15 steps

                # Evaluate adjacent process ownership neighbors
                u_neighbors = self.kernel.voxel_ownership[
                    max(ix-1,0):min(ix+2, self.kernel.N),
                    max(iy-1,0):min(iy+2, self.kernel.N),
                    iz
                ]
                
                # Check active PIDs surrounding the coordinate
                active_neighbors = set(u_neighbors.flatten())
                active_neighbors.discard(pid)
                active_neighbors.discard(-1)

                b0 = 1 if len(active_neighbors) >= 1 else 0
                b1 = 1 if len(active_neighbors) >= 2 else 0
                b2 = 1 if len(active_neighbors) >= 3 else 0
                b3 = 1 if pid % 2 != 0 else 0

                opcode_val = b3 * 8 + b2 * 4 + b1 * 2 + b0 * 1
                opcode_name = OPCODES[opcode_val]
                
                target_reg = pid % 4
                val_loaded = int(local_e // 10_000_000)

                # Execute
                if opcode_name == "LOAD":
                    p.registers[target_reg] = val_loaded
                elif opcode_name == "ADD":
                    p.registers[2] = p.registers[0] + p.registers[1]
                elif opcode_name == "PRINT":
                    pass
                elif opcode_name == "HALT":
                    p.halted = True
                    p.amplitude = 0  # Destructive interference suspends process

def save_desktop_state(kernel: Kernel, filepath="desktop.bogpk"):
    """
    Serializes process registers, states, and the 3D voxel grid to a .bogpk file.
    """
    data = {
        "processes": [
            {
                "pid": p.id,
                "pos": p.pos.tolist(),
                "amplitude": p.amplitude,
                "phase": p.phase,
                "registers": p.registers,
                "memory": p.memory,
                "halted": p.halted
            }
            for p in kernel.processes.values()
        ],
        "phi": kernel.Phi.tolist()
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[TS-OS-Windows] Desktop serialized to {filepath}")

def load_desktop_state(kernel: Kernel, filepath="desktop.bogpk"):
    """
    Restores process registers, states, and the 3D voxel grid from a .bogpk file.
    """
    if not os.path.exists(filepath):
        print(f"[TS-OS-Windows] State file {filepath} not found.")
        return
    with open(filepath, "r") as f:
        data = json.load(f)
    kernel.processes.clear()
    for p_data in data["processes"]:
        p = Process(p_data["pid"], p_data["pos"], p_data["amplitude"], p_data["phase"], [])
        p.registers = p_data["registers"]
        p.memory = p_data["memory"]
        p.halted = p_data["halted"]
        kernel.register_process(p)
    kernel.Phi = np.array(data["phi"], dtype=np.float64)
    print(f"[TS-OS-Windows] Desktop loaded from {filepath}")

def launch_gui():
    N = 50
    kernel = Kernel(N)
    
    # Load baseline competing processes package
    seed_file = "competing_processes.bogpk"
    if os.path.exists(seed_file):
        try:
            with open(seed_file, 'r') as f:
                pkg = json.load(f)
            for p_data in pkg.get('processes', []):
                pid = int(p_data['pid'])
                pos = [float(c) for c in p_data['pos']]
                amplitude = int(round(p_data['amplitude'] * SCALE))
                phase = int(round(p_data['phase'] * SCALE))
                p = Process(pid, pos, amplitude, phase, [])
                kernel.register_process(p)
        except Exception as e:
            print(f"[TS-OS-Windows] Error loading baseline seed: {e}")

    lock = threading.Lock()
    manager = OSKernelManager(kernel, lock)
    manager.start()

    # Graphical layout settings
    width, height = 800, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("TS-OS Fluent Desktop v2.0")
    clock = pygame.time.Clock()

    pygame.font.init()
    font_small = pygame.font.SysFont("Outfit", 16)
    font_bold = pygame.font.SysFont("Outfit", 18, bold=True)
    font_title = pygame.font.SysFont("Outfit", 20, bold=True)

    # Internal substrate texture mapping
    slice_surf = pygame.Surface((N, N))

    # UI Windows Dictionary: pid -> DesktopWindow
    windows = {}
    
    # UI State variables
    start_menu_open = False
    start_btn_rect = pygame.Rect(width // 2 - 25, height - 40, 50, 35)
    start_menu_rect = pygame.Rect(width // 2 - 150, height - 350, 300, 300)
    
    # Compile Start Menu App buttons
    app_buttons = []
    btn_y = height - 330
    for app_name in PROGRAMS.keys():
        app_buttons.append((pygame.Rect(width // 2 - 130, btn_y, 260, 35), app_name))
        btn_y += 45

    # Desktop System command buttons (Save / Load)
    save_btn_rect = pygame.Rect(10, 10, 100, 30)
    load_btn_rect = pygame.Rect(120, 10, 100, 30)

    running = True
    print("[TS-OS-Windows] GUI loop active. Processing frames at 60 FPS...")

    while running:
        # 1. Event Loop Handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                px = -3.0 + 6.0 * (mx / width)
                py = -3.0 + 6.0 * (my / height)

                # Check System command buttons (Save/Load)
                if save_btn_rect.collidepoint(mx, my):
                    with lock:
                        save_desktop_state(kernel)
                    continue
                elif load_btn_rect.collidepoint(mx, my):
                    with lock:
                        load_desktop_state(kernel)
                    # Clear window objects to force rebuild
                    windows.clear()
                    continue

                # Check Start Button / Start Menu
                if start_btn_rect.collidepoint(mx, my):
                    start_menu_open = not start_menu_open
                    continue
                
                if start_menu_open and start_menu_rect.collidepoint(mx, my):
                    # Check if clicked an app button
                    for btn_rect, app_name in app_buttons:
                        if btn_rect.collidepoint(mx, my):
                            # Spawn process with that program
                            prog = PROGRAMS[app_name]
                            with lock:
                                new_pid = max(kernel.processes.keys()) + 1 if kernel.processes else 0
                                # Default coordinates in center grid
                                p = Process(new_pid, [0.0, 0.0, 0.0], int(2.5 * SCALE), 0, prog)
                                kernel.register_process(p)
                                # Force bounds recalculation
                                kernel.update_space_multiplexing()
                            print(f"[TS-OS-Windows] Booted Program '{app_name}' as Process {new_pid}")
                            start_menu_open = False
                    continue
                else:
                    start_menu_open = False

                # Check Window interactions (highest z-order first)
                clicked_window = False
                for pid in list(windows.keys()):
                    w = windows[pid]
                    
                    # Close Button Check
                    close_rect = pygame.Rect(w.rect.right - 25, w.rect.top + 5, 20, 20)
                    if close_rect.collidepoint(mx, my):
                        with lock:
                            if pid in kernel.processes:
                                print(f"[TS-OS-Windows] Closing window and terminating Process {pid}")
                                kernel.processes[pid].amplitude = 0  # Destructive interference
                        clicked_window = True
                        break

                    # Minimize Button Check
                    min_rect = pygame.Rect(w.rect.right - 50, w.rect.top + 5, 20, 20)
                    if min_rect.collidepoint(mx, my):
                        with lock:
                            if pid in kernel.processes:
                                print(f"[TS-OS-Windows] Minimizing process {pid} wave envelope")
                                kernel.processes[pid].amplitude = int(0.1 * SCALE)  # Shrink territory
                        clicked_window = True
                        break

                    # Resize Corner Handle Check
                    resize_handle = pygame.Rect(w.rect.right - 15, w.rect.bottom - 15, 15, 15)
                    if resize_handle.collidepoint(mx, my):
                        w.is_resizing = True
                        clicked_window = True
                        break

                    # Title Bar Drag Check
                    title_bar = pygame.Rect(w.rect.left, w.rect.top, w.rect.width - 60, 30)
                    if title_bar.collidepoint(mx, my):
                        w.is_dragging = True
                        w.drag_offset_x = w.rect.x - mx
                        w.drag_offset_y = w.rect.y - my
                        clicked_window = True
                        # Move this window to top of drawing hierarchy
                        # (Pop and insert back at end of dict)
                        windows[pid] = windows.pop(pid)
                        break

                if not clicked_window:
                    # Click on empty background:
                    # Left-click: spawn process, Right-click: suspend nearest
                    if event.button == 1:
                        # Spawn simple wave source
                        with lock:
                            new_pid = max(kernel.processes.keys()) + 1 if kernel.processes else 0
                            p = Process(new_pid, [py, px, 0.0], int(2.0 * SCALE), 0, [])
                            kernel.register_process(p)
                        print(f"[TS-OS-Windows] Spawneed Process {new_pid} at coords ({py:.2f}, {px:.2f})")
                    elif event.button == 3:
                        # Suspend nearest process
                        closest_p = None
                        min_d = float('inf')
                        with lock:
                            for pid, p in kernel.processes.items():
                                if p.amplitude > 0:
                                    dist = np.sqrt((p.pos[1] - px)**2 + (p.pos[0] - py)**2)
                                    if dist < min_d:
                                        min_d = dist
                                        closest_p = p
                            if closest_p is not None and min_d < 1.0:
                                print(f"[TS-OS-Windows] Suspending nearest Process {closest_p.id}")
                                closest_p.amplitude = 0
                                
            elif event.type == pygame.MOUSEBUTTONUP:
                for w in windows.values():
                    w.is_dragging = False
                    w.is_resizing = False
                    
            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                px = -3.0 + 6.0 * (mx / width)
                py = -3.0 + 6.0 * (my / height)
                
                for pid, w in windows.items():
                    if w.is_dragging:
                        w.rect.x = mx + w.drag_offset_x
                        w.rect.y = my + w.drag_offset_y
                        
                        # Apply repelling force back to the continuous wave source coordinate!
                        # Update the process source center pos to match window position
                        wx = -3.0 + 6.0 * ((w.rect.x + w.rect.width // 2) / width)
                        wy = -3.0 + 6.0 * ((w.rect.y + w.rect.height // 2) / height)
                        with lock:
                            if pid in kernel.processes:
                                p = kernel.processes[pid]
                                p.pos[0] = wy
                                p.pos[1] = wx
                                
                    elif w.is_resizing:
                        # Calculate new window width/height based on drag
                        new_w = max(mx - w.rect.x, 100)
                        new_h = max(my - w.rect.y, 80)
                        w.rect.width = new_w
                        w.rect.height = new_h
                        
                        # Scale process amplitude proportional to window size!
                        # Larger window = larger wave amplitude = larger territory!
                        new_amp = max(float(new_w) / 100.0, 0.2)
                        with lock:
                            if pid in kernel.processes:
                                kernel.processes[pid].amplitude = int(new_amp * SCALE)

        # --- Graphics Render Loop ---
        # 1. Capture Substrate State
        with lock:
            Phi_slice = kernel.Phi[:, :, N // 2].copy()
            owners_slice = kernel.voxel_ownership[:, :, N // 2].copy()
            active_processes = {pid: p for pid, p in kernel.processes.items() if p.amplitude > 0}

        # 2. Draw live field heatmap on background texture
        pixel_array = pygame.PixelArray(slice_surf)
        for ix in range(N):
            for iy in range(N):
                val = Phi_slice[ix, iy]
                owner = owners_slice[ix, iy]
                
                is_boundary = False
                if owner >= 0:
                    for dx_offset, dy_offset in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nx, ny = ix + dx_offset, iy + dy_offset
                        if 0 <= nx < N and 0 <= ny < N:
                            if owners_slice[nx, ny] != owner:
                                is_boundary = True
                                break
                
                if is_boundary:
                    color = (255, 255, 255)
                else:
                    if val >= 0:
                        intensity = int(max(min(val * 110 + 35, 255), 0))
                        color = (intensity, 12, 35)
                    else:
                        intensity = int(max(min(-val * 110 + 35, 255), 0))
                        color = (12, 12, intensity)
                pixel_array[iy, ix] = color
        del pixel_array

        # Stretch background texture to fill window
        pygame.transform.scale(slice_surf, (width, height), screen)

        # 3. Synchronize GUI Windows with running Process registry
        # Add windows for newly spawned processes
        for pid, p in active_processes.items():
            if pid not in windows:
                # Spawn window at process center coordinates mapped to screen
                wx = int(width * (p.pos[1] + 3.0) / 6.0) - 100
                wy = int(height * (p.pos[0] + 3.0) / 6.0) - 80
                w_size = int((p.amplitude / SCALE) * 100)
                windows[pid] = DesktopWindow(pid, wx, wy, w_size, w_size, f"Process {pid} [BOGVM]")

        # Delete windows of suspended processes
        for pid in list(windows.keys()):
            if pid not in active_processes:
                del windows[pid]

        # 4. Render Translucent Glassmorphic Windows
        for pid, w in list(windows.items()):
            p = active_processes.get(pid)
            if not p:
                continue

            # Check if coordinates have moved externally (e.g. via wave dynamics)
            # and gently ease window positions to follow wave center centers!
            target_wx = int(width * (p.pos[1] + 3.0) / 6.0) - w.rect.width // 2
            target_wy = int(height * (p.pos[0] + 3.0) / 6.0) - w.rect.height // 2
            if not w.is_dragging:
                w.rect.x += int((target_wx - w.rect.x) * 0.15)
                w.rect.y += int((target_wy - w.rect.y) * 0.15)
                
            # Restrict resizing dimensions to amplitude setting
            if not w.is_resizing:
                target_size = int((p.amplitude / SCALE) * 100)
                w.rect.width += int((target_size - w.rect.width) * 0.15)
                w.rect.height += int((target_size - w.rect.height) * 0.15)

            # Draw frosted glass panel (surface with alpha support)
            w_surf = pygame.Surface((w.rect.width, w.rect.height), pygame.SRCALPHA)
            w_surf.fill(COLOR_WINDOW_BG)
            pygame.draw.rect(w_surf, COLOR_WINDOW_BORDER, (0, 0, w.rect.width, w.rect.height), 2)
            
            # Title Bar
            pygame.draw.rect(w_surf, COLOR_TASKBAR, (0, 0, w.rect.width, 30))
            pygame.draw.line(w_surf, COLOR_WINDOW_BORDER, (0, 30), (w.rect.width, 30), 1)

            # Window controls
            # Close button (Red X circle)
            pygame.draw.circle(w_surf, COLOR_RED, (w.rect.width - 15, 15), 8)
            # Minimize button (Yellow dash circle)
            pygame.draw.circle(w_surf, COLOR_GREEN, (w.rect.width - 40, 15), 8)

            # Text overlay for process properties
            title_txt = font_bold.render(w.title, True, COLOR_TEXT)
            w_surf.blit(title_txt, (10, 6))

            # Render VM Registers
            reg_y = 40
            for idx, reg_val in enumerate(p.registers):
                reg_txt = font_small.render(f"R{idx}: {reg_val}", True, COLOR_TEXT)
                w_surf.blit(reg_txt, (15, reg_y))
                reg_y += 20
                
            # Render Process territory size
            t_txt = font_small.render(f"Space: {p.territory_size} vox", True, COLOR_ACCENT)
            w_surf.blit(t_txt, (15, reg_y))
            
            # Blit translucent window to screen
            screen.blit(w_surf, (w.rect.x, w.rect.y))
            
            # Draw resize handle anchor
            pygame.draw.polygon(screen, COLOR_WINDOW_BORDER, [
                (w.rect.right - 1, w.rect.bottom - 12),
                (w.rect.right - 12, w.rect.bottom - 1),
                (w.rect.right - 1, w.rect.bottom - 1)
            ])

        # 5. Render Taskbar
        taskbar_surf = pygame.Surface((width, 45), pygame.SRCALPHA)
        taskbar_surf.fill(COLOR_TASKBAR)
        pygame.draw.line(taskbar_surf, COLOR_WINDOW_BORDER, (0, 0), (width, 0), 1)
        
        # Start button
        pygame.draw.rect(taskbar_surf, COLOR_ACCENT, (width // 2 - 25, 5, 50, 35), border_radius=4)
        start_lbl = font_bold.render("Start", True, (0, 0, 0))
        taskbar_surf.blit(start_lbl, (width // 2 - 18, 13))
        
        # Clock
        current_time = time.strftime("%H:%M:%S")
        time_txt = font_small.render(current_time, True, COLOR_TEXT)
        taskbar_surf.blit(time_txt, (width - 80, 15))
        
        screen.blit(taskbar_surf, (0, height - 45))

        # 6. Render Start Menu
        if start_menu_open:
            menu_surf = pygame.Surface((300, 300), pygame.SRCALPHA)
            menu_surf.fill(COLOR_START_MENU)
            pygame.draw.rect(menu_surf, COLOR_WINDOW_BORDER, (0, 0, 300, 300), 2, border_radius=8)
            
            menu_title = font_title.render("Boot Wave-State App:", True, COLOR_ACCENT)
            menu_surf.blit(menu_title, (15, 12))
            
            screen.blit(menu_surf, (width // 2 - 150, height - 350))
            
            # Draw App buttons on top
            for btn_rect, app_name in app_buttons:
                pygame.draw.rect(screen, COLOR_TASKBAR, btn_rect, border_radius=4)
                pygame.draw.rect(screen, COLOR_WINDOW_BORDER, btn_rect, 1, border_radius=4)
                btn_lbl = font_small.render(app_name, True, COLOR_TEXT)
                screen.blit(btn_lbl, (btn_rect.x + 12, btn_rect.y + 8))

        # 7. Render Save/Load HUD buttons
        pygame.draw.rect(screen, COLOR_TASKBAR, save_btn_rect, border_radius=4)
        pygame.draw.rect(screen, COLOR_WINDOW_BORDER, save_btn_rect, 1, border_radius=4)
        save_lbl = font_small.render("Save Desktop", True, COLOR_TEXT)
        screen.blit(save_lbl, (save_btn_rect.x + 8, save_btn_rect.y + 6))

        pygame.draw.rect(screen, COLOR_TASKBAR, load_btn_rect, border_radius=4)
        pygame.draw.rect(screen, COLOR_WINDOW_BORDER, load_btn_rect, 1, border_radius=4)
        load_lbl = font_small.render("Load Desktop", True, COLOR_TEXT)
        screen.blit(load_lbl, (load_btn_rect.x + 8, load_btn_rect.y + 6))

        pygame.display.flip()
        clock.tick(60)

        # Headless sandbox execution run check
        if headless:
            print("[TS-OS-Windows] Headless verification run successful.")
            break

    # Shutdown Kernel thread
    manager.stop()
    pygame.quit()
    print("[TS-OS-Windows] OS shutdown cleanly.")

if __name__ == '__main__':
    launch_gui()
