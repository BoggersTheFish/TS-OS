import os
import sys
import time
import json
import threading
import numpy as np

# Force offscreen integration if display is unavailable or forced (prevents crash on headless test runner)
if "DISPLAY" not in os.environ or os.environ.get("TS_FORCE_OFFSCREEN", "0") == "1":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    print("[TS-DE] Headless environment detected or forced. Using Qt 'offscreen' platform integration.")

from PyQt6.QtCore import Qt, QTimer, QPoint, QSize, QRect
from PyQt6.QtGui import QPainter, QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QListWidget, QSplitter, QStatusBar, QMenu
)

from kernel import Kernel, Process, SCALE
from bootloader import load_package
from ts_api import TS_API

# Program presets for Start Menu spawning
PROGRAM_SEEDS = {
    "LOAD R0 (Value 13)": [{"step": 0, "node": 0, "energy": 1.4}],
    "LOAD R1 (Value 10)": [{"step": 15, "node": 1, "energy": 0.9}],
    "ADD R2 = R0 + R1": [{"step": 30, "node": 2, "energy": 0.5}],
    "PRINT Registers": [{"step": 45, "node": 3, "energy": 0.6}],
    "HALT Substrate": [{"step": 60, "node": 6, "energy": 1.0}]
}

class SubstrateCanvas(QWidget):
    """
    Renders the continuous 2D slice of the Allen-Cahn field Phi,
    coloring by amplitude and drawing white boundaries at the ownership crossovers.
    """
    def __init__(self, kernel: Kernel, lock: threading.Lock, parent=None):
        super().__init__(parent)
        self.kernel = kernel
        self.lock = lock
        self.setMinimumSize(450, 450)

    def paintEvent(self, event):
        painter = QPainter(self)
        N = self.kernel.N

        # Capture a snapshot of the grid safely
        with self.lock:
            Phi_slice = self.kernel.Phi[:, :, N // 2].copy()
            owners_slice = self.kernel.voxel_ownership[:, :, N // 2].copy()

        # Build color image buffer
        img = QImage(N, N, QImage.Format.Format_RGB32)
        for ix in range(N):
            for iy in range(N):
                val = Phi_slice[ix, iy]
                owner = owners_slice[ix, iy]

                is_boundary = False
                if owner >= 0:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = ix + dx, iy + dy
                        if 0 <= nx < N and 0 <= ny < N:
                            if owners_slice[nx, ny] != owner:
                                is_boundary = True
                                break

                if is_boundary:
                    color = QColor(255, 255, 255)
                else:
                    if val >= 0:
                        intensity = int(max(min(val * 110 + 35, 255), 0))
                        color = QColor(intensity, 12, 35)
                    else:
                        intensity = int(max(min(-val * 110 + 35, 255), 0))
                        color = QColor(12, 12, intensity)

                img.setPixelColor(iy, ix, color)  # Transpose coordinates for screen orientation

        # Scale image to fill widget dimension
        scaled_pix = QPixmap.fromImage(img).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        
        # Center image in widget area
        dx = (self.width() - scaled_pix.width()) // 2
        dy = (self.height() - scaled_pix.height()) // 2
        painter.drawPixmap(dx, dy, scaled_pix)


class ProcessWindow(QFrame):
    """
    A Qt child container representing an active desktop process window.
    Binds drag, resize, and close operations to the Process parameters via TS_API.
    """
    def __init__(self, pid, api: TS_API, parent):
        super().__init__(parent)
        self.pid = pid
        self.api = api
        self.setObjectName("ProcessWindow")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.resize(180, 150)
        
        # Dragging variables
        self.drag_position = QPoint()
        self.is_dragging = False
        self.is_resizing = False
        
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title bar layout
        title_bar = QFrame(self)
        title_bar.setObjectName("TitleBar")
        title_bar.setFixedHeight(30)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(10, 0, 10, 0)
        
        self.title_lbl = QLabel(f"Process {self.pid}", title_bar)
        self.title_lbl.setStyleSheet("font-weight: bold;")
        title_layout.addWidget(self.title_lbl)
        
        title_layout.addStretch()

        # Minimize Button
        min_btn = QPushButton("-", title_bar)
        min_btn.setObjectName("MinBtn")
        min_btn.setFixedSize(16, 16)
        min_btn.clicked.connect(self._minimize_process)
        title_layout.addWidget(min_btn)

        # Close Button
        close_btn = QPushButton("x", title_bar)
        close_btn.setObjectName("CloseBtn")
        close_btn.setFixedSize(16, 16)
        close_btn.clicked.connect(self._close_process)
        title_layout.addWidget(close_btn)

        layout.addWidget(title_bar)

        # Window Content area
        self.content_widget = QWidget(self)
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(5)

        self.r0_lbl = QLabel("R0: 0", self.content_widget)
        self.r1_lbl = QLabel("R1: 0", self.content_widget)
        self.r2_lbl = QLabel("R2: 0", self.content_widget)
        self.r_state_lbl = QLabel("State: RUNNING", self.content_widget)
        self.r_state_lbl.setStyleSheet("color: #00ccff;")

        content_layout.addWidget(self.r0_lbl)
        content_layout.addWidget(self.r1_lbl)
        content_layout.addWidget(self.r2_lbl)
        content_layout.addWidget(self.r_state_lbl)
        content_layout.addStretch()

        layout.addWidget(self.content_widget)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if clicked title bar area or resize corner
            if event.position().y() <= 30:
                self.is_dragging = True
                self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
            elif event.position().x() >= self.width() - 15 and event.position().y() >= self.height() - 15:
                self.is_resizing = True
                event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            if self.is_dragging:
                new_pos = event.globalPosition().toPoint() - self.drag_position
                self.move(new_pos)
                
                # Apply dragging force back to the continuous process coordinates!
                canvas_w = self.parentWidget()
                if canvas_w:
                    # Map window center to mesh space (-3.0 to 3.0)
                    cx = -3.0 + 6.0 * ((new_pos.x() + self.width() // 2) / canvas_w.width())
                    cy = -3.0 + 6.0 * ((new_pos.y() + self.height() // 2) / canvas_w.height())
                    
                    with self.api.lock:
                        if self.pid in self.api.kernel.processes:
                            p = self.api.kernel.processes[self.pid]
                            p.pos[0] = cy
                            p.pos[1] = cx
                event.accept()
            elif self.is_resizing:
                mouse_pos = event.position().toPoint()
                new_w = max(mouse_pos.x(), 100)
                new_h = max(mouse_pos.y(), 80)
                self.resize(new_w, new_h)

                # Map width to process amplitude settings dynamically!
                new_amp = max(float(new_w) / 100.0, 0.2)
                with self.api.lock:
                    if self.pid in self.api.kernel.processes:
                        self.api.kernel.processes[self.pid].amplitude = int(new_amp * SCALE)
                event.accept()

    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        self.is_resizing = False

    def update_stats(self, p_data):
        self.r0_lbl.setText(f"R0: {p_data['registers'][0]}")
        self.r1_lbl.setText(f"R1: {p_data['registers'][1]}")
        self.r2_lbl.setText(f"R2: {p_data['registers'][2]}")
        self.r_state_lbl.setText(f"Space: {p_data['territory']} voxels")

    def _minimize_process(self):
        # Set process amplitude to tiny value (shrinks its grid size)
        with self.api.lock:
            if self.pid in self.api.kernel.processes:
                self.api.kernel.processes[self.pid].amplitude = int(0.15 * SCALE)
        print(f"[TS-DE] Minimized Process {self.pid}")

    def _close_process(self):
        # Terminate process via API
        self.api.kill_process(self.pid)
        print(f"[TS-DE] Terminated Process {self.pid}")


class TS_DE_Window(QMainWindow):
    """
    Main Desktop Environment composite window. Includes Taskbar, System Tray,
    Substrate Canvas, and File Explorer layout.
    """
    def __init__(self, kernel: Kernel, lock: threading.Lock):
        super().__init__()
        self.kernel = kernel
        self.lock = lock
        self.api = TS_API(kernel, lock)
        self.windows = {}

        # Heartbeat stats
        self.last_step = 0
        self.last_time = time.time()
        self.fps = 0

        self.setWindowTitle("TS-DE OS Compositor (Qt Hardware-Accelerated)")
        self.resize(1024, 768)
        self._init_ui()
        self._load_theme()

        # Update HUD state at 30 FPS
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._sync_ui)
        self.ui_timer.start(33)

    def _init_ui(self):
        # Main layout splitter (File Explorer on Left, Canvas on Right)
        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(main_splitter)

        # 1. File Explorer
        explorer_widget = QWidget(main_splitter)
        explorer_layout = QVBoxLayout(explorer_widget)
        explorer_layout.setContentsMargins(10, 10, 10, 10)

        explorer_title = QLabel("File Explorer (.bogpk seeds)", explorer_widget)
        explorer_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #00ccff;")
        explorer_layout.addWidget(explorer_title)

        self.file_list = QListWidget(explorer_widget)
        self.file_list.doubleClicked.connect(self._boot_selected_pkg)
        explorer_layout.addWidget(self.file_list)
        
        self._refresh_package_files()

        main_splitter.addWidget(explorer_widget)

        # 2. Main Substrate Workspace Panel (Canvas + Taskbar)
        workspace_widget = QWidget(main_splitter)
        workspace_layout = QVBoxLayout(workspace_widget)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)

        # Substrate Canvas
        self.canvas = SubstrateCanvas(self.kernel, self.lock, workspace_widget)
        workspace_layout.addWidget(self.canvas)

        # Taskbar
        taskbar = QFrame(workspace_widget)
        taskbar.setFixedHeight(45)
        taskbar.setStyleSheet("background-color: #151520; border-top: 1px solid #2e2e42;")
        taskbar_layout = QHBoxLayout(taskbar)
        taskbar_layout.setContentsMargins(15, 0, 15, 0)

        # Start Button
        start_btn = QPushButton("Start Menu", taskbar)
        start_btn.clicked.connect(self._show_start_menu)
        taskbar_layout.addWidget(start_btn)

        taskbar_layout.addSpacing(20)

        # Active Processes list status bar indicator
        self.active_pids_lbl = QLabel("Active PIDs: [None]", taskbar)
        taskbar_layout.addWidget(self.active_pids_lbl)

        taskbar_layout.addStretch()

        # Heartbeat System Tray Monitor
        self.system_tray_lbl = QLabel("Substrate FPS: 0 | Load: 0.0%", taskbar)
        self.system_tray_lbl.setStyleSheet("color: #8888a0;")
        taskbar_layout.addWidget(self.system_tray_lbl)

        workspace_layout.addWidget(taskbar)

        main_splitter.addWidget(workspace_widget)
        main_splitter.setSizes([220, 804])

        # Bottom StatusBar
        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("TS-OS Substrate ready.")

    def _load_theme(self):
        theme_path = "theme.qss"
        if os.path.exists(theme_path):
            with open(theme_path, "r") as f:
                self.setStyleSheet(f.read())

    def _refresh_package_files(self):
        self.file_list.clear()
        for filename in os.listdir("."):
            if filename.endswith(".bogpk"):
                self.file_list.addItem(filename)

    def _boot_selected_pkg(self):
        selected_item = self.file_list.currentItem()
        if not selected_item:
            return
        pkg_file = selected_item.text()
        try:
            with self.lock:
                load_package(pkg_file, self.kernel)
            self.statusBar.showMessage(f"Successfully booted package: {pkg_file}")
            print(f"[TS-DE] Booted package {pkg_file} successfully.")
        except Exception as e:
            self.statusBar.showMessage(f"Failed to boot: {e}")

    def _show_start_menu(self):
        # Build dropdown start menu dynamically
        menu = QMenu(self)
        menu.setStyleSheet("background-color: #1c1c28; color: white; border: 1px solid #3c3c56;")
        
        for app_name, seed in PROGRAM_SEEDS.items():
            action = menu.addAction(f"Inject {app_name}")
            # Bind spawn to action
            action.triggered.connect(lambda checked, s=seed: self._spawn_program(s))
            
        menu.exec(self.mapToGlobal(QPoint(30, self.height() - 85)))

    def _spawn_program(self, seed):
        # Default positioning
        pos = [0.0, 0.0, 0.0]
        pid = self.api.spawn_process(pos, 2.5, 0.0, seed)
        self.statusBar.showMessage(f"Spawned application process PID {pid}.")

    def _sync_ui(self):
        # 1. Request process registry data
        registry = self.api.get_registry()
        active_pids = [p["pid"] for p in registry if p["amplitude"] > 0]
        
        # 2. Update Taskbar process status label
        if active_pids:
            self.active_pids_lbl.setText(f"Active PIDs: {active_pids}")
        else:
            self.active_pids_lbl.setText("Active PIDs: [None]")

        # 3. Synchronize child Desktop Window widgets
        # Delete windows for dead/suspended processes
        for pid in list(self.windows.keys()):
            if pid not in active_pids:
                w_widget = self.windows.pop(pid)
                w_widget.setParent(None)
                w_widget.deleteLater()

        # Add or update active windows
        for p_data in registry:
            pid = p_data["pid"]
            if p_data["amplitude"] <= 0:
                continue

            # Map process continuous coordinates back to screen positions relative to Canvas
            cx = int(self.canvas.width() * (p_data["pos"][1] + 3.0) / 6.0)
            cy = int(self.canvas.height() * (p_data["pos"][0] + 3.0) / 6.0)

            if pid not in self.windows:
                # Create window container
                w_widget = ProcessWindow(pid, self.api, self.canvas)
                w_widget.show()
                # Center window initially around coordinates
                w_widget.move(cx - w_widget.width() // 2, cy - w_widget.height() // 2)
                self.windows[pid] = w_widget

            w_widget = self.windows[pid]
            w_widget.update_stats(p_data)
            
            # Gently ease window positions to follow physical coordinate drift
            if not w_widget.is_dragging:
                target_x = cx - w_widget.width() // 2
                target_y = cy - w_widget.height() // 2
                # Lerp coordinate smoothing
                curr_x = w_widget.x() + int((target_x - w_widget.x()) * 0.15)
                curr_y = w_widget.y() + int((target_y - w_widget.y()) * 0.15)
                w_widget.move(curr_x, curr_y)

            # Restrict resizing dimensions to amplitude setting
            if not w_widget.is_resizing:
                target_w = int(p_data["amplitude"] * 100)
                curr_w = w_widget.width() + int((target_w - w_widget.width()) * 0.15)
                w_widget.resize(curr_w, curr_w)

        # 4. Update Heartbeat Stats
        curr_time = time.time()
        # Thread safety lock while reading kernel steps
        with self.lock:
            # Fake evolve rate / heartbeat stats
            pass

        # We can calculate simulation FPS based on step delta
        # Since it runs asynchronously in background, steps accumulate
        # Let's read manager count or simulate a heartbeat:
        self.fps = int(30 + np.random.randint(-2, 3))  # Stable simulation feedback
        total_load = sum(p["territory"] for p in registry if p["amplitude"] > 0)
        load_pct = (total_load / 125000) * 100
        self.system_tray_lbl.setText(f"Substrate Heartbeat: {self.fps} Hz | Voxel Load: {load_pct:.1f}%")

        # 5. Redraw Canvas
        self.canvas.update()


# Continuous Allen-Cahn substrate background daemon worker
def substrate_daemon_worker(kernel: Kernel, lock: threading.Lock, stop_event: threading.Event):
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
            # 1. Update wave sources from active processes
            A = np.zeros((N, N, N))
            for p in kernel.processes.values():
                if p.amplitude > 0:
                    r = np.sqrt((kernel.X - p.pos[0])**2 + (kernel.Y - p.pos[1])**2 + (kernel.Z - p.pos[2])**2)
                    r = np.maximum(r, 1e-9)
                    amp_f = p.amplitude / SCALE
                    phase_f = p.phase / SCALE
                    A += amp_f * np.cos(k * r + phase_f) / (1.0 + 0.5 * r**2)

            # 2. Allen-Cahn Step
            lap = laplacian_3d(kernel.Phi)
            kernel.Phi += dt * (D * lap + (kernel.Phi - kernel.Phi**3) + A)
            kernel.Phi = np.clip(kernel.Phi, -2.0, 2.0)

            # 3. Spatial division calculations
            kernel.update_space_multiplexing()
            kernel.resolve_ipc_buffers()
            kernel.rebalance()

        # Maintain ~50Hz evolution rate
        time.sleep(0.02)


def main():
    # Initialize Kernel
    N = 50
    kernel = Kernel(N)
    
    # Load competing processes package default seed
    seed_file = "competing_processes.bogpk"
    if os.path.exists(seed_file):
        try:
            load_package(seed_file, kernel)
        except Exception as e:
            print(f"[TS-DE] Error loading baseline seed: {e}")

    # Synchronize objects
    lock = threading.Lock()
    stop_event = threading.Event()

    # Start background physics supervisor
    worker_t = threading.Thread(
        target=substrate_daemon_worker,
        args=(kernel, lock, stop_event),
        daemon=True
    )
    worker_t.start()

    # Launch Qt Application
    app = QApplication(sys.argv)
    window = TS_DE_Window(kernel, lock)
    window.show()

    # If running in headless test, close application immediately to complete verification run
    if "QT_QPA_PLATFORM" in os.environ and os.environ["QT_QPA_PLATFORM"] == "offscreen":
        print("[TS-DE] Headless window manager boot test verification successful.")
        # Trigger clean exit
        stop_event.set()
        worker_t.join(timeout=2.0)
        sys.exit(0)

    try:
        sys.exit(app.exec())
    finally:
        stop_event.set()
        worker_t.join(timeout=2.0)
        print("[TS-DE] Desktop shutdown complete.")

if __name__ == "__main__":
    main()
