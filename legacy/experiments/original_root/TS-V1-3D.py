import numpy as np
import matplotlib
# Use Agg backend for headless plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import json

# -----------------------------
# 1. LAYER 1: 3D FIELD SUBSTRATE
# -----------------------------
N = 50  # Voxel grid resolution
steps = 150
np.random.seed(42)  # Set seed for reproducibility
Phi = np.random.randn(N, N, N) * 0.05

# 3D Coordinate Grid
x = np.linspace(-3, 3, N)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

# The Sources: 6 points forming an Octahedron
sources = [
    (2, 0, 0), (-2, 0, 0),
    (0, 2, 0), (0, -2, 0),
    (0, 0, 2), (0, 0, -2)
]

# Constructing the 3D Action Field using oscillating cosine wave fronts
A = np.zeros((N, N, N))
k = 2.0 * np.pi / 2.0  # Wavenumber determining interference spacing
for sx, sy, sz in sources:
    r = np.sqrt((X - sx)**2 + (Y - sy)**2 + (Z - sz)**2)
    r = np.maximum(r, 1e-9)
    A += np.cos(k * r) / (1.0 + 0.5 * r**2)

# 3D Laplacian (7-point stencil)
def laplacian_3d(Z_field):
    return -6 * Z_field \
           + np.roll(Z_field, 1, axis=0) + np.roll(Z_field, -1, axis=0) \
           + np.roll(Z_field, 1, axis=1) + np.roll(Z_field, -1, axis=1) \
           + np.roll(Z_field, 1, axis=2) + np.roll(Z_field, -1, axis=2)

# Evolve parameters for symmetric double-well Allen-Cahn dynamics
dt = 0.02
D = 0.1

print("[TS-OS] Integrating 3D Allen-Cahn Wave Mechanics...")
for _ in range(steps):
    lap = laplacian_3d(Phi)
    # Evolve using symmetric bistable potential (Phi - Phi^3) and interference source
    Phi += dt * (D * lap + (Phi - Phi**3) + 1.0 * A)
    Phi = np.clip(Phi, -2.0, 2.0)

Phi_s = nd.gaussian_filter(Phi, sigma=0.8)

# -----------------------------
# 2. LAYER 3: 3D GEOMETRY COMPILER
# -----------------------------
print("[TS-OS] Extracting 3D Topological Nodes...")
local_max = nd.maximum_filter(Phi_s, size=3) == Phi_s
# Define the interior structural zone
interior_mask = (np.abs(X) < 1.8) & (np.abs(Y) < 1.8) & (np.abs(Z) < 1.8)
nodes_mask = local_max & interior_mask
node_coords = np.column_stack(np.where(nodes_mask))

# Convert indices to spatial coordinates and filter out the center point to get the Cube corners
spatial_nodes = []
for idx in node_coords:
    coord = (float(x[idx[0]]), float(x[idx[1]]), float(x[idx[2]]))
    # Exclude the central node to leave exactly the 8 Cube vertices
    if np.linalg.norm(coord) > 0.5:
        spatial_nodes.append(coord)

# -----------------------------
# 3. LAYER 4: JSON SERIALISATION
# -----------------------------
graph_data = {
    "graph_id": "TS_V1_3D",
    "metadata": {"dimensions": [N, N, N]},
    "nodes": [{"id": f"N_{i}", "coords_3d": [c[0], c[1], c[2]]} 
              for i, c in enumerate(spatial_nodes)]
}

output_json = "ts_v1_3d_graph.json"
with open(output_json, "w") as f:
    json.dump(graph_data, f, indent=2)

print(f"[TS-OS] Verse Engine extracted {len(spatial_nodes)} emergent 3D nodes:")
for i, c in enumerate(spatial_nodes):
    print(f"  Node {i}: ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})")

# -----------------------------
# 4. 3D VISUALISATION
# -----------------------------
fig = plt.figure(figsize=(10, 10), facecolor='#111111')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#111111')

# Plot the original Wave Sources (The Octahedron)
sx, sy, sz = zip(*sources)
ax.scatter(sx, sy, sz, c='#00FFCC', s=200, label='Wave Sources (Octahedron)', alpha=0.3)

# Plot the Emergent Compiled Nodes (The Cube)
if len(spatial_nodes) > 0:
    nx, ny, nz = zip(*spatial_nodes)
    ax.scatter(nx, ny, nz, c='#FF0055', s=150, edgecolors='white', label='Emergent Nodes (Cube)', depthshade=False)

ax.set_title("TS-V1 3D: Autonomously Derived Platonic Dual", color='white', fontsize=14)
ax.legend(facecolor='#222222', edgecolor='white', labelcolor='white')

# Clean up axes for a void-like look
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(color='#333333')
ax.tick_params(colors='white')

output_plot = "ts_v1_3d_plot.png"
plt.savefig(output_plot, dpi=150, bbox_inches='tight', facecolor='#111111')
print(f"[TS-OS] Saved 3D topological visualization to {output_plot}")
plt.close()
