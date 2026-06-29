import numpy as np
import scipy.ndimage as nd

N = 50
steps = 150
np.random.seed(42)
Phi = np.random.randn(N, N, N) * 0.05

x = np.linspace(-3, 3, N)
dx = x[1] - x[0]
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

sources = [
    (2, 0, 0), (-2, 0, 0),
    (0, 2, 0), (0, -2, 0),
    (0, 0, 2), (0, 0, -2)
]

A = np.zeros((N, N, N))
k = 2.0 * np.pi / 2.0
for sx, sy, sz in sources:
    r = np.sqrt((X - sx)**2 + (Y - sy)**2 + (Z - sz)**2)
    r = np.maximum(r, 1e-9)
    A += np.cos(k * r) / (1.0 + 0.5 * r**2)

def laplacian_3d(Z_field):
    return -6 * Z_field \
           + np.roll(Z_field, 1, axis=0) + np.roll(Z_field, -1, axis=0) \
           + np.roll(Z_field, 1, axis=1) + np.roll(Z_field, -1, axis=1) \
           + np.roll(Z_field, 1, axis=2) + np.roll(Z_field, -1, axis=2)

dt = 0.02
D = 0.1

for _ in range(steps):
    lap = laplacian_3d(Phi)
    Phi += dt * (D * lap + (Phi - Phi**3) + 1.0 * A)
    Phi = np.clip(Phi, -2.0, 2.0)

Phi_s = nd.gaussian_filter(Phi, sigma=0.8)

grad_x = np.gradient(Phi_s, axis=0) / dx
grad_y = np.gradient(Phi_s, axis=1) / dx
grad_z = np.gradient(Phi_s, axis=2) / dx

# Extract Nodes (Cube Vertices)
local_max = nd.maximum_filter(Phi_s, size=3) == Phi_s
interior_mask = (np.abs(X) < 1.8) & (np.abs(Y) < 1.8) & (np.abs(Z) < 1.8)
node_coords = np.column_stack(np.where(local_max & interior_mask))

nodes = []
node_id = 0
for idx in node_coords:
    coord = (float(x[idx[0]]), float(x[idx[1]]), float(x[idx[2]]))
    if np.linalg.norm(coord) > 0.5:  # Exclude center node
        nodes.append({
            'id': node_id,
            'pos': coord,
            'grid_idx': (int(idx[0]), int(idx[1]), int(idx[2]))
        })
        node_id += 1

def interpolate_3d(field, px, py, pz):
    px = np.clip(px, x[0], x[-1])
    py = np.clip(py, x[0], x[-1])
    pz = np.clip(pz, x[0], x[-1])
    
    tx = (px - x[0]) / dx
    ty = (py - x[0]) / dx
    tz = (pz - x[0]) / dx
    
    i = min(max(int(np.floor(tx)), 0), N - 2)
    j = min(max(int(np.floor(ty)), 0), N - 2)
    k = min(max(int(np.floor(tz)), 0), N - 2)
    
    wx = tx - i
    wy = ty - j
    wz = tz - k
    
    c000 = field[i, j, k]
    c100 = field[i+1, j, k]
    c010 = field[i, j+1, k]
    c110 = field[i+1, j+1, k]
    c001 = field[i, j, k+1]
    c101 = field[i+1, j, k+1]
    c011 = field[i, j+1, k+1]
    c111 = field[i+1, j+1, k+1]
    
    c00 = c000 * (1.0 - wx) + c100 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    
    return c0 * (1.0 - wz) + c1 * wz

def trace_3d(start_pos, direction):
    base_step_size = 0.5 * dx
    step_size = base_step_size
    node_threshold = 1.8 * dx
    
    curr_pos = np.array(start_pos)
    path = [tuple(curr_pos)]
    
    curr_pos = curr_pos + step_size * np.array(direction)
    path.append(tuple(curr_pos))
    
    prev_dir = None
    max_steps = 300
    
    for _ in range(max_steps):
        # Check proximity to nodes
        for node in nodes:
            node_pos = np.array(node['pos'])
            if np.linalg.norm(curr_pos - node_pos) < node_threshold:
                path.append(tuple(node_pos))
                return node['id'], path
                
        gx = interpolate_3d(grad_x, curr_pos[0], curr_pos[1], curr_pos[2])
        gy = interpolate_3d(grad_y, curr_pos[0], curr_pos[1], curr_pos[2])
        gz = interpolate_3d(grad_z, curr_pos[0], curr_pos[1], curr_pos[2])
        
        g_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        if g_mag < 1e-8:
            break
            
        curr_dir = np.array([gx, gy, gz]) / g_mag
        
        if prev_dir is not None:
            if np.dot(curr_dir, prev_dir) < 0.0:
                step_size = max(step_size * 0.5, base_step_size * 0.01)
                
        curr_pos = curr_pos + step_size * curr_dir
        prev_dir = curr_dir
        
        if (curr_pos[0] < x[0] or curr_pos[0] > x[-1] or
            curr_pos[1] < x[0] or curr_pos[1] > x[-1] or
            curr_pos[2] < x[0] or curr_pos[2] > x[-1]):
            break
            
        path.append(tuple(curr_pos))
        
    for node in nodes:
        node_pos = np.array(node['pos'])
        if np.linalg.norm(curr_pos - node_pos) < node_threshold * 2.0:
            path.append(tuple(node_pos))
            return node['id'], path
            
    return None, path

print("Checking node pairs:")
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        node_a = nodes[i]
        node_b = nodes[j]
        dist = np.linalg.norm(np.array(node_a['pos']) - np.array(node_b['pos']))
        
        if 2.0 < dist < 3.0:
            midpoint = tuple(0.5 * (np.array(node_a['pos']) + np.array(node_b['pos'])))
            direction = np.array(node_a['pos']) - np.array(midpoint)
            dir_len = np.linalg.norm(direction)
            if dir_len > 1e-8:
                direction /= dir_len
                
            res_a, path_a = trace_3d(midpoint, direction)
            res_b, path_b = trace_3d(midpoint, -direction)
            print(f"Pair ({node_a['id']}, {node_b['id']}) at dist {dist:.4f}: midpoint={midpoint}")
            print(f"  res_a={res_a} (steps={len(path_a)}), res_b={res_b} (steps={len(path_b)})")
