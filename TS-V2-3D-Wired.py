import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for headless environments
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import json

def compile_3d_wired_geometry():
    # =========================================================================
    # LAYER 1: 3D FIELD SUBSTRATE (PDE EVOLUTION)
    # =========================================================================
    N = 50  # Voxel volume resolution
    steps = 150
    np.random.seed(42)
    Phi = np.random.randn(N, N, N) * 0.05

    # 3D Coordinate Grid
    x = np.linspace(-3, 3, N)
    dx = x[1] - x[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Wave sources arranged in an Octahedron (the dual of the cube)
    sources = [
        (2, 0, 0), (-2, 0, 0),
        (0, 2, 0), (0, -2, 0),
        (0, 0, 2), (0, 0, -2)
    ]

    # Superposition of decay-damped spherical wave fronts
    A = np.zeros((N, N, N))
    k = 2.0 * np.pi / 2.0  # Determining phase interference wavelength
    for sx, sy, sz in sources:
        r = np.sqrt((X - sx)**2 + (Y - sy)**2 + (Z - sz)**2)
        r = np.maximum(r, 1e-9)
        A += np.cos(k * r) / (1.0 + 0.5 * r**2)

    # 3D 7-point Laplacian Stencil
    def laplacian_3d(Z_field):
        return -6 * Z_field \
               + np.roll(Z_field, 1, axis=0) + np.roll(Z_field, -1, axis=0) \
               + np.roll(Z_field, 1, axis=1) + np.roll(Z_field, -1, axis=1) \
               + np.roll(Z_field, 1, axis=2) + np.roll(Z_field, -1, axis=2)

    dt = 0.02
    D = 0.1

    print("[TS-OS] Integrating 3D Allen-Cahn Wave Mechanics...")
    for _ in range(steps):
        lap = laplacian_3d(Phi)
        # Bistable double-well potential reaction with constructive wave interference
        Phi += dt * (D * lap + (Phi - Phi**3) + 1.0 * A)
        Phi = np.clip(Phi, -2.0, 2.0)

    # Smooth the phase field to suppress discretization artifacts
    Phi_s = nd.gaussian_filter(Phi, sigma=0.8)

    # =========================================================================
    # LAYER 2: 3D CURVATURE TENSOR (HESSIAN CALCULATOR)
    # =========================================================================
    print("[TS-OS] Computing 3D gradients and Hessian matrices...")
    grad_x = np.gradient(Phi_s, axis=0) / dx
    grad_y = np.gradient(Phi_s, axis=1) / dx
    grad_z = np.gradient(Phi_s, axis=2) / dx
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    phi_xx = np.gradient(grad_x, axis=0) / dx
    phi_xy = np.gradient(grad_x, axis=1) / dx
    phi_xz = np.gradient(grad_x, axis=2) / dx
    phi_yy = np.gradient(grad_y, axis=1) / dx
    phi_yz = np.gradient(grad_y, axis=2) / dx
    phi_zz = np.gradient(grad_z, axis=2) / dx

    print("[TS-OS] Performing 3D analytical Hessian eigensolution...")
    # Invariants for the characteristic cubic equation: lambda^3 - p*lambda^2 + q*lambda - r_det = 0
    p = phi_xx + phi_yy + phi_zz
    q = phi_xx * phi_yy + phi_yy * phi_zz + phi_xx * phi_zz - phi_xy**2 - phi_yz**2 - phi_xz**2
    r_det = (
        phi_xx * (phi_yy * phi_zz - phi_yz**2) -
        phi_xy * (phi_xy * phi_zz - phi_xz * phi_yz) +
        phi_xz * (phi_xy * phi_yz - phi_xz * phi_yy)
    )

    # Viete's analytical solution for real symmetric 3x3 matrices
    Q = (p**2 - 3.0 * q) / 9.0
    R = (9.0 * p * q - 2.0 * p**3 - 27.0 * r_det) / 54.0
    
    Q = np.maximum(Q, 0.0)
    sqrt_Q3 = np.sqrt(Q**3)
    denom = np.maximum(sqrt_Q3, 1e-15)
    arg = np.clip(R / denom, -1.0, 1.0)
    theta_ang = np.arccos(arg)

    # Compute three analytical eigenvalues
    l1 = p/3.0 + 2.0 * np.sqrt(Q) * np.cos(theta_ang / 3.0)
    l2 = p/3.0 + 2.0 * np.sqrt(Q) * np.cos((theta_ang + 2.0 * np.pi) / 3.0)
    l3 = p/3.0 + 2.0 * np.sqrt(Q) * np.cos((theta_ang + 4.0 * np.pi) / 3.0)

    # Sort eigenvalues at each voxel: lambda_1 <= lambda_2 <= lambda_3
    lambdas = np.stack([l1, l2, l3], axis=-1)
    lambdas = np.sort(lambdas, axis=-1)
    lambda_1 = lambdas[..., 0]
    lambda_2 = lambdas[..., 1]
    lambda_3 = lambdas[..., 2]

    # Print global curvature tensor diagnostics
    print(f"  [Diagnostics] Eigenvalues range:")
    print(f"    lambda_1 (min curvature): [{np.min(lambda_1):.2f}, {np.max(lambda_1):.2f}]")
    print(f"    lambda_2 (mid curvature): [{np.min(lambda_2):.2f}, {np.max(lambda_2):.2f}]")
    print(f"    lambda_3 (max curvature): [{np.min(lambda_3):.2f}, {np.max(lambda_3):.2f}]")

    # =========================================================================
    # LAYER 3: 3D TOPOLOGY COMPILER (NODE EXTRACTION & EDGE WALKER)
    # =========================================================================
    print("[TS-OS] Compiling Node vertices (Cube corners)...")
    local_max = nd.maximum_filter(Phi_s, size=3) == Phi_s
    interior_mask = (np.abs(X) < 1.8) & (np.abs(Y) < 1.8) & (np.abs(Z) < 1.8)
    nodes_mask = local_max & interior_mask
    node_coords = np.column_stack(np.where(nodes_mask))

    raw_nodes = []
    for idx in node_coords:
        coord = (float(x[idx[0]]), float(x[idx[1]]), float(x[idx[2]]))
        # Filter out the center node to keep the 8 cube corners
        if np.linalg.norm(coord) > 0.5:
            raw_nodes.append({
                'pos': coord,
                'grid_idx': (int(idx[0]), int(idx[1]), int(idx[2])),
                'intensity': float(Phi_s[idx[0], idx[1], idx[2]]),
                'curvature': (float(lambda_1[idx[0], idx[1], idx[2]]), 
                              float(lambda_2[idx[0], idx[1], idx[2]]), 
                              float(lambda_3[idx[0], idx[1], idx[2]]))
            })

    # Sort nodes by spatial coordinates to make Node IDs deterministic
    raw_nodes.sort(key=lambda n: (n['pos'][0], n['pos'][1], n['pos'][2]))
    nodes = []
    for nid, node in enumerate(raw_nodes):
        node['id'] = nid
        nodes.append(node)
    
    print(f"  [Topology] Successfully crystallized {len(nodes)} Node vertices.")

    # Trilinear Interpolator for gradient tracking
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

    # 3D Projected Gradient Ascent Edge Walker
    def trace_3d_projected(start_pos, direction, edge_dir):
        base_step_size = 0.5 * dx
        step_size = base_step_size
        node_threshold = 1.8 * dx
        
        curr_pos = np.array(start_pos)
        path = [tuple(curr_pos)]
        
        curr_pos = curr_pos + step_size * np.array(direction)
        path.append(tuple(curr_pos))
        
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
            
            # Project gradient onto the edge axis
            g_proj = gx * edge_dir[0] + gy * edge_dir[1] + gz * edge_dir[2]
            
            if abs(g_proj) < 1e-8:
                break
                
            # Move along the projected segment direction
            step_dir = np.sign(g_proj) * edge_dir
            curr_pos = curr_pos + step_size * step_dir
            
            if (curr_pos[0] < x[0] or curr_pos[0] > x[-1] or
                curr_pos[1] < x[0] or curr_pos[1] > x[-1] or
                curr_pos[2] < x[0] or curr_pos[2] > x[-1]):
                break
                
            path.append(tuple(curr_pos))
            
        # Fallback check at the end of maximum steps
        for node in nodes:
            node_pos = np.array(node['pos'])
            if np.linalg.norm(curr_pos - node_pos) < node_threshold * 2.0:
                path.append(tuple(node_pos))
                return node['id'], path
                
        return None, path

    print("[TS-OS] Tracing structural edges between adjacent nodes (Edge Walker)...")
    edges = []
    edge_id = 0
    seen_connections = set()
    
    # Cube edges connect adjacent vertices (separated by distance approx 2.57 units)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_a = nodes[i]
            node_b = nodes[j]
            dist = np.linalg.norm(np.array(node_a['pos']) - np.array(node_b['pos']))
            
            # Check if within the spatial distance range of cube edges [2.0, 3.0]
            if 2.0 < dist < 3.0:
                # Segment Midpoint (acting as the saddle point candidate)
                midpoint = 0.5 * (np.array(node_a['pos']) + np.array(node_b['pos']))
                
                # Unit direction vector of the segment
                edge_dir = np.array(node_a['pos']) - np.array(node_b['pos'])
                edge_dir /= np.linalg.norm(edge_dir)
                
                # Initialize tracers at 25% offset from midpoint towards each node
                start_a = midpoint + 0.25 * (np.array(node_a['pos']) - midpoint)
                start_b = midpoint + 0.25 * (np.array(node_b['pos']) - midpoint)
                
                # Perform projected gradient ascent in both directions
                res_a, path_a = trace_3d_projected(start_a, edge_dir, edge_dir)
                res_b, path_b = trace_3d_projected(start_b, -edge_dir, edge_dir)
                
                # Register edge if both paths successfully connect to distinct nodes
                if res_a is not None and res_b is not None and res_a != res_b:
                    conn_key = tuple(sorted([res_a, res_b]))
                    if conn_key not in seen_connections:
                        seen_connections.add(conn_key)
                        
                        # Merge the bidirectional paths
                        full_path = list(reversed(path_b)) + [tuple(midpoint)] + path_a
                        # Interpolate field values along the path to compute edge weight (resistance)
                        phi_values = [float(interpolate_3d(Phi_s, p[0], p[1], p[2])) for p in full_path]
                        min_phi = min(phi_values)
                        mean_phi = sum(phi_values) / len(phi_values)
                        # Resistance formula: R = 2.0 - min_phi (guarantees positive value since Phi in [-2,2])
                        weight = 2.0 - min_phi
                        
                        # Compute tension from field intensity at midpoint
                        tension = float(interpolate_3d(Phi_s, midpoint[0], midpoint[1], midpoint[2]))
                        
                        edges.append({
                            'id': edge_id,
                            'source': res_a,
                            'target': res_b,
                            'saddle_pos': tuple(midpoint),
                            'tension': tension,
                            'weight': weight,
                            'mean_phi': mean_phi,
                            'min_phi': min_phi,
                            'path': full_path
                        })
                        edge_id += 1

    print(f"  [Topology] Successfully compiled {len(edges)} connecting edges.")

    # =========================================================================
    # LAYER 4: JSON SERIALISATION
    # =========================================================================
    graph_data = {
        "graph_id": "TS_V2_3D_Wired",
        "metadata": {
            "dimensions": [N, N, N],
            "voxel_size": float(dx)
        },
        "nodes": [
            {
                "id": int(node['id']),
                "pos": [float(node['pos'][0]), float(node['pos'][1]), float(node['pos'][2])],
                "grid_idx": [int(node['grid_idx'][0]), int(node['grid_idx'][1]), int(node['grid_idx'][2])],
                "intensity": float(node['intensity']),
                "curvature": [float(node['curvature'][0]), float(node['curvature'][1]), float(node['curvature'][2])]
            }
            for node in nodes
        ],
        "edges": [
            {
                "id": int(edge['id']),
                "source": int(edge['source']),
                "target": int(edge['target']),
                "saddle_pos": [float(edge['saddle_pos'][0]), float(edge['saddle_pos'][1]), float(edge['saddle_pos'][2])],
                "tension": float(edge['tension']),
                "weight": float(edge['weight']),
                "path": [[float(p[0]), float(p[1]), float(p[2])] for p in edge['path']]
            }
            for edge in edges
        ]
    }

    output_json = "ts_v2_3d_graph.json"
    with open(output_json, "w") as f:
        json.dump(graph_data, f, indent=2)
    print(f"[TS-OS] Serialized 3D Wired Graph to {output_json}")

    # =========================================================================
    # 3D VISUALISATION
    # =========================================================================
    print("[TS-OS] Plotting 3D topological wired layout...")
    fig = plt.figure(figsize=(12, 12), facecolor='#111111')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#111111')

    # Plot wave sources (Octahedron vertices)
    sx, sy, sz = zip(*sources)
    ax.scatter(sx, sy, sz, c='#00FFCC', s=200, label='Wave Sources (Octahedron)', alpha=0.3, depthshade=False)

    # Plot emergent nodes (Cube vertices)
    nx = [node['pos'][0] for node in nodes]
    ny = [node['pos'][1] for node in nodes]
    nz = [node['pos'][2] for node in nodes]
    ax.scatter(nx, ny, nz, c='#FF0055', s=180, edgecolors='white', linewidths=1.5, label='Emergent Nodes (Cube)', depthshade=False, zorder=10)

    # Draw the 12 emergent edges as white lines representing the continuous paths
    for edge in edges:
        epath = np.array(edge['path'])
        px, py, pz = epath[:, 0], epath[:, 1], epath[:, 2]
        ax.plot(px, py, pz, color='white', linewidth=2.5, alpha=0.9, zorder=5)

    ax.set_title("TS-V2 3D: Autonomously Wired Cube Dual from Octahedral Waves", color='white', fontsize=15, pad=10)
    ax.legend(facecolor='#222222', edgecolor='white', labelcolor='white', loc='upper right')

    # Clean axes for elegant dark mode theme
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(color='#333333')
    ax.tick_params(colors='white')
    
    # Label axes
    ax.set_xlabel('X Dimension', color='white')
    ax.set_ylabel('Y Dimension', color='white')
    ax.set_zlabel('Z Dimension', color='white')

    output_plot = "ts_v2_3d_plot.png"
    plt.savefig(output_plot, dpi=150, bbox_inches='tight', facecolor='#111111')
    print(f"[TS-OS] Saved 3D topological visualization to {output_plot}")
    plt.close()

if __name__ == '__main__':
    compile_3d_wired_geometry()
