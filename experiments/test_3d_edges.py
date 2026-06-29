import numpy as np
import scipy.ndimage as nd

def run_test():
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

    print("Evolving 3D field...")
    for _ in range(steps):
        lap = laplacian_3d(Phi)
        Phi += dt * (D * lap + (Phi - Phi**3) + 1.0 * A)
        Phi = np.clip(Phi, -2.0, 2.0)

    Phi_s = nd.gaussian_filter(Phi, sigma=0.8)

    # Compute 3D Gradients
    print("Computing gradients...")
    grad_x = np.gradient(Phi_s, axis=0) / dx
    grad_y = np.gradient(Phi_s, axis=1) / dx
    grad_z = np.gradient(Phi_s, axis=2) / dx

    # Extract Nodes (Cube Vertices)
    print("Extracting Nodes...")
    local_max = nd.maximum_filter(Phi_s, size=3) == Phi_s
    interior_mask = (np.abs(X) < 1.8) & (np.abs(Y) < 1.8) & (np.abs(Z) < 1.8)
    nodes_mask = local_max & interior_mask
    node_coords = np.column_stack(np.where(nodes_mask))

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

    print(f"Nodes found: {len(nodes)}")

    # Trilinear Interpolator
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
            
            # Project gradient onto the edge direction vector
            g_proj = gx * edge_dir[0] + gy * edge_dir[1] + gz * edge_dir[2]
            
            if abs(g_proj) < 1e-8:
                break
                
            # Step along the projected direction
            step_dir = np.sign(g_proj) * edge_dir
            curr_pos = curr_pos + step_size * step_dir
            
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

    # Find edges using midpoints of node pairs
    print("Tracing edges between adjacent nodes...")
    edges = []
    edge_id = 0
    seen_connections = set()
    
    # Adjacent node distance is approx 2.57 units. We search in [2.0, 3.0]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_a = nodes[i]
            node_b = nodes[j]
            dist = np.linalg.norm(np.array(node_a['pos']) - np.array(node_b['pos']))
            
            if 2.0 < dist < 3.0:
                # Midpoint is the candidate saddle
                midpoint = 0.5 * (np.array(node_a['pos']) + np.array(node_b['pos']))
                
                # Direction vector along the segment connecting them
                edge_dir = np.array(node_a['pos']) - np.array(node_b['pos'])
                edge_dir /= np.linalg.norm(edge_dir)
                
                # Perturb 25% of the half-length towards node_a and node_b
                start_a = midpoint + 0.25 * (np.array(node_a['pos']) - midpoint)
                start_b = midpoint + 0.25 * (np.array(node_b['pos']) - midpoint)
                
                # Trace
                res_a, path_a = trace_3d_projected(start_a, edge_dir, edge_dir)
                res_b, path_b = trace_3d_projected(start_b, -edge_dir, edge_dir)
                
                if res_a is not None and res_b is not None and res_a != res_b:
                    conn_key = tuple(sorted([res_a, res_b]))
                    if conn_key not in seen_connections:
                        seen_connections.add(conn_key)
                        full_path = list(reversed(path_b)) + [tuple(midpoint)] + path_a
                        edges.append({
                            'id': edge_id,
                            'source': res_a,
                            'target': res_b,
                            'saddle_pos': tuple(midpoint),
                            'path': full_path
                        })
                        edge_id += 1

    print(f"Edges compiled: {len(edges)}")
    for e in edges:
        print(f"  Edge {e['id']}: Node {e['source']} <--> Node {e['target']}")

if __name__ == '__main__':
    run_test()
