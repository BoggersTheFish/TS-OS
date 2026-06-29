import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from core.substrate import FieldSubstrate
from core.tensors import CurvatureSensor

class TopologyCompiler:
    """
    Layer 3: Topology Compiler
    Extracts the discrete topological graph (Nodes, Edges) from continuous fields.
    Traces saddle manifolds using Morse theory via vectorized gradient ascent.
    """
    def __init__(self, substrate: FieldSubstrate, curvature_sensor: CurvatureSensor):
        self.substrate = substrate
        self.sensor = curvature_sensor
        
    def _interpolate(self, field: np.ndarray, px: float, py: float) -> float:
        """
        Performs bilinear interpolation on a 2D field at continuous coordinate (px, py).
        """
        # Clamp to grid boundaries
        px = np.clip(px, self.substrate.x[0], self.substrate.x[-1])
        py = np.clip(py, self.substrate.y[0], self.substrate.y[-1])
        
        # Calculate fractional indices
        tx = (px - self.substrate.x[0]) / self.substrate.dx
        ty = (py - self.substrate.y[0]) / self.substrate.dy
        
        i = int(np.floor(tx))
        j = int(np.floor(ty))
        
        # Clamp indices for interpolation cell
        i = min(max(i, 0), self.substrate.resolution_x - 2)
        j = min(max(j, 0), self.substrate.resolution_y - 2)
        
        wx = tx - i
        wy = ty - j
        
        # Bilinear interpolation
        val = (
            (1.0 - wx) * (1.0 - wy) * field[i, j] +
            wx * (1.0 - wy) * field[i+1, j] +
            (1.0 - wx) * wy * field[i, j+1] +
            wx * wy * field[i+1, j+1]
        )
        return val

    def extract_nodes(
        self, 
        phi: np.ndarray, 
        diagnostics: Dict[str, np.ndarray],
        min_distance: float = 0.5,
        intensity_percentile: float = 30.0
    ) -> List[Dict[str, any]]:
        """
        Extracts local maxima of the field phi as Nodes.
        Filters them using non-maximum suppression.
        """
        # A pixel is a local maximum if it is greater than all 8 neighbors
        phi_padded = np.pad(phi, 1, mode='edge')
        
        is_max = (
            (phi >= phi_padded[:-2, 1:-1]) & (phi >= phi_padded[2:, 1:-1]) &
            (phi >= phi_padded[1:-1, :-2]) & (phi >= phi_padded[1:-1, 2:]) &
            (phi >= phi_padded[:-2, :-2]) & (phi >= phi_padded[:-2, 2:]) &
            (phi >= phi_padded[2:, :-2]) & (phi >= phi_padded[2:, 2:])
        )
        
        # Filter by intensity threshold to avoid noise in flat regions
        threshold = np.percentile(phi, intensity_percentile)
        is_max = is_max & (phi >= threshold)
        
        # Find indices
        indices_x, indices_y = np.where(is_max)
        
        # Gather node candidates
        candidates = []
        for idx in range(len(indices_x)):
            ix, iy = indices_x[idx], indices_y[idx]
            px = self.substrate.x[ix]
            py = self.substrate.y[iy]
            val = phi[ix, iy]
            l1 = diagnostics['lambda_1'][ix, iy]
            l2 = diagnostics['lambda_2'][ix, iy]
            candidates.append({
                'pos': (px, py),
                'grid_idx': (ix, iy),
                'intensity': float(val),
                'curvature': (float(l1), float(l2))
            })
            
        # Sort candidates by intensity (descending)
        candidates.sort(key=lambda item: item['intensity'], reverse=True)
        
        # Non-maximum suppression based on Euclidean distance
        nodes = []
        node_id = 0
        for cand in candidates:
            too_close = False
            for node in nodes:
                dist = np.sqrt((cand['pos'][0] - node['pos'][0])**2 + 
                               (cand['pos'][1] - node['pos'][1])**2)
                if dist < min_distance:
                    too_close = True
                    break
            if not too_close:
                cand['id'] = node_id
                nodes.append(cand)
                node_id += 1
                
        return nodes

    def extract_saddles(
        self, 
        diagnostics: Dict[str, np.ndarray],
        grad_threshold_percentile: float = 25.0,
        min_distance: float = 0.5
    ) -> List[Dict[str, any]]:
        """
        Extracts saddle points (local minima of gradient magnitude where lambda_1 < 0 < lambda_2).
        Filters them using spatial suppression.
        """
        grad_mag = diagnostics['grad_mag']
        lambda_1 = diagnostics['lambda_1']
        lambda_2 = diagnostics['lambda_2']
        
        # A pixel is a local minimum of gradient magnitude if it's smaller than its 4 cardinal neighbors
        grad_padded = np.pad(grad_mag, 1, mode='edge')
        is_min_grad = (
            (grad_mag <= grad_padded[:-2, 1:-1]) & (grad_mag <= grad_padded[2:, 1:-1]) &
            (grad_mag <= grad_padded[1:-1, :-2]) & (grad_mag <= grad_padded[1:-1, 2:])
        )
        
        # Filter for saddle signature: lambda_1 < 0 and lambda_2 > 0
        # And gradient magnitude is relatively small (below threshold percentile)
        grad_limit = np.percentile(grad_mag, grad_threshold_percentile)
        is_saddle = is_min_grad & (lambda_1 < 0.0) & (lambda_2 > 0.0) & (grad_mag < grad_limit)
        
        indices_x, indices_y = np.where(is_saddle)
        
        candidates = []
        for idx in range(len(indices_x)):
            ix, iy = indices_x[idx], indices_y[idx]
            px = self.substrate.x[ix]
            py = self.substrate.y[iy]
            g_mag = grad_mag[ix, iy]
            l1 = lambda_1[ix, iy]
            l2 = lambda_2[ix, iy]
            v2 = (diagnostics['v2_x'][ix, iy], diagnostics['v2_y'][ix, iy])
            candidates.append({
                'pos': (px, py),
                'grid_idx': (ix, iy),
                'grad_mag': float(g_mag),
                'curvature': (float(l1), float(l2)),
                'v2': v2
            })
            
        # Sort candidates by gradient magnitude (ascending - smaller gradient is closer to mathematical saddle)
        candidates.sort(key=lambda item: item['grad_mag'])
        
        saddles = []
        saddle_id = 0
        for cand in candidates:
            too_close = False
            for saddle in saddles:
                dist = np.sqrt((cand['pos'][0] - saddle['pos'][0])**2 + 
                               (cand['pos'][1] - saddle['pos'][1])**2)
                if dist < min_distance:
                    too_close = True
                    break
            if not too_close:
                cand['id'] = saddle_id
                saddles.append(cand)
                saddle_id += 1
                
        return saddles

    def trace_manifold(
        self, 
        start_pos: Tuple[float, float], 
        direction: Tuple[float, float],
        phi: np.ndarray,
        diagnostics: Dict[str, np.ndarray],
        nodes: List[Dict[str, any]],
        step_multiplier: float = 0.5,
        max_steps: int = 400
    ) -> Tuple[Optional[int], List[Tuple[float, float]]]:
        """
        Traces a single manifold path using gradient ascent from a start position.
        Returns the ID of the reached Node (or None if failed) and the coordinates of the path.
        """
        grad_x = diagnostics['grad_x']
        grad_y = diagnostics['grad_y']
        
        # Step size is half the minimum grid spacing
        base_step_size = step_multiplier * min(self.substrate.dx, self.substrate.dy)
        step_size = base_step_size
        
        # Target node threshold for connection
        node_threshold = 1.5 * min(self.substrate.dx, self.substrate.dy)
        
        curr_pos = np.array(start_pos)
        path = [tuple(curr_pos)]
        
        # Initial perturbation along direction
        curr_pos = curr_pos + step_size * np.array(direction)
        path.append(tuple(curr_pos))
        
        prev_dir = None
        
        for _ in range(max_steps):
            # Check if we are near any node
            for node in nodes:
                node_pos = np.array(node['pos'])
                if np.linalg.norm(curr_pos - node_pos) < node_threshold:
                    path.append(tuple(node_pos))
                    return node['id'], path
            
            # Interpolate gradient at current position
            gx = self._interpolate(grad_x, curr_pos[0], curr_pos[1])
            gy = self._interpolate(grad_y, curr_pos[0], curr_pos[1])
            
            g_mag = np.sqrt(gx**2 + gy**2)
            if g_mag < 1e-8:
                break  # Reached a local critical point, stop
                
            # Gradient ascent step (normalized step direction)
            curr_dir = np.array([gx, gy]) / g_mag
            
            # Step damping: if we are overshooting (direction flips), shrink step size
            if prev_dir is not None:
                dot_prod = np.dot(curr_dir, prev_dir)
                if dot_prod < 0.0:
                    step_size = max(step_size * 0.5, base_step_size * 0.01)
            
            curr_pos = curr_pos + step_size * curr_dir
            prev_dir = curr_dir
            
            # Check if we are out of grid bounds
            if (curr_pos[0] < self.substrate.x[0] or curr_pos[0] > self.substrate.x[-1] or
                curr_pos[1] < self.substrate.y[0] or curr_pos[1] > self.substrate.y[-1]):
                break
                
            path.append(tuple(curr_pos))
            
        # If we finished but are close to a node, connect it
        for node in nodes:
            node_pos = np.array(node['pos'])
            if np.linalg.norm(curr_pos - node_pos) < node_threshold * 2.0:
                path.append(tuple(node_pos))
                return node['id'], path
                
        return None, path

    def compile_graph(
        self, 
        phi: np.ndarray, 
        diagnostics: Dict[str, np.ndarray],
        node_min_dist: float = 0.5,
        saddle_min_dist: float = 0.5,
        intensity_percentile: float = 30.0
    ) -> Dict[str, any]:
        """
        Runs the complete topological extraction pipeline.
        Returns a dictionary representing the extracted graph.
        """
        # Step 1: Extract Nodes
        nodes = self.extract_nodes(
            phi, 
            diagnostics, 
            min_distance=node_min_dist, 
            intensity_percentile=intensity_percentile
        )
        
        # Step 2: Extract Saddles
        saddles = self.extract_saddles(
            diagnostics, 
            grad_threshold_percentile=35.0, 
            min_distance=saddle_min_dist
        )
        
        # Step 3: Trace Edges from Saddles
        edges = []
        edge_id = 0
        seen_connections = set()
        
        for saddle in saddles:
            saddle_pos = saddle['pos']
            v2 = np.array(saddle['v2'])  # Eigenvector associated with lambda_2 (direction of ridge)
            
            # Trace in positive direction (+v2)
            node_a, path_a = self.trace_manifold(saddle_pos, v2, phi, diagnostics, nodes)
            
            # Trace in negative direction (-v2)
            node_b, path_b = self.trace_manifold(saddle_pos, -v2, phi, diagnostics, nodes)
            
            # If both paths successfully reach nodes, and they are distinct nodes
            if node_a is not None and node_b is not None and node_a != node_b:
                # Order connection key to avoid duplicates
                conn_key = tuple(sorted([node_a, node_b]))
                if conn_key not in seen_connections:
                    seen_connections.add(conn_key)
                    
                    # Combine paths: reverse path_a, append saddle_pos, append path_b
                    # This forms a single continuous spline tracing the ridge through the saddle
                    full_path = list(reversed(path_a)) + [saddle_pos] + path_b
                    
                    # Calculate tension (average field intensity along the path)
                    intensities = [self._interpolate(phi, p[0], p[1]) for p in full_path]
                    avg_tension = float(np.mean(intensities))
                    
                    edges.append({
                        'id': edge_id,
                        'source': node_a,
                        'target': node_b,
                        'saddle_pos': saddle_pos,
                        'tension': avg_tension,
                        'path': full_path
                    })
                    edge_id += 1
                    
        return {
            'nodes': nodes,
            'edges': edges,
            'saddles': saddles
        }
