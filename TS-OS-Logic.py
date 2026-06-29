import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_propagation():
    # 1. Ingest graph data
    input_json = "ts_v2_3d_graph.json"
    print(f"[TS-OS-Logic] Loading graph structure from {input_json}...")
    with open(input_json, "r") as f:
        graph = json.load(f)

    nodes = graph['nodes']
    edges = graph['edges']
    num_nodes = len(nodes)
    
    # 2. Build Adjacency/Resistance structure
    # We represent the graph connectivity as a dictionary of neighbors and their weights (resistances)
    neighbors = {node['id']: [] for node in nodes}
    for edge in edges:
        u = edge['source']
        v = edge['target']
        w = edge['weight']
        neighbors[u].append((v, w))
        neighbors[v].append((u, w))

    # Print loaded structure details
    print(f"[TS-OS-Logic] Graph loaded: {num_nodes} nodes, {len(edges)} edges.")
    print("[TS-OS-Logic] Edge Resistances:")
    for edge in edges:
        print(f"  Edge {edge['id']} (Node {edge['source']} <--> Node {edge['target']}): R = {edge['weight']:.4f}")

    # 3. Initialize states
    E = np.zeros(num_nodes)
    E[0] = 1.0  # Truth Pulse at Node 0
    
    print(f"\n[TS-OS-Logic] Initial Energy Distribution:")
    for i, val in enumerate(E):
        print(f"  Node {i}: {val:.4f}")

    # Calculate stable time step (CFL condition)
    # dt < 1 / sum(1/R_ij) for all nodes i
    min_limit = float('inf')
    for u in range(num_nodes):
        sum_inv_r = sum(1.0 / w for v, w in neighbors[u])
        limit = 1.0 / sum_inv_r
        if limit < min_limit:
            min_limit = limit

    dt = 0.9 * min_limit  # Safe time step
    print(f"[TS-OS-Logic] Calculated CFL limit: {min_limit:.5f} s. Using dt = {dt:.5f} s.")

    # 4. State Propagation Loop
    max_steps = 2000
    tolerance = 1e-7
    history = [E.copy()]
    steps_executed = 0
    
    print("\n[TS-OS-Logic] Simulating state propagation...")
    for step in range(1, max_steps + 1):
        E_next = E.copy()
        
        # Calculate update for each node
        for u in range(num_nodes):
            flow = 0.0
            for v, w in neighbors[u]:
                # Flow = (E_neighbor - E_current) / Resistance
                flow += (E[v] - E[u]) / w
            E_next[u] += dt * flow
            
        # Check convergence (max change)
        max_diff = np.max(np.abs(E_next - E))
        E = E_next
        history.append(E.copy())
        steps_executed = step
        
        # Print intermediate states
        if step in [1, 10, 50, 100, 200, 500] or max_diff < tolerance:
            print(f"  Step {step:4d} | Sum Energy: {np.sum(E):.6f} | Max Delta: {max_diff:.2e}")
            state_str = " | ".join(f"N{i}:{E[i]:.4f}" for i in range(num_nodes))
            print(f"    States: {state_str}")
            
        if max_diff < tolerance:
            print(f"[TS-OS-Logic] Convergence reached at step {step}!")
            break

    # 5. Output results
    print(f"\n[TS-OS-Logic] Final Equilibrium State (Step {steps_executed}):")
    for i, val in enumerate(E):
        print(f"  Node {i} (coords: {nodes[i]['pos']}): Energy = {val:.6f}")
    print(f"  Total Network Energy: {np.sum(E):.6f} (Conserved: {np.isclose(np.sum(E), 1.0)})")

    # 6. Plot energy evolution profile
    history = np.array(history)
    plt.figure(figsize=(10, 6), facecolor='#111111')
    ax = plt.axes()
    ax.set_facecolor('#111111')
    
    colors = ['#FF0055', '#FF7700', '#FFCC00', '#00FFCC', '#00CCFF', '#9900FF', '#FF00CC', '#FFFFFF']
    for i in range(num_nodes):
        coords_str = f"({nodes[i]['pos'][0]:.1f},{nodes[i]['pos'][1]:.1f},{nodes[i]['pos'][2]:.1f})"
        plt.plot(history[:, i], label=f"Node {i} {coords_str}", color=colors[i % len(colors)], linewidth=2)
        
    plt.title("TS-OS Logic: State Propagation & Convergence (Truth Pulse)", color='white', fontsize=14, pad=15)
    plt.xlabel("Simulation Steps", color='white')
    plt.ylabel("Energy Level", color='white')
    plt.grid(color='#333333', linestyle='--')
    ax.tick_params(colors='white')
    plt.legend(facecolor='#222222', edgecolor='white', labelcolor='white')
    
    output_plot = "ts_os_logic_plot.png"
    plt.savefig(output_plot, dpi=150, bbox_inches='tight', facecolor='#111111')
    print(f"[TS-OS-Logic] Saved energy propagation plot to {output_plot}")
    plt.close()

if __name__ == '__main__':
    run_propagation()
