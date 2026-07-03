import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Scale factor for 100% integer fixed-point math
SCALE = 100_000_000

# 16-Opcode Mapping
# Triggered based on binary combination of 3 neighbor states:
# b_3 (node ID is odd) * 8 + b_2 * 4 + b_1 * 2 + b_0 * 1
OPCODES = {
    0: "LOAD",  # Node 0 (step 0)
    1: "NOP",
    2: "NOP",
    3: "ADD",   # Node 2 (step 30)
    4: "NOP",
    5: "NOP",
    6: "NOP",
    7: "HALT",  # Node 6 (step 60)
    8: "NOP",
    9: "LOAD",  # Node 1 (step 15)
    10: "NOP",
    11: "NOP",
    12: "NOP",
    13: "NOP",
    14: "NOP",
    15: "PRINT" # Node 3 (step 45)
}

def run_bogvm():
    input_json = "ts_v2_3d_graph.json"
    print(f"[BOGVM-0] Loading topological substrate from {input_json}...")
    with open(input_json, "r") as f:
        graph = json.load(f)

    nodes = graph['nodes']
    edges = graph['edges']
    num_nodes = len(nodes)

    # 1. Build fixed-point Adjacency/Resistance structure
    neighbors = {node['id']: [] for node in nodes}
    for edge in edges:
        u = edge['source']
        v = edge['target']
        # Convert weight (float) to integer fixed-point
        r_val = int(round(edge['weight'] * SCALE))
        neighbors[u].append((v, r_val))
        neighbors[v].append((u, r_val))

    # Sort neighbor entries to ensure absolute order determinism
    for u in neighbors:
        neighbors[u].sort(key=lambda item: item[0])

    # 2. Initialize BOGVM-0 Registers & Memory
    R = [0, 0, 0, 0]  # R0, R1, R2, R3
    memory = [0] * 16  # Memory cells
    halted = False

    # 3. Wave-State Initialization
    E = np.zeros(num_nodes, dtype=np.int64)
    prev_E = np.zeros(num_nodes, dtype=np.int64)
    refractory = np.zeros(num_nodes, dtype=np.int64)

    # Select small safe dt to enable gradual propagation and rising-edge triggering
    dt = int(0.05 * SCALE)
    print(f"[BOGVM-0] Timestep dt = {dt / SCALE:.5f} ({dt} scaled)")

    # Program Pulse Queue (time_step, node_id, energy_float)
    # We inject structured pulses to trigger specific operations:
    pulses = [
        (0, 0, 1.4),   # Step 0: Inject 1.4 E to Node 0 -> LOAD R0, 14
        (15, 1, 0.9),  # Step 15: Inject 0.9 E to Node 1 -> LOAD R1, 9
        (30, 2, 0.5),  # Step 30: Inject 0.5 E to Node 2 -> ADD R2 = R0 + R1
        (45, 3, 0.6),  # Step 45: Inject 0.6 E to Node 3 -> PRINT R
        (60, 6, 1.0)   # Step 60: Inject 1.0 E to Node 6 -> HALT
    ]

    history = []
    
    print("\n[BOGVM-0] Starting Execution Loop...")
    for step in range(150):
        if halted:
            print(f"[BOGVM-0] System halted at step {step}.")
            break

        prev_E = E.copy()

        # Handle Pulse Injections
        for p_step, p_node, p_val in pulses:
            if step == p_step:
                val_scaled = int(round(p_val * SCALE))
                E[p_node] += val_scaled
                print(f"[BOGVM-0] Pulse Injected: Node {p_node} +{p_val:.2f} E -> New Energy: {E[p_node]/SCALE:.4f}")

        # Continuous diffusion update (fixed-point integer division)
        E_next = E.copy()
        for u in range(num_nodes):
            flow = 0
            for v, r_val in neighbors[u]:
                # In integer math: flow = (E[v] - E[u]) * dt // r_val
                diff = E[v] - E[u]
                flow += (diff * dt) // r_val
            E_next[u] += flow
        
        E = E_next
        history.append(E.copy())

        # Update refractory period counters
        for u in range(num_nodes):
            if refractory[u] > 0:
                refractory[u] -= 1

        # Check execution triggers (rising-edge crossing 0.50 threshold)
        trigger_threshold = int(0.50 * SCALE)
        neighbor_active_threshold = int(0.15 * SCALE)

        for u in range(num_nodes):
            if prev_E[u] < trigger_threshold and E[u] >= trigger_threshold:
                if refractory[u] == 0:
                    # Trigger Opcode Execution!
                    refractory[u] = 12  # Refractory period of 12 steps

                    # Compute 4-bit Opcode based on 3 neighbors + parity
                    # Sort neighbors by ID to be deterministic
                    u_neighbors = neighbors[u]
                    b0 = 1 if E[u_neighbors[0][0]] >= neighbor_active_threshold else 0
                    b1 = 1 if E[u_neighbors[1][0]] >= neighbor_active_threshold else 0
                    b2 = 1 if E[u_neighbors[2][0]] >= neighbor_active_threshold else 0
                    b3 = 1 if u % 2 != 0 else 0

                    opcode_val = b3 * 8 + b2 * 4 + b1 * 2 + b0 * 1
                    opcode_name = OPCODES[opcode_val]

                    # Execute Opcode Instruction
                    target_reg = u % 4
                    val_loaded = int(E[u] // (10_000_000)) # e.g. 1.4 -> 14

                    print(f"--- TRIGGER step {step:3d} | Node {u} (E={E[u]/SCALE:.4f}) ---")
                    print(f"    Neighbor States: Node {u_neighbors[0][0]}={b0}, Node {u_neighbors[1][0]}={b1}, Node {u_neighbors[2][0]}={b2} | Parity={b3}")
                    print(f"    Instruction decoded: 0x{opcode_val:02X} -> {opcode_name}")

                    if opcode_name == "LOAD":
                        R[target_reg] = val_loaded
                        print(f"    EXEC: LOAD value {val_loaded} into R{target_reg} | Registers: R={R}")
                    elif opcode_name == "ADD":
                        R[2] = R[0] + R[1]
                        print(f"    EXEC: ADD R0 ({R[0]}) + R1 ({R[1]}) -> R2 = {R[2]} | Registers: R={R}")
                    elif opcode_name == "SUB":
                        R[2] = R[0] - R[1]
                        print(f"    EXEC: SUB R0 ({R[0]}) - R1 ({R[1]}) -> R2 = {R[2]} | Registers: R={R}")
                    elif opcode_name == "MUL":
                        R[2] = R[0] * R[1]
                        print(f"    EXEC: MUL R0 ({R[0]}) * R1 ({R[1]}) -> R2 = {R[2]} | Registers: R={R}")
                    elif opcode_name == "DIV":
                        denom = R[1] if R[1] != 0 else 1
                        R[2] = R[0] // denom
                        print(f"    EXEC: DIV R0 ({R[0]}) / R1 ({R[1]}) -> R2 = {R[2]} | Registers: R={R}")
                    elif opcode_name == "PRINT":
                        print(f"    EXEC: PRINT -> Registers: R={R} | Memory: {memory}")
                    elif opcode_name == "STORE":
                        mem_addr = u % 16
                        memory[mem_addr] = R[2]
                        print(f"    EXEC: STORE R2 ({R[2]}) into Memory[{mem_addr}] | Memory: {memory}")
                    elif opcode_name == "HALT":
                        print(f"    EXEC: HALT virtual machine.")
                        halted = True
                    elif opcode_name == "NOP":
                        print(f"    EXEC: NOP (No Operation)")

    # 4. Save Execution Visual
    history = np.array(history) / SCALE
    plt.figure(figsize=(10, 6), facecolor='#111111')
    ax = plt.axes()
    ax.set_facecolor('#111111')
    colors = ['#FF0055', '#FF7700', '#FFCC00', '#00FFCC', '#00CCFF', '#9900FF', '#FF00CC', '#FFFFFF']
    for i in range(num_nodes):
        plt.plot(history[:, i], label=f"Node {i}", color=colors[i % len(colors)], linewidth=2)
    plt.axhline(0.50, color='red', linestyle='--', alpha=0.5, label='Trigger Threshold (0.50)')
    plt.title("BOGVM-0: Wave-State VM Energy Propagation", color='white', fontsize=14, pad=15)
    plt.xlabel("Simulation Steps", color='white')
    plt.ylabel("Energy Level", color='white')
    plt.grid(color='#333333', linestyle='--')
    ax.tick_params(colors='white')
    plt.legend(facecolor='#222222', edgecolor='white', labelcolor='white')
    
    output_plot = "bogvm_exec_plot.png"
    plt.savefig(output_plot, dpi=150, bbox_inches='tight', facecolor='#111111')
    print(f"[BOGVM-0] Saved execution plot to {output_plot}")
    plt.close()

if __name__ == '__main__':
    run_bogvm()
