import os
import sys
import numpy as np

# Add the project root to python path to import core and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.substrate import FieldSubstrate
from core.tensors import CurvatureSensor
from core.compiler import TopologyCompiler
from utils.graph_exporter import GraphExporter
from utils.visualizer import TSVisualizer

def run_experiment():
    print("==================================================")
    print("TS-VERSE ENGINE: Topological Compilation Experiment")
    print("==================================================")
    
    # 1. Initialize Field Substrate (Layer 1)
    # 10x10 units with 200x200 grid resolution
    width, height = 10.0, 10.0
    res_x, res_y = 200, 200
    substrate = FieldSubstrate(width=width, height=height, resolution_x=res_x, resolution_y=res_y)
    
    # Place four isotropic wave sources to construct an interference grid
    # Sources placed symmetrically to form a nice phase-boundary skeleton
    substrate.add_source(2.5, 2.5, amplitude=1.2, k=2.0 * np.pi / 3.0, phase=0.0)
    substrate.add_source(7.5, 2.5, amplitude=1.2, k=2.0 * np.pi / 3.0, phase=0.0)
    substrate.add_source(2.5, 7.5, amplitude=1.2, k=2.0 * np.pi / 3.0, phase=0.0)
    substrate.add_source(7.5, 7.5, amplitude=1.2, k=2.0 * np.pi / 3.0, phase=0.0)
    
    # Superpose sources to calculate raw wave field (constructive & destructive interference)
    print("\n[Layer 1] Computing isotropic wave interference...")
    substrate.compute_wave_interference()
    
    # Evolve under Allen-Cahn double-well PDE to sharpen boundaries and stabilize ridges
    print("[Layer 1] Evolving field via Allen-Cahn PDE (stabilizing phase-boundaries)...")
    substrate.evolve_allen_cahn(dt=0.01, D=0.05, iterations=40, beta=1.0)
    
    # Normalize field to [0, 1] range
    phi = substrate.normalize_field()
    
    # 2. Compute Curvature Tensors (Layer 2)
    print("\n[Layer 2] Analyzing field curvature and gradients via Hessian eigenvalues...")
    sensor = CurvatureSensor(dx=substrate.dx, dy=substrate.dy)
    # Apply a light smoothing filter (sigma=1.0) to stabilize spatial derivatives
    diagnostics = sensor.analyze_field(phi, sigma=1.0)
    
    # 3. Extract Topology (Layer 3)
    print("\n[Layer 3] Extracting critical points and tracing saddle manifolds...")
    compiler = TopologyCompiler(substrate, sensor)
    graph = compiler.compile_graph(
        phi, 
        diagnostics, 
        node_min_dist=0.8, 
        saddle_min_dist=0.8,
        intensity_percentile=35.0
    )
    
    # 4. Serialize and Export Graph (Layer 4)
    print("\n[Layer 4] Collapsing physics into discrete logical graph JSON...")
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    json_path = os.path.join(output_dir, 'constraint_graph.json')
    GraphExporter.export_to_json(graph, json_path)
    
    # 5. Visualizer
    print("\n[Visualizer] Generating Matplotlib debugging overlay plots...")
    plot_path = os.path.join(output_dir, 'topology_plot.png')
    visualizer = TSVisualizer(width=width, height=height)
    visualizer.plot_field_and_topology(
        phi, 
        graph, 
        diagnostics=diagnostics, 
        save_path=plot_path, 
        show=False
    )
    print("==================================================")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == '__main__':
    run_experiment()
