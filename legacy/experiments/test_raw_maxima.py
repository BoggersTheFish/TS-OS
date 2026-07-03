import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.substrate import FieldSubstrate
from core.tensors import CurvatureSensor
from core.compiler import TopologyCompiler

def test():
    width, height = 10.0, 10.0
    res_x, res_y = 200, 200
    substrate = FieldSubstrate(width=width, height=height, resolution_x=res_x, resolution_y=res_y)
    
    substrate.add_source(2.5, 2.5, amplitude=1.2, k=2.0 * np.pi / 3.0, phase=0.0)
    substrate.add_source(7.5, 2.5, amplitude=1.2, k=2.0 * np.pi / 3.0, phase=0.0)
    substrate.add_source(2.5, 7.5, amplitude=1.2, k=2.0 * np.pi / 3.0, phase=0.0)
    substrate.add_source(7.5, 7.5, amplitude=1.2, k=2.0 * np.pi / 3.0, phase=0.0)
    
    substrate.compute_wave_interference()
    phi_raw = substrate.phi.copy()
    
    # Normalize raw field to check maxima
    phi_min = np.min(phi_raw)
    phi_max = np.max(phi_raw)
    phi_norm = (phi_raw - phi_min) / (phi_max - phi_min)
    
    sensor = CurvatureSensor(dx=substrate.dx, dy=substrate.dy)
    diagnostics = sensor.analyze_field(phi_norm, sigma=1.0)
    
    compiler = TopologyCompiler(substrate, sensor)
    nodes = compiler.extract_nodes(phi_norm, diagnostics, min_distance=0.8, intensity_percentile=35.0)
    
    print(f"Nodes extracted from raw wave field: {len(nodes)}")
    for n in nodes:
        print(f"  Node {n['id']} at {n['pos']} with intensity {n['intensity']:.4f}")

if __name__ == '__main__':
    test()
