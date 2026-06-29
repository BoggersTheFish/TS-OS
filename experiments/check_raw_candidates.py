import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.substrate import FieldSubstrate
from core.tensors import CurvatureSensor
from core.compiler import TopologyCompiler

def check():
    width, height = 10.0, 10.0
    res_x, res_y = 200, 200
    substrate = FieldSubstrate(width=width, height=height, resolution_x=res_x, resolution_y=res_y)
    
    substrate.add_source(2.5, 2.5, amplitude=1.2, k=2.0 * np.pi / 3.0, phase=0.0)
    substrate.add_source(7.5, 2.5, amplitude=1.2, k=2.0 * np.pi / 3.0, phase=0.0)
    substrate.add_source(2.5, 7.5, amplitude=1.2, k=2.0 * np.pi / 3.0, phase=0.0)
    substrate.add_source(7.5, 7.5, amplitude=1.2, k=2.0 * np.pi / 3.0, phase=0.0)
    
    substrate.compute_wave_interference()
    phi_raw = substrate.phi.copy()
    
    phi_min = np.min(phi_raw)
    phi_max = np.max(phi_raw)
    phi_norm = (phi_raw - phi_min) / (phi_max - phi_min)
    
    phi_padded = np.pad(phi_norm, 1, mode='edge')
    is_max = (
        (phi_norm > phi_padded[:-2, 1:-1]) & (phi_norm > phi_padded[2:, 1:-1]) &
        (phi_norm > phi_padded[1:-1, :-2]) & (phi_norm > phi_padded[1:-1, 2:]) &
        (phi_norm > phi_padded[:-2, :-2]) & (phi_norm > phi_padded[:-2, 2:]) &
        (phi_norm > phi_padded[2:, :-2]) & (phi_norm > phi_padded[2:, 2:])
    )
    
    indices_x, indices_y = np.where(is_max)
    print(f"Candidates before NMS: {len(indices_x)}")
    for i in range(len(indices_x)):
        ix, iy = indices_x[i], indices_y[i]
        px = substrate.x[ix]
        py = substrate.y[iy]
        print(f"  Candidate at ({px:.4f}, {py:.4f}) with val {phi_norm[ix, iy]:.4f}")

if __name__ == '__main__':
    check()
