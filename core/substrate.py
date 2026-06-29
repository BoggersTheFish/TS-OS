import numpy as np
from typing import List, Dict, Tuple, Optional

class FieldSubstrate:
    """
    Layer 1: Field Substrate
    Manages continuous wave interference fields and reaction-diffusion (Allen-Cahn) PDEs.
    Operates strictly on the continuous grid space, evaluating scalar fields and energy states.
    """
    def __init__(
        self, 
        width: float = 10.0, 
        height: float = 10.0, 
        resolution_x: int = 200, 
        resolution_y: int = 200
    ):
        self.width = width
        self.height = height
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        
        # Grid definition
        self.x = np.linspace(0, width, resolution_x)
        self.y = np.linspace(0, height, resolution_y)
        self.dx = width / (resolution_x - 1)
        self.dy = height / (resolution_y - 1)
        
        # 2D Meshgrids
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Field initialization (zeros by default)
        self.phi = np.zeros((resolution_x, resolution_y), dtype=np.float64)
        
        # Sources for wave interference: list of dicts with keys (x, y, amplitude, k, phase)
        self.sources: List[Dict[str, float]] = []

    def clear_sources(self) -> None:
        """Removes all wave sources."""
        self.sources = []

    def add_source(
        self, 
        x: float, 
        y: float, 
        amplitude: float = 1.0, 
        k: float = 2.0 * np.pi / 2.0, 
        phase: float = 0.0
    ) -> None:
        """Adds an isotropic wave source to the substrate."""
        self.sources.append({
            'x': x,
            'y': y,
            'amplitude': amplitude,
            'k': k,
            'phase': phase
        })

    def compute_wave_interference(self, t: float = 0.0, omega: float = 0.0) -> np.ndarray:
        """
        Computes the stationary or time-dependent wave interference field from all sources.
        Uses the superposition principle: Phi(x, y) = sum_j A_j * cos(k_j * r_j - omega * t + phase_j)
        """
        field = np.zeros_like(self.X)
        for src in self.sources:
            r = np.sqrt((self.X - src['x'])**2 + (self.Y - src['y'])**2)
            # Prevent singularity at the exact source center
            r = np.maximum(r, 1e-9)
            field += src['amplitude'] * np.cos(src['k'] * r - omega * t + src['phase'])
        
        self.phi = field
        return self.phi

    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Computes the 2D Laplacian of the field using finite differences with Neumann boundary conditions.
        """
        lap = np.zeros_like(field)
        
        # Interior points
        lap[1:-1, 1:-1] = (
            (field[2:, 1:-1] + field[:-2, 1:-1] - 2 * field[1:-1, 1:-1]) / (self.dx**2) +
            (field[1:-1, 2:] + field[1:-1, :-2] - 2 * field[1:-1, 1:-1]) / (self.dy**2)
        )
        
        # Neumann boundary conditions: zero derivative normal to boundaries
        # Copy values from adjacent interior rows/columns
        lap[0, :] = lap[1, :]
        lap[-1, :] = lap[-2, :]
        lap[:, 0] = lap[:, 1]
        lap[:, -1] = lap[:, -2]
        
        return lap

    def evolve_allen_cahn(
        self, 
        dt: float = 0.01, 
        D: float = 0.1, 
        iterations: int = 100,
        beta: float = 1.0
    ) -> np.ndarray:
        """
        Evolves the scalar field using the Allen-Cahn reaction-diffusion equation.
        dPhi/dt = D * Laplacian(Phi) - beta * f'(Phi)
        where f(Phi) = 0.25 * (Phi^2 - 1)^2 is the double-well potential,
        so f'(Phi) = Phi^3 - Phi.
        """
        for _ in range(iterations):
            lap = self.compute_laplacian(self.phi)
            # Reaction term f'(Phi) = Phi^3 - Phi
            reaction = self.phi**3 - self.phi
            # Euler integration step
            self.phi += dt * (D * lap - beta * reaction)
            
            # Boundary damping or clipping to ensure stability
            self.phi = np.clip(self.phi, -2.0, 2.0)
            
        return self.phi

    def normalize_field(self) -> np.ndarray:
        """Normalizes the field values to [0, 1] range."""
        phi_min = np.min(self.phi)
        phi_max = np.max(self.phi)
        if phi_max - phi_min > 1e-9:
            self.phi = (self.phi - phi_min) / (phi_max - phi_min)
        else:
            self.phi = np.zeros_like(self.phi)
        return self.phi
