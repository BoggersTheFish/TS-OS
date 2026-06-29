import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Dict, Tuple, Optional

class CurvatureSensor:
    """
    Layer 2: Curvature Sensor
    Computes first and second-order differential geometric features of the field.
    Calculates gradient vector fields and Hessian eigenvalue/eigenvector fields.
    """
    def __init__(self, dx: float, dy: float):
        self.dx = dx
        self.dy = dy

    def analyze_field(
        self, 
        phi: np.ndarray, 
        sigma: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """
        Analyzes the scalar field Phi.
        Optional sigma parameter applies Gaussian smoothing to suppress high-frequency noise.
        Returns a dictionary containing:
            - 'grad_x', 'grad_y': First derivatives
            - 'grad_mag': Gradient magnitude
            - 'phi_xx', 'phi_xy', 'phi_yy': Hessian components
            - 'lambda_1', 'lambda_2': Sorted eigenvalues (lambda_1 <= lambda_2)
            - 'v2_x', 'v2_y': Major eigenvector orientation components (for lambda_2)
            - 'v1_x', 'v1_y': Minor eigenvector orientation components (for lambda_1)
        """
        # Apply smoothing if requested
        if sigma > 0.0:
            phi_smooth = gaussian_filter(phi, sigma=sigma)
        else:
            phi_smooth = phi.copy()

        # Compute gradient (1st derivatives)
        grad_x = np.gradient(phi_smooth, axis=0) / self.dx
        grad_y = np.gradient(phi_smooth, axis=1) / self.dy
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Compute Hessian components (2nd derivatives)
        phi_xx = np.gradient(grad_x, axis=0) / self.dx
        phi_xy = np.gradient(grad_x, axis=1) / self.dy
        phi_yy = np.gradient(grad_y, axis=1) / self.dy

        # Compute trace and determinant of Hessian
        trace = phi_xx + phi_yy
        det = phi_xx * phi_yy - phi_xy**2

        # Eigenvalues: lambda = (Tr +- sqrt(Tr^2 - 4*det)) / 2
        # Since Hessian is symmetric, the discriminant is always >= 0
        discriminant = np.maximum((phi_xx - phi_yy)**2 + 4.0 * phi_xy**2, 0.0)
        sqrt_disc = np.sqrt(discriminant)

        lambda_1 = (trace - sqrt_disc) / 2.0
        lambda_2 = (trace + sqrt_disc) / 2.0

        # Vectorized eigenvector calculation:
        # The angle of the principal eigenvector (for lambda_2) is:
        # theta = 0.5 * arctan2(2 * phi_xy, phi_xx - phi_yy)
        theta = 0.5 * np.arctan2(2.0 * phi_xy, phi_xx - phi_yy)
        
        # Major eigenvector (associated with lambda_2)
        v2_x = np.cos(theta)
        v2_y = np.sin(theta)

        # Minor eigenvector (associated with lambda_1, orthogonal to v2)
        v1_x = -v2_y
        v1_y = v2_x

        return {
            'grad_x': grad_x,
            'grad_y': grad_y,
            'grad_mag': grad_mag,
            'phi_xx': phi_xx,
            'phi_xy': phi_xy,
            'phi_yy': phi_yy,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'v2_x': v2_x,
            'v2_y': v2_y,
            'v1_x': v1_x,
            'v1_y': v1_y
        }
        
    def classify_topology(
        self, 
        diagnostics: Dict[str, np.ndarray], 
        grad_threshold: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Classifies critical points (where gradient magnitude is low).
        Returns three boolean masks: maxima_mask, minima_mask, saddle_mask.
        """
        grad_mag = diagnostics['grad_mag']
        lambda_1 = diagnostics['lambda_1']
        lambda_2 = diagnostics['lambda_2']

        # Critical points are local extrema of the gradient magnitude (near zero)
        # We find points where gradient magnitude is below a threshold
        is_critical = grad_mag < grad_threshold

        # Local Maxima: lambda_1 < 0 and lambda_2 < 0
        maxima_mask = is_critical & (lambda_1 < 0.0) & (lambda_2 < 0.0)

        # Local Minima: lambda_1 > 0 and lambda_2 > 0
        minima_mask = is_critical & (lambda_1 > 0.0) & (lambda_2 > 0.0)

        # Saddles: lambda_1 < 0 and lambda_2 > 0 (different signs)
        saddle_mask = is_critical & (lambda_1 < 0.0) & (lambda_2 > 0.0)

        return maxima_mask, minima_mask, saddle_mask
