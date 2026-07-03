import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Optional

class TSVisualizer:
    """
    Visualizer utility for the TS-Verse Engine.
    Plots continuous fields and overlays the extracted nodes, saddles, and paths.
    """
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def plot_field_and_topology(
        self, 
        phi: np.ndarray, 
        graph: Dict[str, Any], 
        diagnostics: Optional[Dict[str, np.ndarray]] = None,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> None:
        """
        Creates a high-quality visualization of the field and the extracted graph.
        """
        # Set up a dark, modern theme for visualization
        plt.style.use('dark_background')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        
        # Grid bounds for imshow (adjust for extent format [xmin, xmax, ymin, ymax])
        extent = [0, self.width, 0, self.height]
        
        # ----------------------------------------------------
        # Plot 1: The Continuous Field & Topological Skeleton
        # ----------------------------------------------------
        ax_field = axes[0]
        # Transpose phi to match (x, y) orientation in imshow (which expects (row, col) or (y, x))
        im1 = ax_field.imshow(phi.T, origin='lower', extent=extent, cmap='magma', aspect='auto')
        fig.colorbar(im1, ax=ax_field, label='Scalar Phase Field $\\Phi(x, y)$')
        
        # Draw edges (traced paths)
        for edge in graph['edges']:
            path = np.array(edge['path'])
            ax_field.plot(path[:, 0], path[:, 1], color='#00e676', linewidth=2.5, alpha=0.9, zorder=2)
            
        # Draw nodes (local maxima)
        nodes_x = [node['pos'][0] for node in graph['nodes']]
        nodes_y = [node['pos'][1] for node in graph['nodes']]
        ax_field.scatter(nodes_x, nodes_y, color='#ffffff', edgecolor='#00e676', s=120, 
                         marker='o', label='Nodes (Local Maxima)', zorder=4)
        
        # Draw saddles
        saddles_x = [saddle['pos'][0] for saddle in graph['saddles']]
        saddles_y = [saddle['pos'][1] for saddle in graph['saddles']]
        ax_field.scatter(saddles_x, saddles_y, color='#ff9100', edgecolor='#ff3d00', s=100, 
                         marker='^', label='Saddles', zorder=3)
        
        ax_field.set_title("Layer 1 & 3: Wave Interference & Topological Skeleton", fontsize=13)
        ax_field.set_xlabel("X Coordinate")
        ax_field.set_ylabel("Y Coordinate")
        ax_field.legend(loc='upper right')
        
        # ----------------------------------------------------
        # Plot 2: Hessian Curvature & Vector Fields
        # ----------------------------------------------------
        ax_tensor = axes[1]
        
        if diagnostics is not None:
            # We show gradient magnitude or the primary eigenvalue
            # Show lambda_1 to visualize compressive tension valleys/ridges
            l1 = diagnostics['lambda_1']
            im2 = ax_tensor.imshow(l1.T, origin='lower', extent=extent, cmap='coolwarm', aspect='auto')
            fig.colorbar(im2, ax=ax_tensor, label='Hessian Minor Eigenvalue $\\lambda_1(x, y)$')
            
            # Subsample for quiver plot (to keep it readable)
            skip = max(1, phi.shape[0] // 15)
            # Meshgrids for plotting
            x_1d = np.linspace(0, self.width, phi.shape[0])
            y_1d = np.linspace(0, self.height, phi.shape[1])
            X, Y = np.meshgrid(x_1d, y_1d, indexing='ij')
            
            # Major Hessian Eigenvectors (v2 - along the ridges)
            quiver_x = X[::skip, ::skip]
            quiver_y = Y[::skip, ::skip]
            v2_x = diagnostics['v2_x'][::skip, ::skip]
            v2_y = diagnostics['v2_y'][::skip, ::skip]
            
            # Draw quiver vectors
            ax_tensor.quiver(quiver_x, quiver_y, v2_x, v2_y, color='white', alpha=0.4, 
                             scale=20, headwidth=2, headlength=3)
            
            ax_tensor.set_title("Layer 2: Curvature Tensor $\\lambda_1$ & Eigenvector Field $\\mathbf{v}_2$", fontsize=13)
        else:
            ax_tensor.set_title("Layer 2 Diagnostic Field (Not Provided)", fontsize=13)
            
        # Draw skeleton on the tensor plot too for alignment
        for edge in graph['edges']:
            path = np.array(edge['path'])
            ax_tensor.plot(path[:, 0], path[:, 1], color='#e040fb', linewidth=1.5, alpha=0.6, zorder=2)
            
        ax_tensor.set_xlabel("X Coordinate")
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[TSVisualizer] Visualization saved successfully to {save_path}")
            
        if show:
            plt.show()
            
        plt.close()
