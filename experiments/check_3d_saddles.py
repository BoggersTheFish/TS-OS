import numpy as np
import scipy.ndimage as nd

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

for _ in range(steps):
    lap = laplacian_3d(Phi)
    Phi += dt * (D * lap + (Phi - Phi**3) + 1.0 * A)
    Phi = np.clip(Phi, -2.0, 2.0)

Phi_s = nd.gaussian_filter(Phi, sigma=0.8)

# Compute gradients and Hessians
grad_x = np.gradient(Phi_s, axis=0) / dx
grad_y = np.gradient(Phi_s, axis=1) / dx
grad_z = np.gradient(Phi_s, axis=2) / dx
grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

phi_xx = np.gradient(grad_x, axis=0) / dx
phi_xy = np.gradient(grad_x, axis=1) / dx
phi_xz = np.gradient(grad_x, axis=2) / dx
phi_yy = np.gradient(grad_y, axis=1) / dx
phi_yz = np.gradient(grad_y, axis=2) / dx
phi_zz = np.gradient(grad_z, axis=2) / dx

# Compute 3D Hessian Eigenvalues Analytically
p = phi_xx + phi_yy + phi_zz
q = phi_xx * phi_yy + phi_yy * phi_zz + phi_xx * phi_zz - phi_xy**2 - phi_yz**2 - phi_xz**2
r_det = (
    phi_xx * (phi_yy * phi_zz - phi_yz**2) -
    phi_xy * (phi_xy * phi_zz - phi_xz * phi_yz) +
    phi_xz * (phi_xy * phi_yz - phi_xz * phi_yy)
)

Q = (p**2 - 3.0 * q) / 9.0
R = (9.0 * p * q - 2.0 * p**3 - 27.0 * r_det) / 54.0
Q = np.maximum(Q, 0.0)
sqrt_Q3 = np.sqrt(Q**3)
denom = np.maximum(sqrt_Q3, 1e-15)
arg = np.clip(R / denom, -1.0, 1.0)
theta_ang = np.arccos(arg)

l1 = p/3.0 + 2.0 * np.sqrt(Q) * np.cos(theta_ang / 3.0)
l2 = p/3.0 + 2.0 * np.sqrt(Q) * np.cos((theta_ang + 2.0 * np.pi) / 3.0)
l3 = p/3.0 + 2.0 * np.sqrt(Q) * np.cos((theta_ang + 4.0 * np.pi) / 3.0)

lambdas = np.stack([l1, l2, l3], axis=-1)
lambdas = np.sort(lambdas, axis=-1)
lambda_1 = lambdas[..., 0]
lambda_2 = lambdas[..., 1]
lambda_3 = lambdas[..., 2]

# Let's inspect a midpoint coordinate (e.g. x = 1.3, y = 1.3, z = 0)
px, py, pz = 1.3, 1.3, 0.0
ix = np.argmin(np.abs(x - px))
iy = np.argmin(np.abs(x - py))
iz = np.argmin(np.abs(x - pz))

print(f"Midpoint inspect at ({x[ix]:.4f}, {x[iy]:.4f}, {x[iz]:.4f}) [idx {ix}, {iy}, {iz}]:")
print(f"  Grad mag: {grad_mag[ix, iy, iz]:.6f}")
print(f"  Eigenvalues: {lambda_1[ix, iy, iz]:.4f}, {lambda_2[ix, iy, iz]:.4f}, {lambda_3[ix, iy, iz]:.4f}")

# Check local minimum of gradient magnitude
g_val = grad_mag[ix, iy, iz]
is_min = True
for dx_idx in [-1, 0, 1]:
    for dy_idx in [-1, 0, 1]:
        for dz_idx in [-1, 0, 1]:
            if dx_idx == 0 and dy_idx == 0 and dz_idx == 0:
                continue
            if g_val > grad_mag[ix+dx_idx, iy+dy_idx, iz+dz_idx]:
                is_min = False
                print(f"  Fail min grad: center {g_val:.6f} > neighbor at offset {dx_idx, dy_idx, dz_idx} which is {grad_mag[ix+dx_idx, iy+dy_idx, iz+dz_idx]:.6f}")
                
print(f"  Is local min of grad mag: {is_min}")
print(f"  35th percentile of grad mag: {np.percentile(grad_mag, 35.0):.6f}")
