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

# Node 0 (-1.4082, -1.2857, -1.2857)
# Node 4 (1.2857, -1.4082, -1.2857)
# Let's parameterize the line segment: p(t) = (1 - t)*Node 0 + t*Node 4
p0 = np.array([-1.4082, -1.2857, -1.2857])
p4 = np.array([1.2857, -1.4082, -1.2857])

# Trilinear Interpolator
def interpolate_3d(field, px, py, pz):
    px = np.clip(px, x[0], x[-1])
    py = np.clip(py, x[0], x[-1])
    pz = np.clip(pz, x[0], x[-1])
    
    tx = (px - x[0]) / dx
    ty = (py - x[0]) / dx
    tz = (pz - x[0]) / dx
    
    i = min(max(int(np.floor(tx)), 0), N - 2)
    j = min(max(int(np.floor(ty)), 0), N - 2)
    k = min(max(int(np.floor(tz)), 0), N - 2)
    
    wx = tx - i
    wy = ty - j
    wz = tz - k
    
    c000 = field[i, j, k]
    c100 = field[i+1, j, k]
    c010 = field[i, j+1, k]
    c110 = field[i+1, j+1, k]
    c001 = field[i, j, k+1]
    c101 = field[i+1, j, k+1]
    c011 = field[i, j+1, k+1]
    c111 = field[i+1, j+1, k+1]
    
    c00 = c000 * (1.0 - wx) + c100 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    
    return c0 * (1.0 - wz) + c1 * wz

print("Profile along Node 0 -> Node 4:")
for t in np.linspace(0, 1, 21):
    pos = (1 - t) * p0 + t * p4
    val = interpolate_3d(Phi_s, pos[0], pos[1], pos[2])
    print(f"  t={t:.2f} pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}): val={val:.4f}")
