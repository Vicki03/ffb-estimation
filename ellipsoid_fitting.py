import open3d as o3d
import numpy as np
import os
import glob

# =====================================================
# SETTINGS
# =====================================================
FFB_ID = 12
FRAMES_DIR = f"sample_{FFB_ID:03d}"
PLY_FILES = sorted(glob.glob(os.path.join(FRAMES_DIR, "ffb_frame_*.ply")))

DENSITY_KG_PER_M3 = 956.28
PERCENTILE = 98   # try 95–97 later if needed
MIN_POINTS = 500

print(f"\nEstimating mass for FFB {FFB_ID}")
print(f"Found {len(PLY_FILES)} point clouds\n")

if not PLY_FILES:
    raise RuntimeError("No ffb_frame_*.ply files found")

# =====================================================
# ELLIPSOID MASS FUNCTION (UNCHANGED LOGIC)
# =====================================================
def ellipsoid_mass_from_ply(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    if len(points) < MIN_POINTS:
        return None

    center = points.mean(axis=0)
    X = points - center

    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    proj = X @ eigvecs

    a = np.percentile(np.abs(proj[:, 0]), PERCENTILE)
    b = np.percentile(np.abs(proj[:, 1]), PERCENTILE)
    c = np.percentile(np.abs(proj[:, 2]), PERCENTILE)

    volume = (4.0 / 3.0) * np.pi * a * b * c
    mass = volume * DENSITY_KG_PER_M3

    return mass, volume, (a, b, c)

# =====================================================
# PROCESS ALL FRAMES
# =====================================================
results = []

for ply in PLY_FILES:
    name = os.path.basename(ply)
    out = ellipsoid_mass_from_ply(ply)

    if out is None:
        print(f"{name:<18} → skipped (too few points)")
        continue

    mass, volume, axes = out
    results.append(mass)

    print(
        f"{name:<18} | "
        f"mass = {mass:6.2f} kg | "
        f"vol = {volume:.5f} m³ | "
        f"a,b,c = ({axes[0]:.3f}, {axes[1]:.3f}, {axes[2]:.3f})"
    )

# =====================================================
# SUMMARY STATISTICS
# =====================================================
results = np.array(results)

print("\n================ SUMMARY ================")
print(f"Frames used:        {len(results)}")
print(f"Mean mass (kg):     {results.mean():.2f}")
print(f"Median mass (kg):   {np.median(results):.2f}")
print(f"Std dev (kg):       {results.std():.2f}")
print("========================================")
