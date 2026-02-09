import cv2
import numpy as np

# =====================================================
# INPUT FILES
# =====================================================
FRONT_DEPTH = r"C:\Users\admin\Documents\Vicki\School stuff\SEGP\sample_001\depth\depth_0001.png"
FRONT_MASK  = r"C:\Users\admin\Documents\Vicki\School stuff\SEGP\sample_001\masks\mask_0001.png"

BACK_DEPTH  = r"C:\Users\admin\Documents\Vicki\School stuff\SEGP\sample_001\depth\depth_0040.png"
BACK_MASK   = r"C:\Users\admin\Documents\Vicki\School stuff\SEGP\sample_001\masks\mask_0040.png"

# =====================================================
# CAMERA INTRINSICS
# =====================================================
fx = 646.2672119140625
fy = 645.5238037109375
cx = 641.302490234375
cy = 360.6357116699219

depth_scale = 0.001  # mm → meters (common for RealSense PNG exports)

# =====================================================
# THESIS PARAMETERS
# =====================================================
DENSITY_KG_PER_M3 = 956.28

# Your thickness guess (tune later)
TOTAL_THICKNESS_M = 0.25  # 25 cm
HALF_THICKNESS_M  = TOTAL_THICKNESS_M / 2

# If you know the real mass, put it here for calibration
ACTUAL_MASS_KG = None  # set None if you don't want thickness calibration

# =====================================================
# HELPER
# =====================================================
def load_depth_and_mask(depth_path, mask_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth: {depth_path}")
    depth = depth.astype(np.float32)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    # Make mask strictly binary (important!)
    # Many masks are 0/255, but sometimes they are soft/anti-aliased.
    mask_bin = (mask >= 128)

    return depth, mask_bin

def silhouette_area_from_view(depth_path, mask_path):
    depth, mask_bin = load_depth_and_mask(depth_path, mask_path)

    depth_m = depth * depth_scale
    valid = mask_bin & (depth_m > 0)

    n_pix = int(np.count_nonzero(valid))
    if n_pix == 0:
        return {
            "n_pix": 0, "z_mean": 0.0, "pixel_area": 0.0,
            "silhouette_area": 0.0
        }

    zs = depth_m[valid]

    # Use median depth (more robust than mean when edges are noisy)
    z_med = float(np.median(zs))

    # Pixel area at representative depth
    px_area = (z_med ** 2) / (fx * fy)

    # Silhouette area = pixel_count * pixel_area
    A = float(n_pix * px_area)

    return {
        "n_pix": n_pix,
        "z_mean": float(np.mean(zs)),
        "z_med": z_med,
        "pixel_area": float(px_area),
        "silhouette_area": A
    }

# =====================================================
# COMPUTE FRONT + BACK
# =====================================================
front = silhouette_area_from_view(FRONT_DEPTH, FRONT_MASK)
back  = silhouette_area_from_view(BACK_DEPTH,  BACK_MASK)

A_f = front["silhouette_area"]
A_b = back["silhouette_area"]
A_sum = A_f + A_b

# Volume estimate using your chosen thickness:
V_est = A_sum * HALF_THICKNESS_M
M_est = V_est * DENSITY_KG_PER_M3

# Thickness needed to match actual mass (optional):
T_needed = None
if ACTUAL_MASS_KG is not None and A_sum > 0:
    V_target = ACTUAL_MASS_KG / DENSITY_KG_PER_M3
    T_needed = (2 * V_target) / A_sum  # because V = (A_sum * T)/2

# =====================================================
# PRINT DIAGNOSTICS
# =====================================================
print("===== FRONT VIEW =====")
print(f"Mask pixels:          {front['n_pix']}")
print(f"Mean depth (m):       {front['z_mean']:.3f}")
print(f"Median depth (m):     {front['z_med']:.3f}")
print(f"Pixel area (m²):      {front['pixel_area']:.8f}")
print(f"Silhouette area (m²): {A_f:.6f}")

print("\n===== BACK VIEW =====")
print(f"Mask pixels:          {back['n_pix']}")
print(f"Mean depth (m):       {back['z_mean']:.3f}")
print(f"Median depth (m):     {back['z_med']:.3f}")
print(f"Pixel area (m²):      {back['pixel_area']:.8f}")
print(f"Silhouette area (m²): {A_b:.6f}")

print("\n===== COMBINED =====")
print(f"Total silhouette area (m²): {A_sum:.6f}")
print(f"Thickness used (m):         {TOTAL_THICKNESS_M:.3f}")
print(f"Estimated volume (m³):      {V_est:.6f}")
print(f"Estimated mass (kg):        {M_est:.2f}")

if T_needed is not None:
    print("\n===== CALIBRATION TO MATCH ACTUAL MASS =====")
    print(f"Actual mass (kg):           {ACTUAL_MASS_KG:.2f}")
    print(f"Target volume (m³):         {ACTUAL_MASS_KG / DENSITY_KG_PER_M3:.6f}")
    print(f"Required thickness (m):     {T_needed:.3f}  ({T_needed*100:.1f} cm)")
