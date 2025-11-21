"""
Simple test to verify if fliplr is being applied in the data loader
"""

import numpy as np
import matplotlib.pyplot as plt

# Load raw BEV file
bev_path = '/data/SimBEV/SimBEV_cvt_label/scene_0020/yaw0pitch0/bev_scene_0020_000000.npz'
bev_data = np.load(bev_path)
bev = bev_data['bev']

# Extract vehicles (classes 1, 2, 3)
raw_vehicles = ((bev[1] > 0) | (bev[2] > 0) | (bev[3] > 0)).astype(np.float32)

# Apply fliplr manually
flipped_vehicles = np.fliplr(raw_vehicles).copy()

# Check positions
raw_indices = np.where(raw_vehicles > 0.5)
flipped_indices = np.where(flipped_vehicles > 0.5)

print("=" * 60)
print("RAW SimBEV (NO flip):")
print(f"  Col range: [{raw_indices[1].min()}, {raw_indices[1].max()}]")
print(f"  Mean col: {raw_indices[1].mean():.1f}")
if raw_indices[1].mean() > 100:
    print("  → Vehicles in FRONT (col > 100)")
else:
    print("  → Vehicles in BACK (col < 100)")

print("\nAFTER fliplr():")
print(f"  Col range: [{flipped_indices[1].min()}, {flipped_indices[1].max()}]")
print(f"  Mean col: {flipped_indices[1].mean():.1f}")
if flipped_indices[1].mean() > 100:
    print("  → Vehicles in FRONT (col > 100)")
else:
    print("  → Vehicles in BACK (col < 100)")

print("\nExpectation: After fliplr(), vehicles should be in FRONT (col > 100)")
print("=" * 60)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(raw_vehicles, cmap='hot', origin='lower')
axes[0].set_title(f'Raw SimBEV (mean_col={raw_indices[1].mean():.1f})')
axes[0].axvline(x=100, color='white', linestyle='--', alpha=0.5, label='x=100 (center)')
axes[0].legend()

axes[1].imshow(flipped_vehicles, cmap='hot', origin='lower')
axes[1].set_title(f'After fliplr() (mean_col={flipped_indices[1].mean():.1f})')
axes[1].axvline(x=100, color='white', linestyle='--', alpha=0.5, label='x=100 (center)')
axes[1].legend()

plt.tight_layout()
plt.savefig('debug_outputs/flip_test.png', dpi=150)
print(f"\nSaved visualization to: debug_outputs/flip_test.png")
