"""
Compare raw SimBEV BEV with camera images to determine correct orientation
Load the EXACT same sample and check manually
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Sample info from our previous test
# Sample 1 from back_vehicle_sample.png had vehicles clearly visible
# Let's find that exact file

# Load a known BEV file
bev_path = '/data/SimBEV/SimBEV_cvt_label/scene_0020/yaw0pitch0/bev_scene_0020_000000.npz'
bev_data = np.load(bev_path)
bev = bev_data['bev']  # Shape: (8, 200, 200)

# Extract vehicles
vehicle_mask = ((bev[1] > 0) | (bev[2] > 0) | (bev[3] > 0)).astype(np.float32)

# Apply flipud
vehicle_mask_flipped = np.flipud(vehicle_mask).copy()

# Find mean position
indices_raw = np.where(vehicle_mask > 0.5)
indices_flipped = np.where(vehicle_mask_flipped > 0.5)

if len(indices_raw[0]) > 0:
    mean_row_raw = indices_raw[0].mean()
    mean_y_raw = (mean_row_raw * 0.5) - 50

    mean_row_flipped = indices_flipped[0].mean()
    mean_y_flipped = (mean_row_flipped * 0.5) - 50
else:
    mean_y_raw = 0
    mean_y_flipped = 0

# Load corresponding camera images
img_dir = Path('/data/SimBEV/sweeps')
cam_names = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'BACK_LEFT', 'BACK', 'BACK_RIGHT']
cam_folders = ['RGB-CAM_FRONT_LEFT', 'RGB-CAM_FRONT', 'RGB-CAM_FRONT_RIGHT',
               'RGB-CAM_BACK_LEFT', 'RGB-CAM_BACK', 'RGB-CAM_BACK_RIGHT']

# Find images for scene_0020_000000
imgs = []
for cam_folder in cam_folders:
    # Try to find matching image
    img_files = list((img_dir / cam_folder).glob('*scene_0020_000000*.png'))
    if len(img_files) > 0:
        img = Image.open(img_files[0])
        imgs.append(np.array(img))
    else:
        imgs.append(np.zeros((224, 480, 3), dtype=np.uint8))

# Visualize
fig = plt.figure(figsize=(20, 15))

# Row 1: Camera images
for i in range(6):
    ax = plt.subplot(4, 6, i + 1)
    ax.imshow(imgs[i])
    ax.set_title(cam_names[i], fontsize=12, fontweight='bold')
    ax.axis('off')

# Row 2: Raw BEV
ax = plt.subplot(4, 2, 3)
im = ax.imshow(vehicle_mask, cmap='hot', origin='lower', extent=[-50, 50, -50, 50])
ax.set_title(f'RAW SimBEV BEV (mean_y={mean_y_raw:.1f}m)', fontsize=14, fontweight='bold')
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.axhline(y=0, color='cyan', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='cyan', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)
plt.colorbar(im, ax=ax)

# Row 3: After flipud
ax = plt.subplot(4, 2, 4)
im = ax.imshow(vehicle_mask_flipped, cmap='hot', origin='lower', extent=[-50, 50, -50, 50])
ax.set_title(f'After flipud() (mean_y={mean_y_flipped:.1f}m)', fontsize=14, fontweight='bold')
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.axhline(y=0, color='cyan', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='cyan', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('debug_outputs/raw_vs_flipud_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: debug_outputs/raw_vs_flipud_comparison.png")

print("\n" + "="*80)
print("MANUAL INSPECTION NEEDED:")
print("="*80)
print(f"RAW BEV: vehicles at mean_y={mean_y_raw:.1f}m")
print(f"After flipud(): vehicles at mean_y={mean_y_flipped:.1f}m")
print("\nLook at the camera images and BEV:")
print("- Which BEV orientation matches the camera views?")
print("- If vehicles visible in FRONT cameras, they should be at +Y")
print("- If vehicles visible in BACK cameras, they should be at -Y")
print("="*80)
