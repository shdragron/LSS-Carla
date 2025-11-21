"""
Load raw BEV file AND processed data from loader for SAME sample
Compare to determine if flipud is needed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

# Temporarily modify data_simbev to NOT apply flipud, so we can compare
import src.data_simbev as data_module

# Get one specific sample
from src.data_simbev import SimBEVDataset

dataset = SimBEVDataset(
    version='unused',
    dataroot='/data/SimBEV',
    is_train=True,
    data_aug_conf={
        'resize_lim': (1.0, 1.0),
        'final_dim': (128, 352),
        'rot_lim': (0.0, 0.0),
        'H': 224,
        'W': 480,
        'rand_flip': False,
        'bot_pct_lim': (0.0, 0.0),
        'Ncams': 6,
    },
    grid_conf={
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
    },
    parser_name='segmentationdata'
)

# Get sample 1 (from our previous test with BMW)
sample_idx = 1
sample = dataset.samples[sample_idx]
print(f"Sample {sample_idx}:")
print(f"  meta_dir: {sample['meta_dir']}")
print(f"  bev: {sample['bev']}")

# Load RAW BEV
bev_path = sample['meta_dir'] / sample['bev']
bev_data = np.load(bev_path)
bev_raw = bev_data['bev']
vehicle_mask_raw = ((bev_raw[1] > 0) | (bev_raw[2] > 0) | (bev_raw[3] > 0)).astype(np.float32)

# Load from dataset (currently NO flipud)
imgs, rots, trans, intrins, post_rots, post_trans, binimg = dataset[sample_idx]
bev_from_loader = binimg[0].numpy()

# Apply flipud manually
vehicle_mask_flipped = np.flipud(vehicle_mask_raw).copy()

# Calculate positions
def get_position(mask):
    indices = np.where(mask > 0.5)
    if len(indices[0]) > 0:
        mean_row = indices[0].mean()
        mean_y = (mean_row * 0.5) - 50
        return mean_y
    return 0

y_raw = get_position(vehicle_mask_raw)
y_flipped = get_position(vehicle_mask_flipped)
y_loader = get_position(bev_from_loader)

print(f"\nBEV vehicle Y positions:")
print(f"  RAW SimBEV: {y_raw:.1f}m")
print(f"  After flipud(): {y_flipped:.1f}m")
print(f"  From loader (current): {y_loader:.1f}m")

# Load camera images
cam_names = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'BACK_LEFT', 'BACK', 'BACK_RIGHT']

fig = plt.figure(figsize=(20, 15))

# Row 1: Cameras
for i in range(6):
    ax = plt.subplot(4, 6, i + 1)
    img = imgs[i].numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    ax.imshow(img)
    ax.set_title(cam_names[i], fontsize=12, fontweight='bold')
    ax.axis('off')

# Row 2: RAW BEV
ax = plt.subplot(4, 3, 7)
im = ax.imshow(vehicle_mask_raw, cmap='hot', origin='lower', extent=[-50, 50, -50, 50])
ax.set_title(f'RAW SimBEV (y={y_raw:.1f}m)', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='cyan', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='cyan', linestyle='--', alpha=0.5)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.text(0, 40, 'Which cameras\nsee these vehicles?', ha='center', fontsize=10,
        color='yellow', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# Row 3: After flipud
ax = plt.subplot(4, 3, 8)
im = ax.imshow(vehicle_mask_flipped, cmap='hot', origin='lower', extent=[-50, 50, -50, 50])
ax.set_title(f'After flipud() (y={y_flipped:.1f}m)', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='cyan', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='cyan', linestyle='--', alpha=0.5)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.text(0, 40, 'Does THIS match\ncamera views better?', ha='center', fontsize=10,
        color='yellow', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# Row 4: From loader
ax = plt.subplot(4, 3, 9)
im = ax.imshow(bev_from_loader, cmap='hot', origin='lower', extent=[-50, 50, -50, 50])
ax.set_title(f'From loader/current (y={y_loader:.1f}m)', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='cyan', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='cyan', linestyle='--', alpha=0.5)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')

plt.tight_layout()
plt.savefig('debug_outputs/definitive_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: debug_outputs/definitive_comparison.png")
print("\n" + "="*80)
print("DECISION GUIDE:")
print("="*80)
print("Look at the camera images (top row):")
print("- If you see vehicles in FRONT cameras → those vehicles should be at +Y in BEV")
print("- If you see vehicles in BACK cameras → those vehicles should be at -Y in BEV")
print("\nWhich BEV layout (RAW vs flipud) matches the camera views?")
print("="*80)
