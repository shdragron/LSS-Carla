"""
Visualize camera images + BEV GT to understand coordinate system
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.data_simbev import compile_data

# Configuration
data_aug_conf = {
    'resize_lim': (1.0, 1.0),
    'final_dim': (128, 352),
    'rot_lim': (0.0, 0.0),
    'H': 224,
    'W': 480,
    'rand_flip': False,
    'bot_pct_lim': (0.0, 0.0),
    'Ncams': 6,
}

grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [4.0, 45.0, 1.0],
}

print("Loading data loader...")
trainloader, _ = compile_data(
    version='unused',
    dataroot='/data/SimBEV',
    data_aug_conf=data_aug_conf,
    grid_conf=grid_conf,
    bsz=1,
    nworkers=0,
    parser_name='segmentationdata'
)

# Get one sample
batch = next(iter(trainloader))
imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch

# Camera names
cam_names = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']

# Extract BEV GT
bev_gt = binimgs[0, 0].cpu().numpy()

# Find vehicle positions in BEV
vehicle_indices = np.where(bev_gt > 0.5)
if len(vehicle_indices[0]) > 0:
    mean_col = vehicle_indices[1].mean()
    mean_row = vehicle_indices[0].mean()
else:
    mean_col = -1
    mean_row = -1

print(f"\nBEV GT vehicle positions:")
print(f"  Mean col: {mean_col:.1f} ({'FRONT' if mean_col > 100 else 'BACK'})")
print(f"  Mean row: {mean_row:.1f}")

# Visualize
fig = plt.figure(figsize=(20, 12))

# Top row: 6 camera images
for i in range(6):
    ax = plt.subplot(3, 6, i + 1)
    img = imgs[0, i].cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    ax.imshow(img)
    ax.set_title(cam_names[i], fontsize=12, fontweight='bold')
    ax.axis('off')

# Bottom: BEV GT
ax = plt.subplot(3, 1, 3)
im = ax.imshow(bev_gt, cmap='hot', origin='lower', extent=[-50, 50, -50, 50])
ax.axhline(y=0, color='cyan', linestyle='--', alpha=0.5, label='Y=0 (ego position)')
ax.axvline(x=0, color='cyan', linestyle='--', alpha=0.5, label='X=0 (ego position)')
ax.set_xlabel('X (meters)', fontsize=12)
ax.set_ylabel('Y (meters)', fontsize=12)
ax.set_title(f'BEV Ground Truth (vehicles mean_col={mean_col:.1f})', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Add annotations
ax.text(0, 45, 'FRONT', ha='center', va='center', fontsize=14, color='white',
        bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
ax.text(0, -45, 'BACK', ha='center', va='center', fontsize=14, color='white',
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
ax.text(-45, 0, 'LEFT', ha='center', va='center', fontsize=14, color='white',
        bbox=dict(boxstyle='round', facecolor='green', alpha=0.7), rotation=90)
ax.text(45, 0, 'RIGHT', ha='center', va='center', fontsize=14, color='white',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7), rotation=90)

plt.colorbar(im, ax=ax, label='Vehicle presence')
plt.tight_layout()
plt.savefig('debug_outputs/camera_bev_visualization.png', dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to: debug_outputs/camera_bev_visualization.png")
print("\nPlease check:")
print("1. Which cameras show vehicles?")
print("2. Do FRONT cameras show vehicles in BEV FRONT area?")
print("3. Do BACK cameras show vehicles in BEV BACK area?")
