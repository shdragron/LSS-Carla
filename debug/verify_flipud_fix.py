"""
Verify that flipud() correctly aligns SimBEV with LSS coordinate system
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

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

print("Loading data loader with flipud() fix...")
trainloader, _ = compile_data(
    version='unused',
    dataroot='/data/SimBEV',
    data_aug_conf=data_aug_conf,
    grid_conf=grid_conf,
    bsz=1,
    nworkers=0,
    parser_name='segmentationdata'
)

cam_names = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'BACK_LEFT', 'BACK', 'BACK_RIGHT']

# Find a sample with vehicles visible in FRONT cameras
print("\nSearching for sample with FRONT camera vehicles...")
for i, batch in enumerate(trainloader):
    if i >= 50:
        break

    imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
    bev_gt = binimgs[0, 0].cpu().numpy()

    vehicle_indices = np.where(bev_gt > 0.5)
    if len(vehicle_indices[0]) == 0:
        continue

    # Check if we have reasonable vehicle density
    if len(vehicle_indices[0]) < 20:
        continue

    # Calculate position
    mean_row = vehicle_indices[0].mean()
    mean_col = vehicle_indices[1].mean()
    mean_y_meters = (mean_row * 0.5) - 50
    mean_x_meters = (mean_col * 0.5) - 50

    print(f"Sample {i}: mean_x={mean_x_meters:.1f}m, mean_y={mean_y_meters:.1f}m")

    # Save visualization
    fig = plt.figure(figsize=(20, 12))

    # Top row: 6 camera images
    for j in range(6):
        ax = plt.subplot(3, 6, j + 1)
        img = imgs[0, j].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img)
        ax.set_title(cam_names[j], fontsize=12, fontweight='bold')
        ax.axis('off')

    # Bottom: BEV GT
    ax = plt.subplot(3, 1, 3)
    im = ax.imshow(bev_gt, cmap='hot', origin='lower', extent=[-50, 50, -50, 50])
    ax.axhline(y=0, color='cyan', linestyle='--', alpha=0.5, label='Y=0')
    ax.axvline(x=0, color='cyan', linestyle='--', alpha=0.5, label='X=0')
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'BEV GT after flipud() (mean_y={mean_y_meters:.1f}m)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add directional labels
    ax.text(0, 45, 'FRONT\n(should match FRONT cameras)', ha='center', va='center',
            fontsize=12, color='white', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
    ax.text(0, -45, 'BACK\n(should match BACK cameras)', ha='center', va='center',
            fontsize=12, color='white', bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

    plt.colorbar(im, ax=ax, label='Vehicle presence')
    plt.tight_layout()
    plt.savefig(f'debug_outputs/flipud_verification_sample{i}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: debug_outputs/flipud_verification_sample{i}.png")

    if i > 20:  # Save a few samples
        break

print("\nCheck the saved images:")
print("- Vehicles in BEV FRONT area (+Y) should match FRONT camera views")
print("- Vehicles in BEV BACK area (-Y) should match BACK camera views")
