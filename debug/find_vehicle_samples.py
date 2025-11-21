"""
Find samples with visible vehicles in camera images
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

cam_names = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'BACK_LEFT', 'BACK', 'BACK_RIGHT']

print("\nSearching for samples with vehicles in different areas...")
print("=" * 80)

front_samples = []  # Vehicles in Y > 0 (front)
back_samples = []   # Vehicles in Y < 0 (back)

for i, batch in enumerate(trainloader):
    if i >= 100:
        break

    imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
    bev_gt = binimgs[0, 0].cpu().numpy()

    vehicle_indices = np.where(bev_gt > 0.5)
    if len(vehicle_indices[0]) == 0:
        continue

    # BEV coordinates: row=Y, col=X
    mean_row = vehicle_indices[0].mean()
    mean_col = vehicle_indices[1].mean()

    # Convert to meters (row 0 = -50m, row 199 = +50m)
    mean_y_meters = (mean_row * 0.5) - 50
    mean_x_meters = (mean_col * 0.5) - 50

    if mean_y_meters > 10:
        front_samples.append((i, mean_x_meters, mean_y_meters, imgs))
    elif mean_y_meters < -10:
        back_samples.append((i, mean_x_meters, mean_y_meters, imgs))

print(f"Found {len(front_samples)} samples with vehicles in FRONT (Y > 10m)")
print(f"Found {len(back_samples)} samples with vehicles in BACK (Y < -10m)")

# Visualize one from each category
if len(front_samples) > 0:
    idx, x, y, imgs = front_samples[0]
    print(f"\nFRONT sample #{idx}: vehicle at X={x:.1f}m, Y={y:.1f}m")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Sample {idx}: Vehicles in FRONT (Y={y:.1f}m)', fontsize=14, fontweight='bold')

    for i in range(6):
        ax = axes[i // 3, i % 3]
        img = imgs[0, i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img)
        ax.set_title(cam_names[i], fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('debug_outputs/front_vehicle_sample.png', dpi=150)
    print(f"Saved: debug_outputs/front_vehicle_sample.png")

if len(back_samples) > 0:
    idx, x, y, imgs = back_samples[0]
    print(f"\nBACK sample #{idx}: vehicle at X={x:.1f}m, Y={y:.1f}m")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Sample {idx}: Vehicles in BACK (Y={y:.1f}m)', fontsize=14, fontweight='bold')

    for i in range(6):
        ax = axes[i // 3, i % 3]
        img = imgs[0, i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img)
        ax.set_title(cam_names[i], fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('debug_outputs/back_vehicle_sample.png', dpi=150)
    print(f"Saved: debug_outputs/back_vehicle_sample.png")

print("\n" + "=" * 80)
print("Check the saved images to see:")
print("- Do FRONT cameras show vehicles when BEV says vehicles are in front?")
print("- Do BACK cameras show vehicles when BEV says vehicles are in back?")
