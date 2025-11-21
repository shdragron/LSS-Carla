"""
Compare raw SimBEV data vs processed data from loader
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data_simbev import compile_data

# Load one sample
grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [4.0, 45.0, 1.0],
}

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

print("Loading from data loader...")
trainloader, _ = compile_data(
    version='unused',
    dataroot='/data/SimBEV',
    data_aug_conf=data_aug_conf,
    grid_conf=grid_conf,
    bsz=1,
    nworkers=0,
    parser_name='segmentationdata'
)

batch = next(iter(trainloader))
gt_from_loader = batch[-1][0, 0].cpu().numpy()

# Load raw file
print("Loading raw BEV file...")
bev_path = '/data/SimBEV/SimBEV_cvt_label/scene_0020/yaw0pitch0/bev_scene_0020_000000.npz'
bev_data = np.load(bev_path)
bev = bev_data['bev']
raw_vehicles = ((bev[1] > 0) | (bev[2] > 0) | (bev[3] > 0)).astype(np.float32)

# Apply fliplr manually
processed_vehicles = np.fliplr(raw_vehicles).copy()

# Compare
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

raw_indices = np.where(raw_vehicles > 0.5)
processed_indices = np.where(processed_vehicles > 0.5)
loader_indices = np.where(gt_from_loader > 0.5)

print("\nRaw SimBEV (before flip):")
print(f"  Col range: [{raw_indices[1].min()}, {raw_indices[1].max()}]")
print(f"  Mean col: {raw_indices[1].mean():.1f}")

print("\nProcessed (after fliplr):")
print(f"  Col range: [{processed_indices[1].min()}, {processed_indices[1].max()}]")
print(f"  Mean col: {processed_indices[1].mean():.1f}")

print("\nFrom data loader:")
print(f"  Col range: [{loader_indices[1].min()}, {loader_indices[1].max()}]")
print(f"  Mean col: {loader_indices[1].mean():.1f}")

print("\n" + "="*60)
if np.allclose(processed_vehicles, gt_from_loader):
    print("✓ Data loader output MATCHES manual fliplr")
elif np.allclose(raw_vehicles, gt_from_loader):
    print("⚠️  Data loader output MATCHES raw (flip NOT applied!)")
else:
    print("⚠️  Data loader output does NOT match either")
    print(f"  Match with processed: {np.sum(np.abs(processed_vehicles - gt_from_loader)) / gt_from_loader.size:.4f} diff")
    print(f"  Match with raw: {np.sum(np.abs(raw_vehicles - gt_from_loader)) / gt_from_loader.size:.4f} diff")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(raw_vehicles, cmap='hot', origin='lower')
axes[0].set_title('Raw SimBEV')
axes[0].axvline(x=100, color='white', linestyle='--', alpha=0.5)

axes[1].imshow(processed_vehicles, cmap='hot', origin='lower')
axes[1].set_title('After fliplr()')
axes[1].axvline(x=100, color='white', linestyle='--', alpha=0.5)

axes[2].imshow(gt_from_loader, cmap='hot', origin='lower')
axes[2].set_title('From Data Loader')
axes[2].axvline(x=100, color='white', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('debug_outputs/raw_vs_processed.png', dpi=150)
print("\nSaved visualization to: debug_outputs/raw_vs_processed.png")
