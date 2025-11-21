"""
Final verification that coordinate system is correctly aligned after fix.
This script validates that GT vehicles are positioned in front of ego (col > 100).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
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

print("\nAnalyzing coordinate system...")
print("=" * 60)

mean_cols = []
samples_analyzed = 0

for i, batch in enumerate(trainloader):
    if i >= 200:
        break

    # Extract GT BEV
    gt = batch[-1][0, 0].cpu().numpy()  # Shape: (200, 200)

    # Find vehicle positions
    vehicle_indices = np.where(gt > 0.5)

    if len(vehicle_indices[0]) == 0:
        continue

    # Calculate mean column position
    mean_col = vehicle_indices[1].mean()
    mean_cols.append(mean_col)
    samples_analyzed += 1

    # Print samples with vehicles near the back
    if mean_col < 105:
        print(f"Sample {i}: mean_col={mean_col:.1f} ({'✓ OK' if mean_col >= 95 else '⚠️  SUSPICIOUS'})")

mean_cols = np.array(mean_cols)

print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
print(f"Samples analyzed: {samples_analyzed}")
print(f"\nPosition statistics:")
print(f"  Mean: {mean_cols.mean():.1f}")
print(f"  Median: {np.median(mean_cols):.1f}")
print(f"  Std: {mean_cols.std():.1f}")
print(f"  Min: {mean_cols.min():.1f}")
print(f"  Max: {mean_cols.max():.1f}")

print(f"\nDistribution:")
print(f"  col < 95 (BACK - BAD): {np.sum(mean_cols < 95)} ({100*np.sum(mean_cols < 95)/len(mean_cols):.1f}%)")
print(f"  95 <= col < 105 (CENTER): {np.sum((mean_cols >= 95) & (mean_cols < 105))} ({100*np.sum((mean_cols >= 95) & (mean_cols < 105))/len(mean_cols):.1f}%)")
print(f"  col >= 105 (FRONT - GOOD): {np.sum(mean_cols >= 105)} ({100*np.sum(mean_cols >= 105)/len(mean_cols):.1f}%)")

print("\n" + "=" * 60)
if np.sum(mean_cols < 95) == 0:
    print("✓ SUCCESS: All samples have vehicles correctly positioned")
    print("  Coordinate system is properly aligned with LSS expectations")
else:
    print(f"⚠️  WARNING: {np.sum(mean_cols < 95)} samples still have misaligned coordinates")

print("=" * 60)
