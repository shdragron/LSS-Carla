"""
Debug script to verify data loading and GT processing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_simbev import compile_data
from src.tools import gen_dx_bx

def check_data_loading():
    """Check if data is properly loaded and transformed"""

    # Configuration
    dataroot = '/data/SimBEV'

    grid_conf = {
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
    }

    data_aug_conf = {
        'resize_lim': (0.9, 1.1),
        'final_dim': (128, 352),
        'rot_lim': (-5.4, 5.4),
        'H': 224,
        'W': 480,
        'rand_flip': False,  # Disable for debugging
        'bot_pct_lim': (0.0, 0.0),
        'Ncams': 6,
    }

    print("=" * 80)
    print("DATA LOADING DEBUG")
    print("=" * 80)

    # Load data
    print("\n1. Loading dataset...")
    trainloader, valloader = compile_data(
        version='unused',
        dataroot=dataroot,
        data_aug_conf=data_aug_conf,
        grid_conf=grid_conf,
        bsz=2,
        nworkers=0,  # Single thread for debugging
        parser_name='segmentationdata'
    )

    # Get one batch
    print("\n2. Getting one batch...")
    for batch_data in trainloader:
        imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch_data
        break

    # Check shapes
    print("\n3. Checking tensor shapes...")
    print(f"   imgs shape:       {imgs.shape}       (expected: [B, N_cams, 3, H, W])")
    print(f"   rots shape:       {rots.shape}       (expected: [B, N_cams, 3, 3])")
    print(f"   trans shape:      {trans.shape}      (expected: [B, N_cams, 3])")
    print(f"   intrins shape:    {intrins.shape}    (expected: [B, N_cams, 3, 3])")
    print(f"   post_rots shape:  {post_rots.shape}  (expected: [B, N_cams, 3, 3])")
    print(f"   post_trans shape: {post_trans.shape} (expected: [B, N_cams, 3])")
    print(f"   binimgs shape:    {binimgs.shape}    (expected: [B, 1, 200, 200])")

    # Check value ranges
    print("\n4. Checking value ranges...")
    print(f"   imgs:    min={imgs.min():.3f}, max={imgs.max():.3f}, mean={imgs.mean():.3f}")
    print(f"   binimgs: min={binimgs.min():.3f}, max={binimgs.max():.3f}, mean={binimgs.mean():.3f}")
    print(f"   binimgs positive ratio: {binimgs.sum() / binimgs.numel():.4f} ({binimgs.sum().item():.0f}/{binimgs.numel()})")

    # Check extrinsics (rotation matrices should be orthogonal)
    print("\n5. Checking rotation matrices (should be orthogonal)...")
    for i in range(min(2, rots.shape[1])):  # Check first 2 cameras
        R = rots[0, i]  # [3, 3]
        identity = R @ R.T
        is_orthogonal = torch.allclose(identity, torch.eye(3), atol=1e-4)
        det = torch.det(R)
        print(f"   Camera {i}: orthogonal={is_orthogonal}, det={det:.4f} (should be ~1.0)")

    # Check intrinsics
    print("\n6. Checking intrinsics...")
    print(f"   First camera intrinsics:")
    print(f"   {intrins[0, 0]}")
    fx = intrins[0, 0, 0, 0]
    fy = intrins[0, 0, 1, 1]
    cx = intrins[0, 0, 0, 2]
    cy = intrins[0, 0, 1, 2]
    print(f"   fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # Check translation vectors
    print("\n7. Checking translation vectors (camera positions)...")
    print(f"   Sample translations for batch 0:")
    for i in range(rots.shape[1]):
        t = trans[0, i]
        print(f"   Camera {i}: x={t[0]:.3f}, y={t[1]:.3f}, z={t[2]:.3f}")

    # Save visualization
    print("\n8. Saving visualizations...")
    save_dir = Path('./debug_outputs')
    save_dir.mkdir(exist_ok=True)

    # Plot GT BEV for first sample
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # GT
    axes[0].imshow(binimgs[0, 0].cpu().numpy(), cmap='gray', origin='lower')
    axes[0].set_title(f'GT BEV (Class 0)\nPositive: {binimgs[0, 0].sum():.0f} pixels')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')

    # First camera image (denormalized)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_denorm = imgs[0, 0] * std + mean
    img_denorm = torch.clamp(img_denorm, 0, 1)
    axes[1].imshow(img_denorm.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title('First Camera Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / 'data_loading_debug.png', dpi=150, bbox_inches='tight')
    print(f"   Saved to: {save_dir / 'data_loading_debug.png'}")

    # Check BEV statistics for multiple batches
    print("\n9. Checking GT statistics across multiple batches...")
    num_batches = min(10, len(trainloader))
    gt_stats = []

    for i, batch_data in enumerate(trainloader):
        if i >= num_batches:
            break
        _, _, _, _, _, _, binimgs = batch_data
        positive_ratio = binimgs.sum() / binimgs.numel()
        gt_stats.append(positive_ratio.item())

    print(f"   Checked {num_batches} batches:")
    print(f"   GT positive ratio: min={min(gt_stats):.4f}, max={max(gt_stats):.4f}, mean={np.mean(gt_stats):.4f}")

    if np.mean(gt_stats) < 0.001:
        print("   ⚠️  WARNING: GT has very few positive pixels! Check if correct class is selected.")
    elif np.mean(gt_stats) > 0.5:
        print("   ⚠️  WARNING: GT has too many positive pixels! Check if BEV is inverted.")
    else:
        print("   ✓ GT positive ratio looks reasonable")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    issues = []

    # Check common issues
    if binimgs.shape[-1] != 200 or binimgs.shape[-2] != 200:
        issues.append("BEV size mismatch")

    if binimgs.max() == 0:
        issues.append("GT is all zeros!")

    if imgs.min() < -5 or imgs.max() > 5:
        issues.append("Image normalization might be wrong")

    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✓ No obvious issues found!")

    print("\nNext steps:")
    print("1. Check the visualization at: ./debug_outputs/data_loading_debug.png")
    print("2. Verify that the GT BEV shows meaningful patterns")
    print("3. If GT looks wrong, investigate class selection in data_simbev.py:235")
    print("4. If camera poses look wrong, check extrinsics transformation in data_simbev.py:178-189")


if __name__ == '__main__':
    check_data_loading()
