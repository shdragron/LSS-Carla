"""
Detailed coordinate system verification - check EXACT alignment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.models import compile_model
from src.data_simbev import compile_data
from src.tools import gen_dx_bx

def detailed_check():
    """Detailed coordinate alignment check"""

    print("=" * 80)
    print("DETAILED COORDINATE SYSTEM CHECK")
    print("=" * 80)

    # Configuration
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
        'rand_flip': False,  # IMPORTANT: No augmentation
        'bot_pct_lim': (0.0, 0.0),
        'Ncams': 6,
    }

    # Load data
    print("\n1. Loading data...")
    dataroot = '/data/SimBEV'
    trainloader, _ = compile_data(
        version='unused',
        dataroot=dataroot,
        data_aug_conf=data_aug_conf,
        grid_conf=grid_conf,
        bsz=1,
        nworkers=0,
        parser_name='segmentationdata'
    )

    # Get multiple samples
    print("\n2. Checking multiple samples...")
    num_samples = 5

    for sample_idx in range(num_samples):
        batch = next(iter(trainloader))
        imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch

        gt = binimgs[0, 0].cpu().numpy()

        # Find GT vehicle locations
        vehicle_indices = np.where(gt > 0.5)
        if len(vehicle_indices[0]) == 0:
            print(f"   Sample {sample_idx}: No vehicles in GT, skipping")
            continue

        rows = vehicle_indices[0]
        cols = vehicle_indices[1]

        # Convert to meters
        dx = grid_conf['xbound'][2]
        x_min = grid_conf['xbound'][0]
        y_min = grid_conf['ybound'][0]

        x_meters = cols * dx + x_min + dx/2
        y_meters = rows * dx + y_min + dx/2

        print(f"\n   Sample {sample_idx}:")
        print(f"     Vehicle pixels: {len(rows)}")
        print(f"     Row range: [{rows.min()}, {rows.max()}]")
        print(f"     Col range: [{cols.min()}, {cols.max()}]")
        print(f"     X (forward) range: [{x_meters.min():.1f}, {x_meters.max():.1f}] m")
        print(f"     Y (left) range: [{y_meters.min():.1f}, {y_meters.max():.1f}] m")
        print(f"     Mean position: X={x_meters.mean():.1f}m, Y={y_meters.mean():.1f}m")

        # Expected: vehicles should be in FRONT (X > 0) and near center (Y ~ 0)
        if x_meters.mean() < 0:
            print(f"     ⚠️  PROBLEM: Vehicles BEHIND ego (X < 0)")
        else:
            print(f"     ✓ Good: Vehicles in front of ego (X > 0)")

        if abs(y_meters.mean()) > 30:
            print(f"     ⚠️  PROBLEM: Vehicles far from center (Y={y_meters.mean():.1f})")
        else:
            print(f"     ✓ Good: Vehicles near center lane")

    # Now check model prediction coordinate system
    print("\n" + "=" * 80)
    print("3. Checking Model Coordinate System")
    print("=" * 80)

    device = torch.device('cuda:0')
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)
    model.eval()

    # Check model's grid parameters
    print(f"\n   Model grid parameters:")
    print(f"     dx (resolution): {model.dx}")
    print(f"     bx (origin): {model.bx}")
    print(f"     nx (size): {model.nx}")

    # Get model prediction
    batch = next(iter(trainloader))
    imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch

    with torch.no_grad():
        preds = model(
            imgs.to(device),
            rots.to(device),
            trans.to(device),
            intrins.to(device),
            post_rots.to(device),
            post_trans.to(device),
        )

    pred = torch.sigmoid(preds[0, 0]).cpu().numpy()
    gt = binimgs[0, 0].cpu().numpy()

    print(f"\n   Prediction shape: {pred.shape}")
    print(f"   GT shape: {gt.shape}")
    print(f"   Shapes match: {pred.shape == gt.shape}")

    # Visualize with grid
    print("\n4. Creating detailed visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    extent = [grid_conf['xbound'][0], grid_conf['xbound'][1],
              grid_conf['ybound'][0], grid_conf['ybound'][1]]

    # Row 1: Original views
    # GT
    im0 = axes[0, 0].imshow(gt, cmap='hot', vmin=0, vmax=1, origin='lower', extent=extent)
    axes[0, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('X (forward, meters)')
    axes[0, 0].set_ylabel('Y (left, meters)')
    axes[0, 0].axhline(y=0, color='white', linewidth=1, linestyle='--', alpha=0.7)
    axes[0, 0].axvline(x=0, color='white', linewidth=1, linestyle='--', alpha=0.7)
    axes[0, 0].plot(0, 0, 'wo', markersize=10)
    axes[0, 0].grid(True, alpha=0.3, color='white')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # Prediction
    im1 = axes[0, 1].imshow(pred, cmap='hot', vmin=0, vmax=1, origin='lower', extent=extent)
    axes[0, 1].set_title('Model Prediction (Random Init)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('X (forward, meters)')
    axes[0, 1].set_ylabel('Y (left, meters)')
    axes[0, 1].axhline(y=0, color='white', linewidth=1, linestyle='--', alpha=0.7)
    axes[0, 1].axvline(x=0, color='white', linewidth=1, linestyle='--', alpha=0.7)
    axes[0, 1].plot(0, 0, 'wo', markersize=10)
    axes[0, 1].grid(True, alpha=0.3, color='white')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Overlay
    overlay = np.zeros((*gt.shape, 3))
    overlay[:, :, 0] = gt
    overlay[:, :, 1] = pred
    axes[0, 2].imshow(overlay, origin='lower', extent=extent)
    axes[0, 2].set_title('Overlay (Red=GT, Green=Pred)', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('X (forward, meters)')
    axes[0, 2].set_ylabel('Y (left, meters)')
    axes[0, 2].axhline(y=0, color='white', linewidth=1, linestyle='--', alpha=0.7)
    axes[0, 2].axvline(x=0, color='white', linewidth=1, linestyle='--', alpha=0.7)
    axes[0, 2].plot(0, 0, 'wo', markersize=10)
    axes[0, 2].grid(True, alpha=0.3, color='white')

    # Row 2: Different potential flips/rotations to check
    # Flip X (left-right)
    gt_fliplr = np.fliplr(gt)
    axes[1, 0].imshow(gt_fliplr, cmap='hot', vmin=0, vmax=1, origin='lower', extent=extent)
    axes[1, 0].set_title('GT Flipped LR (if coords wrong)', fontsize=12)
    axes[1, 0].set_xlabel('X (forward, meters)')
    axes[1, 0].set_ylabel('Y (left, meters)')
    axes[1, 0].grid(True, alpha=0.3)

    # Flip Y (up-down)
    gt_flipud = np.flipud(gt)
    axes[1, 1].imshow(gt_flipud, cmap='hot', vmin=0, vmax=1, origin='lower', extent=extent)
    axes[1, 1].set_title('GT Flipped UD (if coords wrong)', fontsize=12)
    axes[1, 1].set_xlabel('X (forward, meters)')
    axes[1, 1].set_ylabel('Y (left, meters)')
    axes[1, 1].grid(True, alpha=0.3)

    # Transpose
    gt_T = gt.T
    axes[1, 2].imshow(gt_T, cmap='hot', vmin=0, vmax=1, origin='lower', extent=extent)
    axes[1, 2].set_title('GT Transposed (if X/Y swapped)', fontsize=12)
    axes[1, 2].set_xlabel('X (forward, meters)')
    axes[1, 2].set_ylabel('Y (left, meters)')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path('./debug_outputs')
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / 'detailed_coordinate_check.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n   Saved to: {save_path}")
    plt.close()

    # Check index mapping explicitly
    print("\n" + "=" * 80)
    print("5. Explicit Index Mapping Check")
    print("=" * 80)

    print("\n   BEV Grid Layout (LSS expectation):")
    print("     - Array indices: [row, col] = [Y_index, X_index]")
    print("     - Row 0 → Y_min (-50m, rightmost)")
    print("     - Row 199 → Y_max (+50m, leftmost)")
    print("     - Col 0 → X_min (-50m, back)")
    print("     - Col 199 → X_max (+50m, front)")
    print("\n   Ego vehicle at (X=0, Y=0):")
    print("     - Should be at array index [100, 100]")
    print(f"     - GT[100, 100] = {gt[100, 100]:.3f}")

    print("\n   Expected vehicle positions:")
    print("     - Front of car: high column indices (col > 100)")
    print("     - Back of car: low column indices (col < 100)")
    print("     - Left of car: high row indices (row > 100)")
    print("     - Right of car: low row indices (row < 100)")

    # Find actual vehicle positions
    vehicle_indices = np.where(gt > 0.5)
    if len(vehicle_indices[0]) > 0:
        rows = vehicle_indices[0]
        cols = vehicle_indices[1]
        print(f"\n   Actual GT vehicle indices:")
        print(f"     - Row range: [{rows.min()}, {rows.max()}] (center=100)")
        print(f"     - Col range: [{cols.min()}, {cols.max()}] (center=100)")
        print(f"     - Mean: [row={rows.mean():.1f}, col={cols.mean():.1f}]")

        if cols.mean() > 100:
            print("     ✓ CORRECT: Vehicles in FRONT (col > 100)")
        else:
            print("     ⚠️  WRONG: Vehicles in BACK (col < 100) - COORDINATE MISMATCH!")

        if abs(rows.mean() - 100) < 50:
            print("     ✓ CORRECT: Vehicles near center Y")
        else:
            print(f"     ⚠️  WRONG: Vehicles far from center (row={rows.mean():.1f})")

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    vehicle_indices = np.where(gt > 0.5)
    if len(vehicle_indices[0]) > 0:
        cols = vehicle_indices[1]
        if cols.mean() < 100:
            print("\n⚠️  CRITICAL: Coordinate system is STILL WRONG!")
            print("   Vehicles appear in BACK instead of FRONT")
            print("\n   Solutions to try:")
            print("   1. Remove the fliplr() we added (might have made it worse)")
            print("   2. Check if SimBEV uses different axis convention")
            print("   3. Check if we need flipud() instead of fliplr()")
            print("   4. Verify the exact SimBEV coordinate system documentation")
        else:
            print("\n✓ Coordinate system appears CORRECT!")
            print("  Vehicles are positioned in front of ego")
            print("\n  If training still fails, check:")
            print("  - Learning rate (try 1e-4)")
            print("  - Data augmentation (disable flip)")
            print("  - Loss function weights")

    print("=" * 80)


if __name__ == '__main__':
    detailed_check()
