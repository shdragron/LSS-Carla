"""
Verify that BEV coordinate systems match between GT and model prediction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models import compile_model
from src.data_simbev import compile_data
from src.tools import gen_dx_bx

def verify_coordinates():
    """Check if GT and prediction use the same coordinate system"""

    print("=" * 80)
    print("BEV COORDINATE SYSTEM VERIFICATION")
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
        'rand_flip': False,
        'bot_pct_lim': (0.0, 0.0),
        'Ncams': 6,
    }

    # Generate grid parameters
    dx, bx, nx = gen_dx_bx(
        grid_conf['xbound'],
        grid_conf['ybound'],
        grid_conf['zbound']
    )

    print("\n1. Grid Configuration:")
    print(f"   xbound: {grid_conf['xbound']} -> nx={nx[0]}")
    print(f"   ybound: {grid_conf['ybound']} -> ny={nx[1]}")
    print(f"   dx (resolution): {dx.numpy()}")
    print(f"   bx (origin): {bx.numpy()}")
    print(f"   nx (grid size): {nx.numpy()}")

    # Interpretation
    print("\n2. Grid Interpretation:")
    print(f"   X range: [{grid_conf['xbound'][0]}, {grid_conf['xbound'][1]}] meters")
    print(f"   Y range: [{grid_conf['ybound'][0]}, {grid_conf['ybound'][1]}] meters")
    print(f"   Resolution: {grid_conf['xbound'][2]} meters/pixel")
    print(f"   Grid size: {int(nx[0])} x {int(nx[1])}")
    print(f"   Origin (bx): x={bx[0]:.2f}m, y={bx[1]:.2f}m (center of first pixel)")

    # Load data to check GT
    print("\n3. Loading GT data...")
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

    # Get one sample
    batch = next(iter(trainloader))
    imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch

    print(f"   GT shape: {binimgs.shape}")
    print(f"   GT range: [{binimgs.min():.3f}, {binimgs.max():.3f}]")

    # Create model and get prediction
    print("\n4. Getting model prediction...")
    device = torch.device('cuda:0')
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)
    model.eval()

    with torch.no_grad():
        preds = model(
            imgs.to(device),
            rots.to(device),
            trans.to(device),
            intrins.to(device),
            post_rots.to(device),
            post_trans.to(device),
        )

    print(f"   Prediction shape: {preds.shape}")
    pred_sigmoid = torch.sigmoid(preds[0, 0]).cpu().numpy()
    gt = binimgs[0, 0].cpu().numpy()

    print(f"   Shapes match: {pred_sigmoid.shape == gt.shape}")

    # Visualize with coordinate annotations
    print("\n5. Creating annotated visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Create coordinate grid for annotation
    extent = [grid_conf['xbound'][0], grid_conf['xbound'][1],
              grid_conf['ybound'][0], grid_conf['ybound'][1]]

    # GT
    im0 = axes[0].imshow(gt, cmap='gray', vmin=0, vmax=1, origin='lower', extent=extent)
    axes[0].set_title('Ground Truth BEV\n(SimBEV coordinate system)', fontsize=12)
    axes[0].set_xlabel('X (meters, forward)')
    axes[0].set_ylabel('Y (meters, left)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='red', linewidth=1, linestyle='--', alpha=0.5, label='Y=0 (ego X-axis)')
    axes[0].axvline(x=0, color='blue', linewidth=1, linestyle='--', alpha=0.5, label='X=0 (ego Y-axis)')
    axes[0].plot(0, 0, 'ro', markersize=10, label='Ego (0,0)')
    axes[0].legend(fontsize=8)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Prediction
    im1 = axes[1].imshow(pred_sigmoid, cmap='gray', vmin=0, vmax=1, origin='lower', extent=extent)
    axes[1].set_title('Model Prediction BEV\n(LSS coordinate system)', fontsize=12)
    axes[1].set_xlabel('X (meters, forward)')
    axes[1].set_ylabel('Y (meters, left)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='red', linewidth=1, linestyle='--', alpha=0.5)
    axes[1].axvline(x=0, color='blue', linewidth=1, linestyle='--', alpha=0.5)
    axes[1].plot(0, 0, 'ro', markersize=10)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Overlay
    overlay = np.zeros((gt.shape[0], gt.shape[1], 3))
    overlay[:, :, 0] = gt
    overlay[:, :, 1] = pred_sigmoid
    axes[2].imshow(overlay, origin='lower', extent=extent)
    axes[2].set_title('Overlay\n(Red=GT, Green=Pred, Yellow=Match)', fontsize=12)
    axes[2].set_xlabel('X (meters, forward)')
    axes[2].set_ylabel('Y (meters, left)')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='white', linewidth=1, linestyle='--', alpha=0.5)
    axes[2].axvline(x=0, color='white', linewidth=1, linestyle='--', alpha=0.5)
    axes[2].plot(0, 0, 'wo', markersize=10)

    plt.tight_layout()

    # Save
    output_dir = Path('./debug_outputs')
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / 'coordinate_system_verification.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n6. Saved to: {save_path}")

    plt.close(fig)

    # Check specific locations
    print("\n7. Checking specific locations:")
    print(f"   GT[0, 0] (bottom-left corner, x=-50, y=-50): {gt[0, 0]:.3f}")
    print(f"   GT[-1, -1] (top-right corner, x=50, y=50): {gt[-1, -1]:.3f}")
    print(f"   GT[100, 100] (center, x=0, y=0): {gt[100, 100]:.3f}")

    # Find vehicle locations in GT
    vehicle_indices = np.where(gt > 0.5)
    if len(vehicle_indices[0]) > 0:
        print(f"\n8. Vehicle locations in GT:")
        print(f"   Number of vehicle pixels: {len(vehicle_indices[0])}")
        # Convert indices to meters
        y_indices = vehicle_indices[0]  # Row indices
        x_indices = vehicle_indices[1]  # Column indices
        x_meters = x_indices * dx[0].item() + bx[0].item() - dx[0].item()/2
        y_meters = y_indices * dx[1].item() + bx[1].item() - dx[1].item()/2
        print(f"   X range: [{x_meters.min():.1f}, {x_meters.max():.1f}] meters")
        print(f"   Y range: [{y_meters.min():.1f}, {y_meters.max():.1f}] meters")
        print(f"   Mean position: X={x_meters.mean():.1f}m, Y={y_meters.mean():.1f}m")

        # Check if vehicles are in expected location (front of car)
        if x_meters.mean() > 0:
            print("   ✓ Vehicles are in FRONT of ego (positive X) - CORRECT")
        else:
            print("   ⚠️  Vehicles are BEHIND ego (negative X) - CHECK COORDINATE SYSTEM!")

        if abs(y_meters.mean()) < 20:
            print("   ✓ Vehicles are near center lane (Y~0) - CORRECT")
        else:
            print(f"   ⚠️  Vehicles are far from center (Y={y_meters.mean():.1f}m) - CHECK!")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("\nLSS Coordinate System:")
    print("  - X: Forward (front of car)")
    print("  - Y: Left (left side of car)")
    print("  - Origin (0,0): Ego vehicle center")
    print("  - Grid indexing: [Y_index, X_index] (row, col)")
    print("    - Row 0 = Y_min (left)")
    print("    - Row 199 = Y_max (right)")
    print("    - Col 0 = X_min (back)")
    print("    - Col 199 = X_max (front)")
    print("\nSimBEV should match this convention!")
    print("\nIf vehicles appear in wrong location, coordinate transform needed.")
    print("=" * 80)


if __name__ == '__main__':
    verify_coordinates()
