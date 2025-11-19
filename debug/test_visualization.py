"""
Test BEV visualization code
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.models import compile_model
from src.data_simbev import compile_data

def test_visualization():
    """Test the BEV visualization"""

    print("=" * 80)
    print("TESTING BEV VISUALIZATION")
    print("=" * 80)

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
        'rand_flip': False,
        'bot_pct_lim': (0.0, 0.0),
        'Ncams': 6,
    }

    # Load data
    print("\n1. Loading dataset...")
    trainloader, _ = compile_data(
        version='unused',
        dataroot=dataroot,
        data_aug_conf=data_aug_conf,
        grid_conf=grid_conf,
        bsz=2,
        nworkers=0,
        parser_name='segmentationdata'
    )

    # Create model
    print("\n2. Creating model...")
    device = torch.device('cuda:0')
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)
    model.eval()

    # Get one batch
    print("\n3. Getting batch and running inference...")
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

    print(f"   Prediction shape: {preds.shape}")
    print(f"   GT shape: {binimgs.shape}")

    # Create visualization
    print("\n4. Creating visualization...")
    pred_vis = torch.sigmoid(preds[0, 0]).detach().cpu().numpy()
    gt_vis = binimgs[0, 0].cpu().numpy()

    print(f"   Pred range: [{pred_vis.min():.4f}, {pred_vis.max():.4f}]")
    print(f"   GT range: [{gt_vis.min():.4f}, {gt_vis.max():.4f}]")

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground Truth
    axes[0].imshow(gt_vis, cmap='gray', vmin=0, vmax=1, origin='lower')
    axes[0].set_title('Ground Truth BEV')
    axes[0].set_xlabel('X (meters)')
    axes[0].set_ylabel('Y (meters)')
    axes[0].grid(False)

    # Prediction
    axes[1].imshow(pred_vis, cmap='gray', vmin=0, vmax=1, origin='lower')
    axes[1].set_title('Prediction BEV')
    axes[1].set_xlabel('X (meters)')
    axes[1].set_ylabel('Y (meters)')
    axes[1].grid(False)

    # Overlay
    overlay = np.zeros((gt_vis.shape[0], gt_vis.shape[1], 3))
    overlay[:, :, 0] = gt_vis
    overlay[:, :, 1] = pred_vis
    axes[2].imshow(overlay, origin='lower')
    axes[2].set_title('Overlay (GT=Red, Pred=Green, Match=Yellow)')
    axes[2].set_xlabel('X (meters)')
    axes[2].set_ylabel('Y (meters)')
    axes[2].grid(False)

    plt.tight_layout()

    # Save
    output_dir = Path('./debug_outputs')
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / 'test_bev_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n5. Saved visualization to: {save_path}")

    plt.close(fig)

    print("\n" + "=" * 80)
    print("✓ Visualization test completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    test_visualization()
