"""
Test script for SimBEV dataloader with LSS model
"""

import torch
from src.data_simbev import compile_data

# Configuration
dataroot = '/path/to/simbev/dataset'  # Update this path

data_aug_conf = {
    'H': 224,           # Original image height
    'W': 480,           # Original image width
    'final_dim': (128, 352),  # Target image size
    'resize_lim': (0.9, 1.1),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (-5.4, 5.4),
    'rand_flip': True,
    'Ncams': 6,
}

grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [4.0, 45.0, 1.0],
}

if __name__ == '__main__':
    print("Loading SimBEV dataset...")

    try:
        trainloader, valloader = compile_data(
            version='unused',  # Not used for SimBEV
            dataroot=dataroot,
            data_aug_conf=data_aug_conf,
            grid_conf=grid_conf,
            bsz=2,
            nworkers=0,  # Use 0 for debugging
            parser_name='segmentationdata'
        )

        print(f"Train loader created: {len(trainloader)} batches")
        print(f"Val loader created: {len(valloader)} batches")

        # Test loading one batch
        print("\nTesting train loader...")
        for batch_idx, batch in enumerate(trainloader):
            imgs, rots, trans, intrins, post_rots, post_trans, binimg = batch

            print(f"\nBatch {batch_idx}:")
            print(f"  imgs shape: {imgs.shape}")  # (B, N, 3, H, W)
            print(f"  rots shape: {rots.shape}")  # (B, N, 3, 3)
            print(f"  trans shape: {trans.shape}")  # (B, N, 3)
            print(f"  intrins shape: {intrins.shape}")  # (B, N, 3, 3)
            print(f"  post_rots shape: {post_rots.shape}")  # (B, N, 3, 3)
            print(f"  post_trans shape: {post_trans.shape}")  # (B, N, 3)
            print(f"  binimg shape: {binimg.shape}")  # (B, 1, H, W)

            # Only test first batch
            break

        print("\n✓ Dataloader test successful!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
