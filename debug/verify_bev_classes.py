"""
Verify which BEV class corresponds to vehicles by visualizing all classes
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_all_bev_classes():
    """Visualize all 8 BEV classes to identify which one is vehicles"""

    bev_path = '/data/SimBEV/SimBEV_cvt_label/scene_0020/yaw0pitch0/bev_scene_0020_000000.npz'
    bev_data = np.load(bev_path)
    bev = bev_data['bev']  # Shape: (8, 200, 200)

    print("=" * 80)
    print("BEV CLASS VISUALIZATION")
    print("=" * 80)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    class_names = [
        'Class 0 (drivable area?)',
        'Class 1 (vehicles?)',
        'Class 2 (lanes?)',
        'Class 3',
        'Class 4',
        'Class 5',
        'Class 6',
        'Class 7',
    ]

    for i in range(8):
        class_data = bev[i]
        num_positive = (class_data > 0).sum()

        axes[i].imshow(class_data, cmap='hot', origin='lower', vmin=0, vmax=1)
        axes[i].set_title(f'{class_names[i]}\n{num_positive} pixels', fontsize=10)
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].grid(False)

        print(f"{class_names[i]:25s}: {num_positive:6d} positive pixels")

    plt.suptitle('All BEV Classes - Find the Vehicle Class!', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = Path('./debug_outputs/bev_all_classes.png')
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {save_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("Based on typical SimBEV/nuScenes class ordering:")
    print("  - Class 0: Usually drivable area (large coverage)")
    print("  - Class 1: Usually vehicles (moderate coverage)")
    print("  - Class 2: Usually lanes/dividers (small coverage)")
    print("")
    print("⚠️  IMPORTANT: Check the visualization to confirm which class represents vehicles!")
    print("   If Class 0 is too large (drivable area), you should use Class 1 instead.")
    print("   Modify data_simbev.py line 235 accordingly.")

    # Check multiple samples for consistency
    print("\n" + "=" * 80)
    print("CHECKING CONSISTENCY ACROSS SAMPLES")
    print("=" * 80)

    scene_dir = Path('/data/SimBEV/SimBEV_cvt_label/scene_0020/yaw0pitch0')
    bev_files = sorted(list(scene_dir.glob('bev_*.npz')))[:5]

    class_stats = [[] for _ in range(8)]

    for bev_file in bev_files:
        bev_data = np.load(bev_file)
        bev = bev_data['bev']
        for i in range(8):
            class_stats[i].append((bev[i] > 0).sum())

    print("Average positive pixels across 5 samples:")
    for i in range(8):
        avg = np.mean(class_stats[i])
        std = np.std(class_stats[i])
        print(f"  Class {i}: {avg:7.1f} ± {std:6.1f} pixels")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Heuristic: vehicles typically have 100-10000 pixels
    # Drivable area typically has > 10000 pixels
    avg_class_0 = np.mean(class_stats[0])
    avg_class_1 = np.mean(class_stats[1])

    if avg_class_0 > 10000 and 100 < avg_class_1 < 10000:
        print("⚠️  Class 0 appears to be DRIVABLE AREA (too large)")
        print("⚠️  Class 1 appears to be VEHICLES (appropriate size)")
        print("")
        print("ACTION REQUIRED:")
        print("  Change line 235 in src/data_simbev.py from:")
        print("    binimg = bev[0:1, :, :]")
        print("  to:")
        print("    binimg = bev[1:2, :, :]")
    elif 100 < avg_class_0 < 10000:
        print("✓ Class 0 appears to be VEHICLES (appropriate size)")
        print("  Current code is correct!")
    else:
        print("⚠️  Inconclusive - please check the visualization manually")


if __name__ == '__main__':
    visualize_all_bev_classes()
