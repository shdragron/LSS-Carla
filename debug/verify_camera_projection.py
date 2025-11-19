"""
Verify camera extrinsics by projecting 3D points onto images
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image

def project_points_to_image(points_ego, K, R_cam_ego, t_cam_ego):
    """
    Project 3D points from ego frame to image

    Args:
        points_ego: (N, 3) points in ego coordinate system
        K: (3, 3) camera intrinsics
        R_cam_ego: (3, 3) rotation from ego to cam
        t_cam_ego: (3,) translation from ego to cam

    Returns:
        (N, 2) pixel coordinates
    """
    # Transform to camera frame
    # If R, t represent cam->ego, need to invert:
    # P_cam = R^T @ (P_ego - t)
    # If R, t represent ego->cam:
    # P_cam = R @ P_ego + t

    # Try ego->cam (no inversion)
    points_cam = (R_cam_ego @ points_ego.T).T + t_cam_ego

    # Project to image
    points_proj = (K @ points_cam.T).T  # (N, 3)

    # Normalize by depth
    valid = points_proj[:, 2] > 0
    pixels = np.zeros((len(points_proj), 2))
    pixels[valid, 0] = points_proj[valid, 0] / points_proj[valid, 2]
    pixels[valid, 1] = points_proj[valid, 1] / points_proj[valid, 2]

    return pixels, valid


def verify_projection():
    """Check if camera projection makes sense"""

    # Load sample data
    with open('/data/SimBEV/SimBEV_cvt_label/scene_0020/yaw0pitch0/meta.json', 'r') as f:
        meta = json.load(f)

    sample = meta[0]

    # Get front camera data
    cam_idx = 1  # Front camera
    img_path = Path('/data/SimBEV') / sample['images'][cam_idx]
    img = Image.open(img_path)

    K = np.array(sample['intrinsics'][cam_idx])
    extrin = np.array(sample['extrinsics'][cam_idx])
    R = extrin[:3, :3]
    t = extrin[:3, 3]

    print("=" * 80)
    print("CAMERA PROJECTION VERIFICATION")
    print("=" * 80)
    print(f"\nCamera: {sample['cam_channels'][cam_idx]}")
    print(f"Image shape: {img.size}")
    print(f"\nIntrinsics:")
    print(f"  fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    print(f"  cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    # Define test points in ego frame
    # Points in front of the car at ground level
    test_points = np.array([
        [5.0, 0.0, 0.0],   # 5m in front, center
        [10.0, 0.0, 0.0],  # 10m in front, center
        [5.0, 2.0, 0.0],   # 5m front, 2m left
        [5.0, -2.0, 0.0],  # 5m front, 2m right
        [0.0, 0.0, 0.0],   # ego position
    ])

    print("\n" + "=" * 80)
    print("TEST 1: Using extrinsics AS IS (assuming ego->cam)")
    print("=" * 80)

    pixels_v1, valid_v1 = project_points_to_image(test_points, K, R, t)

    print("\n3D Point (ego frame) -> 2D Pixel:")
    for i, (pt, px, v) in enumerate(zip(test_points, pixels_v1, valid_v1)):
        if v:
            in_image = (0 <= px[0] < img.size[0]) and (0 <= px[1] < img.size[1])
            status = "✓ in image" if in_image else "✗ out of bounds"
            print(f"  [{pt[0]:5.1f}, {pt[1]:5.1f}, {pt[2]:5.1f}] -> [{px[0]:7.1f}, {px[1]:7.1f}]  {status}")
        else:
            print(f"  [{pt[0]:5.1f}, {pt[1]:5.1f}, {pt[2]:5.1f}] -> behind camera")

    print("\n" + "=" * 80)
    print("TEST 2: Using INVERTED extrinsics (assuming cam->ego)")
    print("=" * 80)

    R_inv = R.T
    t_inv = -R_inv @ t
    pixels_v2, valid_v2 = project_points_to_image(test_points, K, R_inv, t_inv)

    print("\n3D Point (ego frame) -> 2D Pixel:")
    for i, (pt, px, v) in enumerate(zip(test_points, pixels_v2, valid_v2)):
        if v:
            in_image = (0 <= px[0] < img.size[0]) and (0 <= px[1] < img.size[1])
            status = "✓ in image" if in_image else "✗ out of bounds"
            print(f"  [{pt[0]:5.1f}, {pt[1]:5.1f}, {pt[2]:5.1f}] -> [{px[0]:7.1f}, {px[1]:7.1f}]  {status}")
        else:
            print(f"  [{pt[0]:5.1f}, {pt[1]:5.1f}, {pt[2]:5.1f}] -> behind camera")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("\nFor a FRONT camera:")
    print("  - Points in FRONT of car (x > 0) should project IN FRONT (visible)")
    print("  - Ego position (0,0,0) should be BEHIND camera")
    print("  - Projections should land roughly in image center for straight-ahead points")
    print(f"  - Image center: ({img.size[0]/2:.0f}, {img.size[1]/2:.0f})")

    # Count valid projections
    valid_count_v1 = valid_v1.sum()
    valid_count_v2 = valid_v2.sum()

    # Check if projections make sense for front camera
    front_points_visible_v1 = valid_v1[:4].sum()  # First 4 are in front
    ego_visible_v1 = valid_v1[4]

    front_points_visible_v2 = valid_v2[:4].sum()
    ego_visible_v2 = valid_v2[4]

    print(f"\nTEST 1 (as-is): {front_points_visible_v1}/4 front points visible, ego visible: {ego_visible_v1}")
    print(f"TEST 2 (inverted): {front_points_visible_v2}/4 front points visible, ego visible: {ego_visible_v2}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if front_points_visible_v1 > front_points_visible_v2 and not ego_visible_v1:
        print("✓ TEST 1 (as-is) is CORRECT")
        print("  Extrinsics are in ego->cam format")
        print("  NO inversion needed in data loader")
    elif front_points_visible_v2 > front_points_visible_v1 and not ego_visible_v2:
        print("✓ TEST 2 (inverted) is CORRECT")
        print("  Extrinsics are in cam->ego format")
        print("  INVERSION needed in data loader (current code is correct)")
    else:
        print("⚠️  Results inconclusive - need manual inspection")

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.imshow(img)
    ax1.scatter(pixels_v1[valid_v1, 0], pixels_v1[valid_v1, 1], c='red', s=100, marker='x', linewidths=3)
    ax1.set_title('TEST 1: As-is (ego->cam)', fontsize=14)
    ax1.set_xlim(0, img.size[0])
    ax1.set_ylim(img.size[1], 0)

    ax2.imshow(img)
    ax2.scatter(pixels_v2[valid_v2, 0], pixels_v2[valid_v2, 1], c='blue', s=100, marker='x', linewidths=3)
    ax2.set_title('TEST 2: Inverted (cam->ego)', fontsize=14)
    ax2.set_xlim(0, img.size[0])
    ax2.set_ylim(img.size[1], 0)

    plt.tight_layout()
    save_path = Path('./debug_outputs/camera_projection_test.png')
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")


if __name__ == '__main__':
    verify_projection()
