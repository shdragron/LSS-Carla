"""
SimBEV Dataset for LSS (Lift-Splat-Shoot)
Adapted from CVT SimBEV dataloader structure
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image

from .tools import img_transform, normalize_img, gen_dx_bx


CAMERA_ORDER = [
    'front_left', 'front', 'front_right',
    'back_left', 'back', 'back_right'
]


class SimBEVDataset(torch.utils.data.Dataset):
    """
    SimBEV dataset for LSS model.

    Directory structure expected:
      dataroot/
        SimBEV_cvt_label/
          scene_0000/
            yaw0pitch0/
              meta.json          # list of sample dictionaries
              bev_*.npz          # BEV segmentation maps
              aux_*.npz          # auxiliary data
              visibility_*.png   # visibility masks
          scene_0001/
            ...
        sweeps/
          RGB-CAM_FRONT/
            *.jpg              # camera images
          RGB-CAM_FRONT_LEFT/
            *.jpg
          ...

    Each sample in meta.json should contain:
      - token: unique sample identifier
      - images: list of image file paths relative to dataroot (6 cameras)
      - intrinsics: list of 3x3 camera intrinsic matrices (6 cameras)
      - extrinsics: list of 4x4 extrinsic matrices (ego->camera transform) (6 cameras)
      - bev: BEV .npz file path relative to meta_dir
    """

    def __init__(self, dataroot, is_train, data_aug_conf, grid_conf):
        self.dataroot = Path(dataroot)
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        # Load all samples from all scenes
        self.samples = self._load_all_samples()

        # Setup BEV grid parameters
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        print(self)

    def _load_all_samples(self):
        """Load all samples from all scenes in the label directory."""
        all_samples = []

        # Labels are in SimBEV_cvt_label directory
        labels_dir = self.dataroot / 'SimBEV_cvt_label'

        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        # Get all scene directories
        scene_dirs = sorted([d for d in labels_dir.iterdir() if d.is_dir() and d.name.startswith('scene_')])

        if not scene_dirs:
            raise FileNotFoundError(f"No scene directories found in {labels_dir}")

        # Split scenes into train/val (80/20 split)
        num_scenes = len(scene_dirs)
        train_split = int(0.8 * num_scenes)

        if self.is_train:
            selected_scenes = scene_dirs[:train_split]
        else:
            selected_scenes = scene_dirs[train_split:]

        # Iterate through selected scene directories
        for scene_dir in selected_scenes:
            # Look for meta.json files (could be in orientation subdirectories)
            meta_files = list(scene_dir.rglob('meta.json'))

            for meta_path in meta_files:
                with open(meta_path, 'r') as f:
                    meta_samples = json.load(f)

                # Add scene and base directory information to each sample
                for sample in meta_samples:
                    sample['scene_dir'] = scene_dir
                    sample['meta_dir'] = meta_path.parent
                    all_samples.append(sample)

        if not all_samples:
            split_name = 'train' if self.is_train else 'val'
            raise FileNotFoundError(f"No samples found for {split_name} split in {labels_dir}")

        return all_samples

    def sample_augmentation(self):
        """Sample random augmentation parameters."""
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']

        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, sample, cam_indices):
        """
        Load and process image data for selected cameras.

        Returns:
            imgs: (N, 3, H, W) normalized images
            rots: (N, 3, 3) camera->ego rotation matrices
            trans: (N, 3) camera->ego translation vectors
            intrins: (N, 3, 3) camera intrinsic matrices
            post_rots: (N, 3, 3) post-processing rotation matrices
            post_trans: (N, 3) post-processing translation vectors
        """
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        # Load intrinsics and extrinsics
        intrinsics_list = sample['intrinsics']  # List of 3x3 matrices
        extrinsics_list = sample['extrinsics']  # List of 4x4 matrices (ego->cam)
        image_paths = sample['images']  # List of image paths

        for cam_idx in cam_indices:
            # Load image (path is relative to dataroot)
            img_path = self.dataroot / image_paths[cam_idx]
            img = Image.open(img_path)

            # Initial transforms
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # Camera intrinsics
            intrin = torch.Tensor(intrinsics_list[cam_idx])

            # Extrinsics: SimBEV provides ego->cam transformation
            # LSS uses it directly - see models.py:182-184
            # The naming is confusing but LSS actually uses ego->cam format
            extrin_mat = np.array(extrinsics_list[cam_idx])
            rot = torch.Tensor(extrin_mat[:3, :3])
            tran = torch.Tensor(extrin_mat[:3, 3])

            # Apply data augmentation
            resize, resize_dims, crop, flip, rotate_angle = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(
                img, post_rot, post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate_angle,
            )

            # Convert to 3x3 matrices
            post_tran_3 = torch.zeros(3)
            post_rot_3 = torch.eye(3)
            post_tran_3[:2] = post_tran2
            post_rot_3[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot_3)
            post_trans.append(post_tran_3)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_binimg(self, sample):
        """
        Load BEV segmentation map from .npz file.

        Returns:
            binimg: (1, H, W) binary segmentation map (vehicle class only)
        """
        meta_dir = sample['meta_dir']
        bev_path = meta_dir / sample['bev']

        # Load BEV from .npz file
        bev_data = np.load(bev_path)
        bev = bev_data['bev']  # Shape: (8, 200, 200) - 8 classes

        # Vehicle classes are indices 1, 2, 3 (different vehicle types)
        # Merge them into a single binary vehicle segmentation
        vehicle_mask = ((bev[1] > 0) | (bev[2] > 0) | (bev[3] > 0)).astype(np.float32)

        # IMPORTANT: SimBEV coordinate system is different from LSS
        # SimBEV: low column index = front of car
        # LSS: high column index = front of car
        # Solution: Flip left-right (along X-axis)
        vehicle_mask = np.fliplr(vehicle_mask).copy()  # .copy() to avoid negative stride

        binimg = vehicle_mask[np.newaxis, :, :]  # Shape: (1, 200, 200)

        return torch.from_numpy(binimg)

    def choose_cams(self):
        """Choose which cameras to use."""
        all_cams = list(range(len(CAMERA_ORDER)))

        if self.is_train and 'Ncams' in self.data_aug_conf:
            Ncams = self.data_aug_conf['Ncams']
            if Ncams < len(CAMERA_ORDER):
                cams = np.random.choice(all_cams, Ncams, replace=False)
                return sorted(cams.tolist())

        return all_cams

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        split_name = 'train' if self.is_train else 'val'
        return (f"SimBEVDataset ({split_name}): {len(self)} samples")


class VizData(SimBEVDataset):
    """Dataset for visualization (includes lidar data)."""

    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)

    def get_lidar_data(self, sample):
        """
        Return empty lidar data (SimBEV doesn't include raw lidar points).

        Returns:
            lidar: (3, 0) empty tensor
        """
        return torch.empty(3, 0)

    def __getitem__(self, index):
        sample = self.samples[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(sample, cams)
        lidar_data = self.get_lidar_data(sample)
        binimg = self.get_binimg(sample)

        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(SimBEVDataset):
    """Dataset for training/validation (no lidar data)."""

    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(sample, cams)
        binimg = self.get_binimg(sample)

        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


def worker_rnd_init(x):
    """Initialize random seed for dataloader workers."""
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz, nworkers, parser_name):
    """
    Create train and val dataloaders.

    Args:
        version: not used for SimBEV (kept for API compatibility)
        dataroot: path to SimBEV dataset root directory
        data_aug_conf: data augmentation configuration dict
        grid_conf: BEV grid configuration dict
        bsz: batch size
        nworkers: number of dataloader workers
        parser_name: 'vizdata' or 'segmentationdata'

    Returns:
        trainloader, valloader: PyTorch DataLoaders
    """
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]

    traindata = parser(dataroot, is_train=True, data_aug_conf=data_aug_conf, grid_conf=grid_conf)
    valdata = parser(dataroot, is_train=False, data_aug_conf=data_aug_conf, grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(
        traindata,
        batch_size=bsz,
        shuffle=True,
        num_workers=nworkers,
        drop_last=True,
        worker_init_fn=worker_rnd_init
    )
    valloader = torch.utils.data.DataLoader(
        valdata,
        batch_size=bsz,
        shuffle=False,
        num_workers=nworkers
    )

    return trainloader, valloader
