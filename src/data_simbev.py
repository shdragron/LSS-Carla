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
        train/
          scene_name/
            orientation_folder/  (e.g., yaw0pitch0/)
              meta.json          # list of sample dictionaries
              images/            # camera images
              labels/            # BEV segmentation maps
        val/
          ...

    Each sample in meta.json should contain:
      - token: unique sample identifier
      - images: list of image file paths (6 cameras)
      - intrinsics: list of 3x3 camera intrinsic matrices (6 cameras)
      - extrinsics: list of 4x4 extrinsic matrices (ego->camera transform) (6 cameras)
      - label: BEV segmentation map file path
    """

    def __init__(self, dataroot, is_train, data_aug_conf, grid_conf):
        self.dataroot = Path(dataroot)
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.split = 'train' if is_train else 'val'
        self.split_dir = self.dataroot / self.split

        # Load all samples from all scenes
        self.samples = self._load_all_samples()

        # Setup BEV grid parameters
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        print(self)

    def _load_all_samples(self):
        """Load all samples from all scenes in the split directory."""
        all_samples = []

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Iterate through all scene directories
        for scene_dir in sorted(self.split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

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
            raise FileNotFoundError(f"No samples found in {self.split_dir}")

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
        meta_dir = sample['meta_dir']

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
            # Load image
            img_path = meta_dir / image_paths[cam_idx]
            img = Image.open(img_path)

            # Initial transforms
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # Camera intrinsics
            intrin = torch.Tensor(intrinsics_list[cam_idx])

            # Extrinsics: ego->cam, we need cam->ego
            # extrinsic is 4x4 matrix: [R|t; 0 0 0 1]
            extrin_mat = np.array(extrinsics_list[cam_idx])
            R_ego_cam = extrin_mat[:3, :3]
            t_ego_cam = extrin_mat[:3, 3]

            # Invert to get cam->ego
            R_cam_ego = R_ego_cam.T
            t_cam_ego = -R_cam_ego @ t_ego_cam

            rot = torch.Tensor(R_cam_ego)
            tran = torch.Tensor(t_cam_ego)

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
        Load BEV segmentation map.

        Returns:
            binimg: (1, H, W) binary segmentation map
        """
        meta_dir = sample['meta_dir']
        label_path = meta_dir / sample['label']

        # Load label (assuming it's saved as numpy array or image)
        if label_path.suffix == '.npy':
            binimg = np.load(label_path)
        else:
            # Try loading as image
            binimg = np.array(Image.open(label_path))

        # Ensure it's float32 and has shape (1, H, W) or convert it
        if binimg.ndim == 2:
            binimg = binimg[np.newaxis, ...]
        elif binimg.ndim == 3:
            # If it's (H, W, C), take first channel or convert
            if binimg.shape[2] > 1:
                # Assume vehicle class is in a specific channel or reduce
                binimg = np.max(binimg, axis=2, keepdims=False)
            binimg = binimg[np.newaxis, ...]

        return torch.from_numpy(binimg.astype(np.float32))

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
        return (f"SimBEVDataset: {len(self)} samples. "
                f"Split: {'train' if self.is_train else 'val'}. "
                f"Augmentation Conf: {self.data_aug_conf}")


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
