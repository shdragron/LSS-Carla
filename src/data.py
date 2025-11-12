"""
SimBEV (file-based) Dataset for LSS-style loaders
- Reads SimBEV exported frames instead of nuScenes API.
- Keeps return signatures compatible with the original LSS dataset classes.
"""

import os
import json
from glob import glob
from pathlib import Path
from typing import Dict, Any, List
import torch.nn.functional as F

import torch
import numpy as np

# LSS code often imports these utilities; here we only need gen_dx_bx if grid_conf is used elsewhere.
from .tools import gen_dx_bx


class SimBEV(torch.utils.data.Dataset):
    """
    File-based SimBEV dataset.

    Directory layout (example):
      data_dir/
        train_samples.json
        val_samples.json
        train/
          scene-xxx/123456789012345_abcdef12/
            images.pt            # (6,3,H,W) float [0..1]
            intrinsics.pt        # (6,3,3)
            extrinsics.pt        # (6,4,4)  (assumed: ego -> camera)
            bev.npy              # (1,Hbev,Wbev) float
            center_heatmap.npy   # (1,Hbev,Wbev) float
            visibility.npy       # (Hbev,Wbev)   uint8
            meta.json            # dict: { "view": 3x3, "pose": 4x4 (T_global_ego), "cam_idx": [0..5], ...}
        val/
          ...

    Notes:
      - We convert extrinsics (ego->cam) to LSS' (sensor->ego == cam->ego) by inverting.
      - No on-the-fly augmentation: post_rots = I(3), post_trans = 0(3).
    """

    def __init__(self, data_dir: str, is_train: bool, data_aug_conf: Dict[str, Any], grid_conf: Dict[str, Any]):
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.split = 'train' if is_train else 'val'
        self.sample_infos = self._load_split_index(self.split)

        # optional: for BEV grid params, if others depend on them
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        # expected camera count / selection config
        self.all_cam_indices = None  # will read per-frame from meta['cam_idx']

        print(self)

    # -----------------------------
    # Split index / IO
    # -----------------------------
    def _load_split_index(self, split: str) -> List[Dict[str, Any]]:
        # 1) data_dir/{split}_samples.json
        p = self.data_dir / f"{split}_samples.json"
        # 2) fallback: data_dir/{split}/{split}_samples.json
        if not p.exists():
            p = self.data_dir / split / f"{split}_samples.json"
        if not p.exists():
            raise FileNotFoundError(
                f"Cannot find samples file: tried\n"
                f"  • {self.data_dir / f'{split}_samples.json'}\n"
                f"  • {self.data_dir / split / f'{split}_samples.json'}"
            )
        with open(p, 'r') as f:
            sample_infos = json.load(f)
        return sample_infos

    def _frame_dir(self, info: Dict[str, Any]) -> Path:
        # info['folder'] is relative to data_dir/split/
        return self.data_dir / self.split / info['folder']

    # -----------------------------
    # LSS-style API
    # -----------------------------
    def choose_cams(self, cam_idx_list: List[int]) -> List[int]:
        """Optionally subsample cameras like original LSS."""
        cams = cam_idx_list
        if self.is_train and 'Ncams' in self.data_aug_conf and self.data_aug_conf['Ncams'] < len(cam_idx_list):
            # Random subset without replacement
            N = self.data_aug_conf['Ncams']
            # keep ordering reproducible if needed: np.random.choice
            cams = sorted(np.random.choice(cam_idx_list, N, replace=False).tolist())
        return cams

    def _load_frame_tensors(self, frame_dir: Path):
        images = torch.load(frame_dir / "images.pt")          # (Cams,3,H,W) float [0..1]
        intrinsics = torch.load(frame_dir / "intrinsics.pt")  # (Cams,3,3)
        extrinsics = torch.load(frame_dir / "extrinsics.pt")  # (Cams,4,4)  ego->cam
        bev = torch.from_numpy(np.load(frame_dir / "bev.npy")).float()  # (1,Hbev,Wbev)
        # Optional (unused in SegmentationData): center/visibility available
        # center = torch.from_numpy(np.load(frame_dir / "center_heatmap.npy")).float()
        # visibility = torch.from_numpy(np.load(frame_dir / "visibility.npy")).long()

        with open(frame_dir / "meta.json", "r") as f:
            meta = json.load(f)

        cam_idx = meta.get("cam_idx", list(range(images.shape[0])))  # e.g., [0..5]

        return images, intrinsics, extrinsics, bev, meta, cam_idx

    def get_image_data(self, info: Dict[str, Any], cam_indices: List[int]):
        """
        Returns (imgs, rots, trans, intrins, post_rots, post_trans)
        - imgs:        (N,3,H,W) float
        - rots:        (N,3,3)   rotation (sensor->ego == cam->ego)
        - trans:       (N,3)     translation (sensor->ego)
        - intrins:     (N,3,3)
        - post_rots:   (N,3,3)   identity (no aug)
        - post_trans:  (N,3)     zeros (no aug)
        """
        frame_dir = self._frame_dir(info)
        images, intrinsics, extrinsics, _, meta, cam_idx_all = self._load_frame_tensors(frame_dir)

        # choose/reorder
        sel = torch.tensor(cam_indices, dtype=torch.long)
        imgs = images.index_select(0, sel).contiguous()
        intrins = intrinsics.index_select(0, sel).contiguous()
        extri = extrinsics.index_select(0, sel).contiguous()  # ego->cam

        # Convert to LSS' rots/trans: sensor->ego (i.e., cam->ego)
        # For each 4x4 T_ec (ego->cam), invert to get T_ce (cam->ego):
        # T_ce = inv(T_ec) -> R_ce, t_ce
        rots = []
        trans = []
        for i in range(extri.shape[0]):
            T_ec = extri[i].cpu().numpy()
            R = T_ec[:3, :3]
            t = T_ec[:3, 3]
            # inverse of [R|t] is [R^T | -R^T t]
            R_inv = R.T
            t_inv = -R_inv @ t
            rots.append(torch.from_numpy(R_inv).float())
            trans.append(torch.from_numpy(t_inv).float())
        rots = torch.stack(rots, dim=0)          # (N,3,3)
        trans = torch.stack(trans, dim=0)        # (N,3)

        # No online augmentation: identity transforms
        post_rots = torch.eye(3).unsqueeze(0).repeat(len(cam_indices), 1, 1)
        post_trans = torch.zeros(len(cam_indices), 3)

        return imgs, rots, trans, intrins, post_rots, post_trans

    def get_lidar_data(self, info: Dict[str, Any], nsweeps: int):
        """
        SimBEV 포맷 기본 구성에는 원시 LiDAR 포인트가 없다고 가정.
        시각화 경로만 유지하기 위해 빈 텐서(3,0) 반환.
        필요하면 visibility.npy를 읽어 의사 포인트를 만들도록 확장 가능.
        """
        return torch.empty(3, 0)

    def get_binimg(self, info: Dict[str, Any]):
        frame_dir = self._frame_dir(info)
        bev = np.load(frame_dir / "bev.npy").astype(np.float32)  # (8,Hbev,Wbev)
        bev = np.maximum.reduce(bev[[1, 2, 3]])                  # (H,W)
        bev = bev[None, ...].astype(np.float32)
        return torch.from_numpy(bev)

    # -----------------------------
    # PyTorch Dataset protocol
    # -----------------------------
    def __len__(self):
        return len(self.sample_infos)

    def __str__(self):
        return (f"SimBEV (file-based): {len(self)} samples. "
                f"Split: {'train' if self.is_train else 'val'}. "
                f"Augmentation: none (post_rots=I, post_trans=0)")

    # -----------------------------
    # Children classes (like LSS)
    # -----------------------------


class VizData(SimBEV):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        info = self.sample_infos[index]
        # per-frame cam list from meta
        frame_dir = self._frame_dir(info)
        with open(frame_dir / "meta.json", "r") as f:
            meta = json.load(f)
        all_cams: List[int] = meta.get("cam_idx", [0, 1, 2, 3, 4, 5])
        cams = self.choose_cams(all_cams)

        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(info, cams)
        lidar_data = self.get_lidar_data(info, nsweeps=3)  # empty (3,0)
        binimg = self.get_binimg(info)

        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(SimBEV):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
        self.target_hw = (128, 352)  # (H, W)

    def _resize_and_adjust(self, imgs, intrins, post_rots, post_trans, target_hw):
        Hn, Wn = target_hw
        H0, W0 = imgs.shape[-2:]
        sy = Hn / H0
        sx = Wn / W0

        # 1) 이미지 리사이즈 (bilinear, align_corners=False 권장)
        imgs = F.interpolate(imgs, size=(Hn, Wn), mode='bilinear', align_corners=False)

        # 2) intrinsics 보정 (fx, fy, cx, cy 스케일)
        intrins = intrins.clone()
        intrins[:, 0, 0] *= sx  # fx
        intrins[:, 1, 1] *= sy  # fy
        intrins[:, 0, 2] *= sx  # cx
        intrins[:, 1, 2] *= sy  # cy

        # 3) post_rots / post_trans에 스케일 합성 (3x3, 3-vector라고 가정)
        S = post_rots.new_zeros(imgs.size(0) if imgs.dim()==4 else post_rots.size(0), 3, 3)
        S[:, 0, 0] = sx
        S[:, 1, 1] = sy
        S[:, 2, 2] = 1.0

        # S @ post_rots
        post_rots = torch.bmm(S, post_rots)
        # S @ post_trans
        post_trans = torch.bmm(S, post_trans.unsqueeze(-1)).squeeze(-1)

        return imgs, intrins, post_rots, post_trans

    def __getitem__(self, index):
        info = self.sample_infos[index]
        frame_dir = self._frame_dir(info)
        with open(frame_dir / "meta.json", "r") as f:
            meta = json.load(f)
        all_cams = meta.get("cam_idx", [0, 1, 2, 3, 4, 5])
        cams = self.choose_cams(all_cams)

        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(info, cams)
        binimg = self.get_binimg(info)

        # (6,3,224,480) -> (6,3,128,352)
        imgs, intrins, post_rots, post_trans = self._resize_and_adjust(
            imgs, intrins, post_rots, post_trans, self.target_hw
        )
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg

def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    """
    Kept the same signature for drop-in replacement.

    Args:
      - version: unused (kept for API compatibility)
      - dataroot: path to SimBEV root (containing {split}_samples.json and split folders)
      - parser_name: 'vizdata' | 'segmentationdata'
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
