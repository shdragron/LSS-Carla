import copy
import json
import logging
import torch

from pathlib import Path
from typing import Dict, List, Optional
from .common import get_split
from .transforms import Sample, LoadDataTransform


log = logging.getLogger(__name__)

CAMERA_ORDER = [
    'front_left', 'front', 'front_right',
    'back_left', 'back', 'back_right'
]
CAMERA_NAME_TO_INDEX = {name: idx for idx, name in enumerate(CAMERA_ORDER)}


def get_data(
    dataset_dir,
    labels_dir,
    split,
    num_classes,
    augment='none',
    image=None,                         # image config
    dataset='unused',                   # ignore
    extrinsic_noise=None,
    viewchange=None,
    which_view='all',
    original_extrinsic=False,
    base_orientation='yaw0pitch0',
    swap_images=True,
    **dataset_kwargs
):
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)

    # Override augment if not training
    augment = 'none' if split != 'train' else augment
    noise_cfg = extrinsic_noise if split == 'train' else {}
    transform = LoadDataTransform(dataset_dir, labels_dir, image, num_classes, augment,
                                  extrinsic_noise=noise_cfg)

    split_scenes = get_split(split, 'simbev')

    datasets = []
    for scene in split_scenes:
        datasets.append(
            SimBEVGeneratedDataset(
                scene,
                labels_dir,
                transform=transform,
                split=split,
                viewchange=viewchange,
                which_view=which_view,
                original_extrinsic=original_extrinsic,
                base_orientation=base_orientation,
                swap_images=swap_images,
            )
        )

    return datasets


class SimBEVGeneratedDataset(torch.utils.data.Dataset):
    """Dataset wrapper with optional viewpoint overrides."""

    def __init__(
        self,
        scene_name,
        labels_dir,
        transform=None,
        split='train',
        viewchange=None,
        which_view='all',
        original_extrinsic=False,
        base_orientation='yaw0pitch0',
        swap_images=True,
    ):
        self.scene_name = scene_name
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.split = split

        self.base_orientation = (base_orientation or '').lower() or None
        self.target_orientation = self._normalize_viewchange(viewchange)
        self.original_extrinsic = original_extrinsic
        self.swap_images = swap_images
        self.cam_indices = self._resolve_camera_indices(which_view)

        self.scene_dir = self.labels_dir / self.scene_name
        if not self.scene_dir.is_dir():
            raise FileNotFoundError(f'Could not find metadata for scene {scene_name} in {labels_dir}')

        self.available_orientations = self._discover_orientations()
        self.base_orientation = self._resolve_base_orientation()

        self.samples = self._load_scene_samples()
        self.base_samples = self._filter_samples_by_orientation(self.base_orientation)
        if not self.base_samples:
            raise FileNotFoundError(
                f"No samples found for base orientation '{self.base_orientation}' in scene {self.scene_name}."
            )

        if self._requires_viewchange():
            # Only iterate over the canonical viewpoint when applying overrides.
            self.samples = self.base_samples
            self.alt_samples = self._load_orientation_lookup(self.target_orientation)
        else:
            self.alt_samples = dict()
        self._missing_alt_tokens = set()

    def _normalize_viewchange(self, viewchange: Optional[str]) -> Optional[str]:
        if viewchange is None:
            return None
        normalized = str(viewchange).strip().lower()
        if not normalized or normalized in {'normal', 'none', 'yaw0pitch0'}:
            return None
        return normalized

    def _discover_orientations(self) -> List[Optional[str]]:
        orientations: List[Optional[str]] = []
        for entry in sorted(self.scene_dir.iterdir()):
            if entry.is_dir():
                orientations.append(entry.name.lower())
        if not orientations:
            # Allow datasets where meta.json sits directly under the scene directory
            meta_at_root = list(self.scene_dir.glob('meta.json'))
            if meta_at_root:
                orientations.append(None)
        if not orientations:
            raise FileNotFoundError(f'No orientations found for scene {self.scene_name} under {self.scene_dir}')
        return orientations

    def _resolve_base_orientation(self) -> Optional[str]:
        if self.base_orientation in self.available_orientations:
            return self.base_orientation
        if None in self.available_orientations and self.base_orientation is None:
            return None
        # Default to the first available orientation when requested one is missing
        resolved = self.available_orientations[0]
        if self.base_orientation is not None and self.split in {'val', 'test'}:
            log.warning(
                "Base orientation '%s' not found for scene %s. Falling back to '%s'.",
                self.base_orientation,
                self.scene_name,
                resolved,
            )
        return resolved

    def _requires_viewchange(self) -> bool:
        return self.target_orientation is not None and self.target_orientation != self.base_orientation

    def _resolve_camera_indices(self, which_view: Optional[str]) -> List[int]:
        if which_view is None:
            return list(range(len(CAMERA_ORDER)))
        view_key = which_view.lower()
        if view_key == 'all':
            return list(range(len(CAMERA_ORDER)))
        if view_key not in CAMERA_NAME_TO_INDEX:
            raise ValueError(
                f"Unsupported camera selection '{which_view}'. Choose from ['all'] + {list(CAMERA_NAME_TO_INDEX.keys())}."
            )
        return [CAMERA_NAME_TO_INDEX[view_key]]

    def _load_scene_samples(self, orientations: Optional[set] = None):
        samples = []
        meta_files = sorted(self.scene_dir.rglob('meta.json'))
        if not meta_files:
            raise FileNotFoundError(f'No meta.json files found for scene {self.scene_name} under {self.scene_dir}')

        for meta_path in meta_files:
            orientation_rel = meta_path.parent.relative_to(self.scene_dir)
            orientation = None if not orientation_rel.parts else orientation_rel.as_posix().lower()
            if orientations is not None and orientation not in orientations:
                continue

            meta_samples = json.loads(meta_path.read_text())
            for item in meta_samples:
                if orientation:
                    item.setdefault('orientation', orientation)
            samples.extend(meta_samples)

        if not samples:
            selected = next(iter(orientations)) if orientations else 'all orientations'
            raise FileNotFoundError(
                f'No samples found for scene {self.scene_name} under orientation {selected}'
            )

        return samples

    def _load_orientation_lookup(self, orientation: str) -> Dict[str, dict]:
        if orientation not in self.available_orientations:
            raise FileNotFoundError(
                f"Orientation '{orientation}' not available for scene {self.scene_name}. "
                f"Available: {self.available_orientations}"
            )
        samples = self._load_scene_samples({orientation})
        return {sample['token']: sample for sample in samples}

    def _filter_samples_by_orientation(self, orientation: Optional[str]) -> List[dict]:
        orientation = orientation or None
        filtered = []
        for sample in self.samples:
            sample_orientation = sample.get('orientation')
            sample_orientation = sample_orientation.lower() if isinstance(sample_orientation, str) else None
            if sample_orientation == orientation:
                filtered.append(sample)
            elif sample_orientation is None and orientation is None:
                filtered.append(sample)
        return filtered

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dict = copy.deepcopy(self.samples[idx])

        if self._requires_viewchange():
            token = sample_dict['token']
            target = self.alt_samples.get(token)
            if target is not None:
                self._apply_viewchange(sample_dict, target)
            elif token not in self._missing_alt_tokens:
                log.warning(
                    "Token '%s' not found in orientation '%s' for scene %s. Using base sample.",
                    token,
                    self.target_orientation,
                    self.scene_name,
                )
                self._missing_alt_tokens.add(token)

        data = Sample(**sample_dict)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def _apply_viewchange(self, base_sample: dict, target_sample: dict):
        for cam_idx in self.cam_indices:
            if cam_idx >= len(base_sample['images']) or cam_idx >= len(target_sample['images']):
                continue
            if self.swap_images:
                base_sample['images'][cam_idx] = target_sample['images'][cam_idx]
            if not self.original_extrinsic and cam_idx < len(target_sample['extrinsics']):
                base_sample['extrinsics'][cam_idx] = target_sample['extrinsics'][cam_idx]