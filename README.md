# Lift-Splat-Shoot for SimBEV Dataset

PyTorch implementation of **Lift, Splat, Shoot** ([ECCV 2020](https://arxiv.org/abs/2008.05711)) adapted for the SimBEV dataset.

Original repository: [nv-tlabs/lift-splat-shoot](https://github.com/nv-tlabs/lift-splat-shoot)

## Overview

This is a modified version of the Lift-Splat-Shoot (LSS) model for training on the SimBEV synthetic dataset instead of nuScenes. The model learns to predict Bird's Eye View (BEV) vehicle segmentation from multi-camera images.

**Key Features:**
- ✅ SimBEV dataset loader with proper camera calibration
- ✅ BEV vehicle segmentation (merged from 3 vehicle classes)
- ✅ Wandb integration with BEV visualizations
- ✅ Training/validation monitoring with IoU metrics
- ✅ Data validation and debugging tools

## Project Structure

```
lift-splat-shoot/
├── src/
│   ├── data_simbev.py      # SimBEV dataset loader
│   ├── models.py            # LSS model architecture
│   ├── tools.py             # Utility functions
│   └── ...
├── configs/
│   └── simbev_small.sh      # Training configuration
├── train_simbev.py          # Main training script
├── debug/                   # Data validation scripts
│   ├── debug_data_loading.py
│   ├── verify_bev_classes.py
│   ├── verify_camera_projection.py
│   └── test_visualization.py
├── scripts/                 # Additional scripts
│   └── cvt_simbev_dataloader.py
├── docs/                    # Documentation
│   ├── README_SIMBEV.md
│   ├── SIMBEV_DATALOADER.md
│   └── TRAINING_GUIDE.md
└── debug_outputs/           # Debug visualization outputs
```

## Installation

### Requirements

```bash
# Create conda environment
conda create -n lss python=3.8
conda activate lss

# Install PyTorch (CUDA 11.x)
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

# Install dependencies
pip install -r requirements.txt
pip install efficientnet_pytorch tensorboardX wandb matplotlib Pillow tqdm
```

### Dataset Setup

Place your SimBEV dataset in `/data/SimBEV/` with the following structure:

```
/data/SimBEV/
├── SimBEV_cvt_label/
│   ├── scene_0000/
│   │   └── yaw0pitch0/
│   │       ├── meta.json
│   │       ├── bev_*.npz
│   │       └── ...
│   ├── scene_0001/
│   └── ...
└── sweeps/
    ├── RGB-CAM_FRONT/
    ├── RGB-CAM_FRONT_LEFT/
    └── ...
```

## Quick Start

### 1. Verify Data Loading

```bash
conda activate lss
python debug/debug_data_loading.py
```

Check `debug_outputs/data_loading_debug.png` to verify GT is correct.

### 2. Training

```bash
# Using configuration script
bash configs/simbev_small.sh

# Or directly with Python
python train_simbev.py \
    --dataroot /data/SimBEV \
    --gpuid 0 \
    --nepochs 25 \
    --bsz 8 \
    --nworkers 8 \
    --lr 0.0005 \
    --logdir ./runs/simbev_experiment \
    --use_wandb \
    --wandb_project SIMBEV-lift-splat-shoot \
    --wandb_name my_experiment
```

### 3. Monitor Training

**With Wandb:**
- Training/validation loss and IoU metrics
- BEV visualizations every 50 iterations
- GT vs Prediction overlay images

**With Tensorboard:**
```bash
tensorboard --logdir ./runs/simbev_experiment
```

## Key Implementation Details

### SimBEV Data Loader ([src/data_simbev.py](src/data_simbev.py))

**BEV Classes:**
- **Class 0**: Drivable area (not used)
- **Classes 1-3**: Different vehicle types (merged for training)
- Model predicts binary vehicle segmentation (1 channel)

**Camera Extrinsics:**
- SimBEV provides `ego→cam` transformation
- LSS uses this format directly (no inversion needed)
- Verified with projection tests in `debug/verify_camera_projection.py`

**Data Augmentation:**
- Random resize: 0.9-1.1x
- Random rotation: ±5.4 degrees
- Random horizontal flip
- Random crop
- ImageNet normalization

### Model Configuration

**Grid Configuration:**
```python
grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],  # 200 bins, 100m range
    'ybound': [-50.0, 50.0, 0.5],  # 200 bins, 100m range
    'zbound': [-10.0, 10.0, 20.0], # Height range
    'dbound': [4.0, 45.0, 1.0],    # Depth discretization
}
```

**Output:**
- BEV size: 200×200 (0.5m resolution)
- Output channels: 1 (binary vehicle segmentation)
- Loss: BCEWithLogitsLoss (pos_weight=2.13)

## Wandb Logging

Every **50 iterations** (training):
- `train/loss`: BCE loss
- `train/iou`: IoU metric
- `train/bev_visualization`: GT, Prediction, and Overlay
  - Red = GT only (False Negative)
  - Green = Pred only (False Positive)
  - Yellow = Match (True Positive)

Every **val_step** (validation):
- `val/loss`: Validation loss
- `val/iou`: Validation IoU
- `val/bev_visualization`: Validation BEV visualization

## Debugging Tools

### 1. Data Loading Verification
```bash
python debug/debug_data_loading.py
```
Checks: tensor shapes, value ranges, GT statistics, camera calibration

### 2. BEV Class Visualization
```bash
python debug/verify_bev_classes.py
```
Visualizes all 8 BEV classes to identify vehicle channels

### 3. Camera Projection Test
```bash
python debug/verify_camera_projection.py
```
Verifies extrinsics by projecting 3D points to images

### 4. Training Visualization Test
```bash
python debug/test_visualization.py
```
Tests wandb visualization code without full training

## Training Tips

1. **Batch Size**: Start with 4-8, increase if GPU memory allows
2. **Learning Rate**: 1e-3 (default) or 5e-4 for more stable training
3. **Validation**: Run every 500-1000 iterations
4. **Checkpoints**: Save every 1000 iterations
5. **Early Stopping**: Monitor validation IoU

## Model Performance

**Expected Results** (after ~10-20k iterations):
- Training IoU: 0.3-0.5
- Validation IoU: 0.25-0.45
- Initial predictions: ~0.5 (random)
- Converged predictions: Clear vehicle shapes

## Known Issues & Solutions

### Issue: GT has all zeros
**Solution**: Check BEV class selection in `src/data_simbev.py:234`

### Issue: Camera projection looks wrong
**Solution**: Verify extrinsics with `debug/verify_camera_projection.py`

### Issue: Loss is NaN
**Solution**: Reduce learning rate or check data normalization

### Issue: IoU stuck at 0
**Solution**: Check GT positive ratio (should be ~1-5%)

## Citation

```bibtex
@inproceedings{philion2020lift,
  title={Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d},
  author={Philion, Jonah and Fidler, Sanja},
  booktitle={European Conference on Computer Vision},
  pages={194--210},
  year={2020},
  organization={Springer}
}
```

## License

This project inherits the license from the original LSS repository.
See [LICENSE](LICENSE) for details.

## Additional Documentation

- [docs/README_SIMBEV.md](docs/README_SIMBEV.md) - SimBEV dataset details
- [docs/SIMBEV_DATALOADER.md](docs/SIMBEV_DATALOADER.md) - Data loader implementation
- [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) - Detailed training guide

## Contact

For questions about the SimBEV adaptation, please open an issue.

For questions about the original LSS model, refer to the [original repository](https://github.com/nv-tlabs/lift-splat-shoot).
