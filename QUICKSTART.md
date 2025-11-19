# Quick Start Guide

## 1. Environment Setup (One-time)

```bash
# Activate conda environment
conda activate lss

# If not installed yet:
pip install -r requirements.txt
```

## 2. Verify Data Loading (Recommended)

```bash
python debug/debug_data_loading.py
```

✅ Check `debug_outputs/data_loading_debug.png` to verify GT is correct

## 3. Start Training

### Option A: Using Config Script (Recommended)
```bash
bash configs/simbev_small.sh
```

### Option B: Direct Python Command
```bash
python train_simbev.py \
    --dataroot /data/SimBEV \
    --gpuid 0 \
    --nepochs 25 \
    --bsz 8 \
    --nworkers 8 \
    --lr 0.0005 \
    --logdir ./runs/my_experiment \
    --use_wandb \
    --wandb_project SIMBEV-lift-splat-shoot \
    --wandb_name my_experiment
```

## 4. Monitor Training

### Wandb (Primary)
- Login: `wandb login`
- View: https://wandb.ai/your-team/SIMBEV-lift-splat-shoot

**Metrics logged every 50 iterations:**
- Training loss & IoU
- BEV visualization (GT vs Prediction)

**Validation:**
- Validation loss & IoU
- Validation BEV visualization

### TensorBoard (Alternative)
```bash
tensorboard --logdir ./runs/my_experiment
```

## 5. Common Issues

### Issue: Out of GPU memory
```bash
# Reduce batch size
python train_simbev.py --bsz 4 ...
```

### Issue: GT looks wrong
```bash
# Visualize all BEV classes
python debug/verify_bev_classes.py

# Check output in debug_outputs/bev_all_classes.png
```

### Issue: Camera projection issues
```bash
# Verify camera calibration
python debug/verify_camera_projection.py

# Check output in debug_outputs/camera_projection_test.png
```

## 6. Resume Training

```bash
python train_simbev.py \
    --resume ./runs/my_experiment/model_010000.pt \
    ... (other args)
```

## 7. Check Results

**Best model saved at:**
- `./runs/my_experiment/model_best.pt`

**Regular checkpoints:**
- `./runs/my_experiment/model_001000.pt`
- `./runs/my_experiment/model_002000.pt`
- etc.

## Training Progress Expectations

| Iteration | Training IoU | Notes |
|-----------|--------------|-------|
| 0-100     | ~0.0-0.05   | Random initialization |
| 100-500   | 0.05-0.15   | Learning basic shapes |
| 500-2000  | 0.15-0.30   | Recognizing vehicles |
| 2000+     | 0.30-0.50   | Converged |

**Validation IoU** typically 0.05-0.1 lower than training IoU

## Tips

1. **Start small**: Test with `--bsz 4 --nepochs 5` first
2. **Monitor wandb**: Check visualizations every few minutes
3. **Early stopping**: If validation IoU plateaus, stop training
4. **Learning rate**: Try 1e-3 (default) or 5e-4 if unstable
5. **Checkpoints**: Save frequently, disk space is cheap!

## Full Documentation

- [README.md](README.md) - Complete project documentation
- [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) - Detailed training guide
- [docs/SIMBEV_DATALOADER.md](docs/SIMBEV_DATALOADER.md) - Data loader details

---

**Need help?** Check debug outputs or open an issue!
