"""
Training script for LSS model on SimBEV dataset
"""

import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
from pathlib import Path
import wandb
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models import compile_model
from src.data_simbev import compile_data
from src.tools import SimpleLoss, get_batch_iou, get_val_info


def train(
    dataroot,
    nepochs=100,
    gpuid=0,

    # Image config
    H=224,
    W=480,
    resize_lim=(1.0, 1.0),  # No resize augmentation
    final_dim=(128, 352),
    bot_pct_lim=(0.0, 0.0),  # No bottom crop
    rot_lim=(0.0, 0.0),  # No rotation augmentation
    rand_flip=False,  # No horizontal flip augmentation
    ncams=6,

    # Training config
    max_grad_norm=5.0,
    pos_weight=2.13,
    logdir='./runs/simbev',

    # BEV grid config
    xbound=[-50.0, 50.0, 0.5],
    ybound=[-50.0, 50.0, 0.5],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[4.0, 45.0, 1.0],

    # Optimization config
    bsz=4,
    nworkers=4,
    lr=1e-3,
    weight_decay=1e-7,

    # Validation config
    val_step=500,
    save_step=1000,

    # Resume from checkpoint
    resume=None,

    # Wandb config
    use_wandb=True,
    wandb_project='lift-splat-shoot',
    wandb_name=None,
    wandb_entity=None,
):
    """
    Train LSS model on SimBEV dataset.

    Args:
        dataroot: Path to SimBEV dataset root directory
        nepochs: Number of training epochs
        gpuid: GPU ID to use (-1 for CPU)
        H, W: Original image height and width
        resize_lim: Resize augmentation range
        final_dim: Target image dimensions (H, W)
        bot_pct_lim: Bottom crop percentage range
        rot_lim: Rotation augmentation range (degrees)
        rand_flip: Enable random horizontal flip
        ncams: Number of cameras to use (max 6)
        max_grad_norm: Gradient clipping threshold
        pos_weight: Positive class weight for loss
        logdir: Directory for logs and checkpoints
        xbound, ybound, zbound: BEV grid bounds [min, max, resolution]
        dbound: Depth discretization bounds [min, max, resolution]
        bsz: Batch size
        nworkers: Number of dataloader workers
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        val_step: Validation frequency (iterations)
        save_step: Checkpoint saving frequency (iterations)
        resume: Path to checkpoint to resume from
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_name: W&B run name (auto-generated if None)
        wandb_entity: W&B entity/team name
    """

    # Create log directory
    os.makedirs(logdir, exist_ok=True)

    # Setup configurations
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }

    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H,
        'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'Ncams': ncams,
    }

    # Initialize wandb
    if use_wandb:
        wandb_config = {
            'dataroot': dataroot,
            'nepochs': nepochs,
            'batch_size': bsz,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'num_workers': nworkers,
            'num_cameras': ncams,
            'image_H': H,
            'image_W': W,
            'final_dim': final_dim,
            'max_grad_norm': max_grad_norm,
            'pos_weight': pos_weight,
            'grid_conf': grid_conf,
            'data_aug_conf': data_aug_conf,
            'val_step': val_step,
            'save_step': save_step,
        }

        wandb.init(
            project=wandb_project,
            name=wandb_name,
            entity=wandb_entity,
            config=wandb_config,
            dir=logdir,
        )
        print(f"Wandb initialized: {wandb.run.name}")

    print("=" * 80)
    print("Training Configuration:")
    print(f"  Dataroot: {dataroot}")
    print(f"  Log directory: {logdir}")
    print(f"  Batch size: {bsz}")
    print(f"  Learning rate: {lr}")
    print(f"  Number of epochs: {nepochs}")
    print(f"  Number of cameras: {ncams}")
    print(f"  Image size: {H}x{W} -> {final_dim}")
    print(f"  Wandb logging: {use_wandb}")
    print("=" * 80)

    # Load data
    print("\nLoading SimBEV dataset...")
    trainloader, valloader = compile_data(
        version='unused',
        dataroot=dataroot,
        data_aug_conf=data_aug_conf,
        grid_conf=grid_conf,
        bsz=bsz,
        nworkers=nworkers,
        parser_name='segmentationdata'
    )
    print(f"Train batches: {len(trainloader)}")
    print(f"Val batches: {len(valloader)}")

    # Setup device
    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    print(f"Using device: {device}")

    # Create model
    print("\nCreating LSS model...")
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # Setup optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Setup loss function
    loss_fn = SimpleLoss(pos_weight).to(device)

    # Setup tensorboard writer
    writer = SummaryWriter(logdir=logdir)

    # Resume from checkpoint if specified
    counter = 0
    start_epoch = 0
    if resume is not None and os.path.exists(resume):
        print(f"\nResuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            counter = checkpoint.get('counter', 0)
            start_epoch = checkpoint.get('epoch', 0)
        else:
            model.load_state_dict(checkpoint)
        print(f"Resumed from epoch {start_epoch}, iteration {counter}")

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    model.train()
    best_val_iou = 0.0

    for epoch in range(start_epoch, nepochs):
        np.random.seed()

        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{nepochs}')
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(pbar):
            t0 = time()

            # Forward pass
            opt.zero_grad()
            preds = model(
                imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
            )
            binimgs = binimgs.to(device)

            # Compute loss
            loss = loss_fn(preds, binimgs)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            counter += 1
            t1 = time()

            # Log training loss
            if counter % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                writer.add_scalar('train/loss', loss.item(), counter)
                if use_wandb:
                    wandb.log({'train/loss': loss.item(), 'iteration': counter})

            # Log training IoU and visualizations
            if counter % 100 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})

                if use_wandb:
                    # Create comprehensive visualization with 6 camera views + BEV
                    pred_vis = torch.sigmoid(preds[0, 0]).detach().cpu().numpy()  # (200, 200)
                    gt_vis = binimgs[0, 0].detach().cpu().numpy()  # (200, 200)

                    # Get camera images (first sample in batch)
                    cam_imgs = imgs[0].detach().cpu().numpy()  # (6, 3, H, W)
                    cam_names = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'BACK_LEFT', 'BACK', 'BACK_RIGHT']

                    # Create figure: 3 rows (cameras, BEV GTs, BEV preds)
                    fig = plt.figure(figsize=(20, 12))

                    # Row 1: 6 camera views
                    for i in range(6):
                        ax = plt.subplot(3, 6, i + 1)
                        img = cam_imgs[i].transpose(1, 2, 0)  # (H, W, 3)
                        # Normalize to [0, 1] for display
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                        ax.imshow(img)
                        ax.set_title(cam_names[i], fontsize=10, fontweight='bold')
                        ax.axis('off')

                    # Row 2: BEV Ground Truth
                    ax = plt.subplot(3, 3, 7)
                    ax.imshow(gt_vis, cmap='hot', vmin=0, vmax=1, origin='lower', extent=[-50, 50, -50, 50])
                    ax.set_title('BEV Ground Truth', fontsize=12, fontweight='bold')
                    ax.set_xlabel('X (m)', fontsize=10)
                    ax.set_ylabel('Y (m)', fontsize=10)
                    ax.grid(True, alpha=0.3)

                    # Row 2: BEV Prediction
                    ax = plt.subplot(3, 3, 8)
                    ax.imshow(pred_vis, cmap='hot', vmin=0, vmax=1, origin='lower', extent=[-50, 50, -50, 50])
                    ax.set_title('BEV Prediction', fontsize=12, fontweight='bold')
                    ax.set_xlabel('X (m)', fontsize=10)
                    ax.set_ylabel('Y (m)', fontsize=10)
                    ax.grid(True, alpha=0.3)

                    # Row 2: BEV Overlay
                    ax = plt.subplot(3, 3, 9)
                    overlay = np.zeros((gt_vis.shape[0], gt_vis.shape[1], 3))
                    overlay[:, :, 0] = gt_vis  # GT in red
                    overlay[:, :, 1] = pred_vis  # Pred in green
                    ax.imshow(overlay, origin='lower', extent=[-50, 50, -50, 50])
                    ax.set_title('Overlay (GT=Red, Pred=Green, Match=Yellow)', fontsize=12, fontweight='bold')
                    ax.set_xlabel('X (m)', fontsize=10)
                    ax.set_ylabel('Y (m)', fontsize=10)
                    ax.grid(True, alpha=0.3)

                    plt.suptitle(f'Training Iteration {counter} | IoU: {iou:.4f}', fontsize=14, fontweight='bold')
                    plt.tight_layout()

                    # Log to wandb
                    wandb.log({
                        'train/iou': iou,
                        'train/epoch': epoch,
                        'train/step_time': t1 - t0,
                        'train/visualization': wandb.Image(fig),
                        'iteration': counter,
                    })

                    plt.close(fig)

            # Validation
            if counter % val_step == 0:
                tqdm.write(f"\n  Running validation at iteration {counter}...")
                model.eval()
                val_info = get_val_info(model, valloader, loss_fn, device)

                # Get one validation batch for visualization
                val_batch = next(iter(valloader))
                with torch.no_grad():
                    val_imgs, val_rots, val_trans, val_intrins, val_post_rots, val_post_trans, val_binimgs = val_batch
                    val_preds = model(
                        val_imgs.to(device),
                        val_rots.to(device),
                        val_trans.to(device),
                        val_intrins.to(device),
                        val_post_rots.to(device),
                        val_post_trans.to(device),
                    )

                model.train()

                tqdm.write(f"  Validation - Loss: {val_info['loss']:.4f}, IoU: {val_info['iou']:.4f}")
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

                if use_wandb:
                    # Create comprehensive validation visualization with 6 camera views + BEV
                    val_pred_vis = torch.sigmoid(val_preds[0, 0]).detach().cpu().numpy()
                    val_gt_vis = val_binimgs[0, 0].cpu().numpy()

                    # Get validation camera images
                    val_cam_imgs = val_imgs[0].cpu().numpy()  # (6, 3, H, W)
                    cam_names = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'BACK_LEFT', 'BACK', 'BACK_RIGHT']

                    # Create figure: 3 rows (cameras, BEV GTs, BEV preds)
                    fig = plt.figure(figsize=(20, 12))

                    # Row 1: 6 camera views
                    for i in range(6):
                        ax = plt.subplot(3, 6, i + 1)
                        img = val_cam_imgs[i].transpose(1, 2, 0)  # (H, W, 3)
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                        ax.imshow(img)
                        ax.set_title(cam_names[i], fontsize=10, fontweight='bold')
                        ax.axis('off')

                    # Row 2: BEV Ground Truth
                    ax = plt.subplot(3, 3, 7)
                    ax.imshow(val_gt_vis, cmap='hot', vmin=0, vmax=1, origin='lower', extent=[-50, 50, -50, 50])
                    ax.set_title('BEV Ground Truth', fontsize=12, fontweight='bold')
                    ax.set_xlabel('X (m)', fontsize=10)
                    ax.set_ylabel('Y (m)', fontsize=10)
                    ax.grid(True, alpha=0.3)

                    # Row 2: BEV Prediction
                    ax = plt.subplot(3, 3, 8)
                    ax.imshow(val_pred_vis, cmap='hot', vmin=0, vmax=1, origin='lower', extent=[-50, 50, -50, 50])
                    ax.set_title('BEV Prediction', fontsize=12, fontweight='bold')
                    ax.set_xlabel('X (m)', fontsize=10)
                    ax.set_ylabel('Y (m)', fontsize=10)
                    ax.grid(True, alpha=0.3)

                    # Row 2: BEV Overlay
                    ax = plt.subplot(3, 3, 9)
                    overlay = np.zeros((val_gt_vis.shape[0], val_gt_vis.shape[1], 3))
                    overlay[:, :, 0] = val_gt_vis  # GT in red
                    overlay[:, :, 1] = val_pred_vis  # Pred in green
                    ax.imshow(overlay, origin='lower', extent=[-50, 50, -50, 50])
                    ax.set_title('Overlay (GT=Red, Pred=Green, Match=Yellow)', fontsize=12, fontweight='bold')
                    ax.set_xlabel('X (m)', fontsize=10)
                    ax.set_ylabel('Y (m)', fontsize=10)
                    ax.grid(True, alpha=0.3)

                    plt.suptitle(f'Validation Iteration {counter} | IoU: {val_info["iou"]:.4f}', fontsize=14, fontweight='bold')
                    plt.tight_layout()

                    wandb.log({
                        'val/loss': val_info['loss'],
                        'val/iou': val_info['iou'],
                        'val/visualization': wandb.Image(fig),
                        'train/lr': lr,
                        'iteration': counter,
                    })

                    plt.close(fig)

                # Save best model based on IoU
                if val_info['iou'] > best_val_iou:
                    best_val_iou = val_info['iou']
                    best_path = os.path.join(logdir, "model_best.pt")
                    tqdm.write(f"  New best IoU: {best_val_iou:.4f}, saving to {best_path}")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'counter': counter,
                        'epoch': epoch,
                        'val_iou': best_val_iou,
                    }, best_path)
                    if use_wandb:
                        wandb.run.summary["best_val_iou"] = best_val_iou

            # Save checkpoint
            if counter % save_step == 0:
                model.eval()
                ckpt_path = os.path.join(logdir, f"model_{counter:06d}.pt")
                tqdm.write(f"  Saving checkpoint to {ckpt_path}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'counter': counter,
                    'epoch': epoch,
                }, ckpt_path)
                model.train()

    # Save final model
    final_path = os.path.join(logdir, "model_final.pt")
    print(f"\nTraining complete! Saving final model to {final_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'counter': counter,
        'epoch': nepochs,
    }, final_path)

    writer.close()

    if use_wandb:
        wandb.finish()

    print(f"Best validation IoU: {best_val_iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train LSS on SimBEV dataset')

    # Data arguments
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to SimBEV dataset root directory')

    # Training arguments
    parser.add_argument('--nepochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--gpuid', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--bsz', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--nworkers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-7,
                        help='Weight decay')

    # Image arguments
    parser.add_argument('--H', type=int, default=224,
                        help='Original image height')
    parser.add_argument('--W', type=int, default=480,
                        help='Original image width')
    parser.add_argument('--final_h', type=int, default=128,
                        help='Target image height')
    parser.add_argument('--final_w', type=int, default=352,
                        help='Target image width')
    parser.add_argument('--ncams', type=int, default=6,
                        help='Number of cameras to use')

    # Training config
    parser.add_argument('--logdir', type=str, default='./runs/simbev',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--val_step', type=int, default=500,
                        help='Validation frequency (iterations)')
    parser.add_argument('--save_step', type=int, default=1000,
                        help='Checkpoint saving frequency (iterations)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='lift-splat-shoot',
                        help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='W&B run name (auto-generated if not specified)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity/team name')

    args = parser.parse_args()

    train(
        dataroot=args.dataroot,
        nepochs=args.nepochs,
        gpuid=args.gpuid,
        H=args.H,
        W=args.W,
        final_dim=(args.final_h, args.final_w),
        ncams=args.ncams,
        bsz=args.bsz,
        nworkers=args.nworkers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        logdir=args.logdir,
        val_step=args.val_step,
        save_step=args.save_step,
        resume=args.resume,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
    )


if __name__ == '__main__':
    main()
