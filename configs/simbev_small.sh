#!/bin/bash

# SimBEV 소규모 실험용 설정 (빠른 테스트)

DATAROOT="/data/SimBEV"

GPUID=0
EPOCHS=30
BATCH_SIZE=8
NUM_WORKERS=8
LEARNING_RATE=0.0005

IMAGE_H=224
IMAGE_W=480
FINAL_H=128
FINAL_W=352
NUM_CAMS=6

LOGDIR="./runs/simbev_test_$(date +%Y%m%d_%H%M%S)"

python train_simbev.py \
    --dataroot $DATAROOT \
    --gpuid $GPUID \
    --nepochs $EPOCHS \
    --bsz $BATCH_SIZE \
    --nworkers $NUM_WORKERS \
    --lr $LEARNING_RATE \
    --H $IMAGE_H \
    --W $IMAGE_W \
    --final_h $FINAL_H \
    --final_w $FINAL_W \
    --ncams $NUM_CAMS \
    --logdir $LOGDIR \
    --val_step 8640 \
    --save_step 4320 \
    --use_wandb \
    --wandb_project SIMBEV-lift-splat-shoot \
    --wandb_name simbev_small_experiment
