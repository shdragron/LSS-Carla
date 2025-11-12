#!/bin/bash

# SimBEV 소규모 실험용 설정 (빠른 테스트)

DATAROOT="/path/to/simbev/dataset"

GPUID=0
EPOCHS=10
BATCH_SIZE=2
NUM_WORKERS=2
LEARNING_RATE=0.001

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
    --val_step 50 \
    --save_step 100
