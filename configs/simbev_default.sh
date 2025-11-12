#!/bin/bash

# SimBEV 기본 학습 설정

# 데이터 경로 (실제 데이터셋 경로로 수정하세요)
DATAROOT="/path/to/simbev/dataset"

# GPU 설정
GPUID=0

# 학습 하이퍼파라미터
EPOCHS=100
BATCH_SIZE=4
NUM_WORKERS=4
LEARNING_RATE=0.001

# 이미지 설정 (SimBEV 기본값)
IMAGE_H=224
IMAGE_W=480
FINAL_H=128
FINAL_W=352
NUM_CAMS=6

# 로그 디렉토리
LOGDIR="./runs/simbev_$(date +%Y%m%d_%H%M%S)"

# 학습 실행
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
    --val_step 500 \
    --save_step 1000
