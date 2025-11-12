# SimBEV 학습 가이드

LSS (Lift-Splat-Shoot) 모델을 SimBEV 데이터셋으로 학습하는 방법을 설명합니다.

## 빠른 시작

### 1. 데이터셋 준비

SimBEV 데이터셋이 다음 구조로 준비되어 있는지 확인하세요:

```
/path/to/simbev/dataset/
├── train/
│   ├── scene_0001/
│   │   └── yaw0pitch0/
│   │       ├── meta.json
│   │       ├── images/
│   │       └── labels/
│   └── ...
└── val/
    └── ...
```

자세한 데이터 형식은 [SIMBEV_DATALOADER.md](SIMBEV_DATALOADER.md)를 참고하세요.

### 2. 설정 파일 수정

`configs/simbev_default.sh` 파일을 열어서 데이터셋 경로를 수정하세요:

```bash
# 이 부분을 실제 데이터셋 경로로 수정
DATAROOT="/path/to/simbev/dataset"
```

### 3. 학습 시작

#### 방법 1: 쉘 스크립트 사용 (권장)

```bash
# 기본 설정으로 학습
bash configs/simbev_default.sh

# 빠른 테스트용 (소규모)
bash configs/simbev_small.sh
```

#### 방법 2: Python 직접 실행

```bash
python train_simbev.py \
    --dataroot /path/to/simbev/dataset \
    --gpuid 0 \
    --nepochs 100 \
    --bsz 4 \
    --nworkers 4 \
    --logdir ./runs/my_experiment
```

## 명령줄 인자

### 필수 인자

- `--dataroot`: SimBEV 데이터셋 경로 (필수)

### 학습 설정

- `--nepochs`: 학습 에포크 수 (기본값: 100)
- `--gpuid`: 사용할 GPU ID, -1이면 CPU (기본값: 0)
- `--bsz`: 배치 크기 (기본값: 4)
- `--nworkers`: 데이터로더 워커 수 (기본값: 4)
- `--lr`: 학습률 (기본값: 1e-3)
- `--weight_decay`: Weight decay (기본값: 1e-7)

### 이미지 설정

- `--H`: 원본 이미지 높이 (기본값: 224)
- `--W`: 원본 이미지 너비 (기본값: 480)
- `--final_h`: 타겟 이미지 높이 (기본값: 128)
- `--final_w`: 타겟 이미지 너비 (기본값: 352)
- `--ncams`: 사용할 카메라 개수 (기본값: 6)

### 로깅 및 체크포인트

- `--logdir`: 로그 및 체크포인트 저장 디렉토리 (기본값: ./runs/simbev)
- `--val_step`: 검증 주기 (iterations) (기본값: 500)
- `--save_step`: 체크포인트 저장 주기 (iterations) (기본값: 1000)
- `--resume`: 재개할 체크포인트 경로 (선택)

## 학습 재개

학습 중단된 경우 체크포인트에서 재개할 수 있습니다:

```bash
python train_simbev.py \
    --dataroot /path/to/simbev/dataset \
    --resume ./runs/simbev/model_010000.pt \
    --logdir ./runs/simbev
```

## TensorBoard로 모니터링

학습 중 TensorBoard로 실시간 모니터링:

```bash
tensorboard --logdir ./runs
```

브라우저에서 `http://localhost:6006` 접속

### 모니터링 메트릭

- `train/loss`: 학습 손실
- `train/iou`: 학습 IoU
- `train/epoch`: 현재 에포크
- `train/step_time`: 스텝당 소요 시간
- `val/loss`: 검증 손실
- `val/iou`: 검증 IoU

## 저장된 모델

학습 중 다음 모델들이 저장됩니다:

- `model_best.pt`: 최고 검증 IoU를 달성한 모델
- `model_XXXXXX.pt`: 주기적인 체크포인트 (XXXXXX는 iteration 번호)
- `model_final.pt`: 최종 학습 완료 모델

## 추론 (Inference)

학습된 모델로 추론하기:

```python
import torch
from src.models import compile_model

# 설정 (학습 시와 동일하게)
grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [4.0, 45.0, 1.0],
}

data_aug_conf = {
    'resize_lim': (0.9, 1.1),
    'final_dim': (128, 352),
    'rot_lim': (-5.4, 5.4),
    'H': 224,
    'W': 480,
    'rand_flip': True,
    'bot_pct_lim': (0.0, 0.0),
    'Ncams': 6,
}

# 모델 로드
model = compile_model(grid_conf, data_aug_conf, outC=1)
checkpoint = torch.load('runs/simbev/model_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.cuda()
model.eval()

# 추론
with torch.no_grad():
    preds = model(imgs, rots, trans, intrins, post_rots, post_trans)
```

## 메모리 최적화

GPU 메모리가 부족한 경우:

1. **배치 크기 줄이기**:
   ```bash
   --bsz 2  # 또는 1
   ```

2. **이미지 크기 줄이기**:
   ```bash
   --final_h 96 --final_w 256
   ```

3. **카메라 개수 줄이기**:
   ```bash
   --ncams 3  # 전방 카메라만 사용
   ```

4. **Gradient checkpointing** (코드 수정 필요):
   `src/models.py`에서 gradient checkpointing 활성화

## 성능 팁

### 데이터 로딩 속도 향상

- `--nworkers` 증가 (CPU 코어 수에 맞게)
- SSD 사용 권장
- 이미지를 미리 리사이즈하여 저장

### 학습 속도 향상

- 더 큰 배치 크기 사용 (GPU 메모리가 허용하는 선에서)
- Mixed precision training (AMP) 사용 (코드 수정 필요)
- Multi-GPU 학습 (코드 수정 필요)

## 하이퍼파라미터 튜닝

권장 시작점:

```bash
# 좋은 시작점
--lr 0.001
--bsz 4
--weight_decay 1e-7

# Learning rate scheduling (코드 수정 필요)
# - Cosine annealing
# - Step decay
# - ReduceLROnPlateau
```

## 문제 해결

### CUDA out of memory

```bash
# 배치 크기 줄이기
--bsz 1

# 이미지 크기 줄이기
--final_h 96 --final_w 256
```

### 데이터로더 오류

```bash
# 워커 수 줄이기 (디버깅용)
--nworkers 0
```

### Loss가 NaN

- Learning rate 낮추기: `--lr 1e-4`
- Gradient clipping 확인 (기본 5.0)
- 데이터 정규화 확인

## 예제 실험

### 빠른 테스트 (5분)

```bash
python train_simbev.py \
    --dataroot /path/to/simbev \
    --nepochs 1 \
    --bsz 2 \
    --nworkers 0 \
    --val_step 10 \
    --save_step 20
```

### 전체 학습 (몇 시간~하루)

```bash
python train_simbev.py \
    --dataroot /path/to/simbev \
    --nepochs 100 \
    --bsz 8 \
    --nworkers 8 \
    --lr 1e-3 \
    --logdir ./runs/full_training
```

## 참고 자료

- [SIMBEV_DATALOADER.md](SIMBEV_DATALOADER.md): 데이터로더 상세 설명
- [원본 LSS 논문](https://arxiv.org/abs/2008.05711)
- [LSS 공식 저장소](https://github.com/nv-tlabs/lift-splat-shoot)
