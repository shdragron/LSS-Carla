# LSS with SimBEV Dataset

이 저장소는 원본 [Lift-Splat-Shoot (LSS)](https://github.com/nv-tlabs/lift-splat-shoot) 모델을 SimBEV 데이터셋으로 학습할 수 있도록 확장한 버전입니다.

## 빠른 시작

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt
```

필요한 주요 패키지:
- PyTorch
- torchvision
- numpy
- pillow
- tensorboardX

### 2. 데이터셋 준비

SimBEV 데이터셋을 다음 구조로 준비:

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

자세한 내용은 [SIMBEV_DATALOADER.md](SIMBEV_DATALOADER.md) 참고.

### 3. 학습 실행

```bash
# 1. 설정 파일에서 데이터 경로 수정
vim configs/simbev_default.sh  # DATAROOT 수정

# 2. 학습 시작
bash configs/simbev_default.sh
```

또는 직접 실행:

```bash
python train_simbev.py \
    --dataroot /path/to/simbev/dataset \
    --gpuid 0 \
    --nepochs 100 \
    --bsz 4 \
    --logdir ./runs/my_experiment
```

### 4. 모니터링

```bash
tensorboard --logdir ./runs
```

## 주요 파일

### 데이터로더
- **[src/data_simbev.py](src/data_simbev.py)**: SimBEV 데이터로더 구현
- **[SIMBEV_DATALOADER.md](SIMBEV_DATALOADER.md)**: 데이터로더 상세 문서

### 학습
- **[train_simbev.py](train_simbev.py)**: SimBEV 학습 스크립트
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**: 학습 가이드 (자세한 설명)
- **[configs/simbev_default.sh](configs/simbev_default.sh)**: 기본 학습 설정
- **[configs/simbev_small.sh](configs/simbev_small.sh)**: 테스트용 설정

### 테스트
- **[test_simbev_loader.py](test_simbev_loader.py)**: 데이터로더 테스트

## 문서

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**: 학습 실행 방법, 하이퍼파라미터, 문제 해결
- **[SIMBEV_DATALOADER.md](SIMBEV_DATALOADER.md)**: 데이터 형식, 데이터로더 API

## 예제

### 빠른 테스트

```bash
python train_simbev.py \
    --dataroot /path/to/simbev \
    --nepochs 1 \
    --bsz 2 \
    --val_step 10
```

### 전체 학습

```bash
bash configs/simbev_default.sh
```

### 학습 재개

```bash
python train_simbev.py \
    --dataroot /path/to/simbev \
    --resume ./runs/simbev/model_010000.pt \
    --logdir ./runs/simbev
```

## LSS 모델 구조

```
Input: 멀티 카메라 이미지 (N cameras)
  ↓
Camera Encoder (EfficientNet-B0)
  ↓
Lift: Image features → 3D frustum
  ↓
Splat: 3D features → BEV grid (voxel pooling)
  ↓
BEV Encoder (ResNet-18)
  ↓
Output: BEV segmentation map
```

## 모델 입력/출력

### 입력
- `imgs`: (B, N, 3, H, W) - 카메라 이미지
- `rots`: (B, N, 3, 3) - 카메라→ego 회전
- `trans`: (B, N, 3) - 카메라→ego 이동
- `intrins`: (B, N, 3, 3) - 카메라 내부 파라미터
- `post_rots`: (B, N, 3, 3) - 후처리 회전
- `post_trans`: (B, N, 3) - 후처리 이동

### 출력
- `preds`: (B, 1, H_bev, W_bev) - BEV 세그멘테이션 맵

## 설정

### 기본 이미지 설정
- 원본: 224×480
- 타겟: 128×352
- 카메라: 6개 (front_left, front, front_right, back_left, back, back_right)

### 기본 BEV 그리드
- X: [-50, 50] m, 해상도 0.5m → 200 bins
- Y: [-50, 50] m, 해상도 0.5m → 200 bins
- Z: [-10, 10] m, 해상도 20m → 1 bin (collapse)
- Depth: [4, 45] m, 해상도 1m → 41 bins

### 하이퍼파라미터
- Batch size: 4
- Learning rate: 1e-3
- Weight decay: 1e-7
- Optimizer: Adam
- Loss: Binary Cross Entropy with pos_weight=2.13

## GPU 메모리 요구사항

- **최소**: 6GB (batch_size=1, 카메라 3개)
- **권장**: 12GB+ (batch_size=4, 카메라 6개)
- **최적**: 24GB+ (batch_size=8+, 카메라 6개)

메모리 부족 시 해결 방법:
```bash
# 배치 크기 줄이기
--bsz 2

# 이미지 크기 줄이기
--final_h 96 --final_w 256

# 카메라 개수 줄이기
--ncams 3
```

## 성능 벤치마크

예상 학습 시간 (대략적):
- **1 epoch**: 10-30분 (데이터셋 크기에 따라)
- **100 epochs**: 17-50시간
- **Inference**: ~100ms/sample (single GPU)

## CVT와의 차이점

SimBEV 데이터를 사용하지만, Cross View Transformer와는 다른 모델입니다:

| | LSS | CVT |
|---|---|---|
| **아키텍처** | Lift-Splat-Shoot | Transformer |
| **특징 표현** | 3D Voxel Pooling | Cross-attention |
| **백본** | EfficientNet-B0 | Various |
| **BEV 인코더** | ResNet-18 | CNN |

## 라이센스

원본 LSS 코드는 [NVIDIA Source Code License](https://github.com/nv-tlabs/lift-splat-shoot/blob/master/LICENSE)를 따릅니다.

## 인용

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

## 문제 해결

문제가 발생하면 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)의 "문제 해결" 섹션을 참고하세요.

주요 문제:
- CUDA out of memory → 배치 크기/이미지 크기 줄이기
- 데이터로더 오류 → 데이터 경로 및 형식 확인
- Loss가 NaN → learning rate 낮추기

## 참고 자료

- [원본 LSS 논문](https://arxiv.org/abs/2008.05711)
- [LSS GitHub](https://github.com/nv-tlabs/lift-splat-shoot)
- [SimBEV 데이터셋](https://github.com/OpenDriveLab/Birds-eye-view-Perception)
