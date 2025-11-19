# SimBEV DataLoader for LSS

이 문서는 LSS (Lift-Splat-Shoot) 모델에서 SimBEV 데이터셋을 사용하는 방법을 설명합니다.

## 데이터셋 구조

SimBEV 데이터셋은 다음과 같은 디렉토리 구조를 따라야 합니다:

```
dataroot/
├── train/
│   ├── scene_0001/
│   │   └── yaw0pitch0/
│   │       ├── meta.json
│   │       ├── images/
│   │       │   ├── front_left_0000.jpg
│   │       │   ├── front_0000.jpg
│   │       │   ├── front_right_0000.jpg
│   │       │   ├── back_left_0000.jpg
│   │       │   ├── back_0000.jpg
│   │       │   └── back_right_0000.jpg
│   │       └── labels/
│   │           └── bev_0000.npy
│   ├── scene_0002/
│   │   └── ...
│   └── ...
└── val/
    └── ...
```

## meta.json 형식

각 scene의 orientation 폴더 안에 있는 `meta.json` 파일은 다음과 같은 형식을 따릅니다:

```json
[
  {
    "token": "unique_sample_id_0000",
    "images": [
      "images/front_left_0000.jpg",
      "images/front_0000.jpg",
      "images/front_right_0000.jpg",
      "images/back_left_0000.jpg",
      "images/back_0000.jpg",
      "images/back_right_0000.jpg"
    ],
    "intrinsics": [
      [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      [...],
      [...]
    ],
    "extrinsics": [
      [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz], [0, 0, 0, 1]],
      [...],
      [...]
    ],
    "label": "labels/bev_0000.npy"
  },
  ...
]
```

### 필드 설명

- **token**: 샘플의 고유 식별자
- **images**: 6개 카메라 이미지의 상대 경로 리스트 (순서: front_left, front, front_right, back_left, back, back_right)
- **intrinsics**: 6개 카메라의 내부 파라미터 행렬 (3×3)
- **extrinsics**: 6개 카메라의 외부 파라미터 행렬 (4×4, ego→camera 변환)
- **label**: BEV 세그멘테이션 맵의 상대 경로

## 사용 방법

### 1. 기본 사용

```python
from src.data_simbev import compile_data

# 설정
dataroot = '/path/to/simbev/dataset'

data_aug_conf = {
    'H': 224,                    # 원본 이미지 높이
    'W': 480,                    # 원본 이미지 너비
    'final_dim': (128, 352),     # 타겟 이미지 크기
    'resize_lim': (0.9, 1.1),    # 리사이즈 범위
    'bot_pct_lim': (0.0, 0.0),   # 하단 크롭 비율
    'rot_lim': (-5.4, 5.4),      # 회전 범위 (도)
    'rand_flip': True,           # 랜덤 좌우 반전
    'Ncams': 6,                  # 사용할 카메라 개수 (6 이하)
}

grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],  # BEV x 범위 [최소, 최대, 해상도]
    'ybound': [-50.0, 50.0, 0.5],  # BEV y 범위
    'zbound': [-10.0, 10.0, 20.0], # BEV z 범위
    'dbound': [4.0, 45.0, 1.0],    # 깊이 범위
}

# 데이터로더 생성
trainloader, valloader = compile_data(
    version='unused',
    dataroot=dataroot,
    data_aug_conf=data_aug_conf,
    grid_conf=grid_conf,
    bsz=4,
    nworkers=4,
    parser_name='segmentationdata'  # 또는 'vizdata'
)
```

### 2. LSS 모델과 함께 사용

```python
from src.models import compile_model
from src.data_simbev import compile_data

# 데이터로더 생성
trainloader, valloader = compile_data(...)

# 모델 생성
model = compile_model(grid_conf, data_aug_conf, outC=1)
model.cuda()

# 학습
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(trainloader):
        imgs, rots, trans, intrins, post_rots, post_trans, binimg = batch

        # GPU로 이동
        imgs = imgs.cuda()
        rots = rots.cuda()
        trans = trans.cuda()
        intrins = intrins.cuda()
        post_rots = post_rots.cuda()
        post_trans = post_trans.cuda()
        binimg = binimg.cuda()

        # Forward pass
        preds = model(imgs, rots, trans, intrins, post_rots, post_trans)

        # Loss 계산 및 backward
        loss = criterion(preds, binimg)
        loss.backward()
        optimizer.step()
```

## 데이터 반환 형식

### SegmentationData
학습/검증용 데이터셋. 각 배치는 다음을 반환합니다:

- `imgs`: (B, N, 3, H, W) - 카메라 이미지
- `rots`: (B, N, 3, 3) - 카메라→ego 회전 행렬
- `trans`: (B, N, 3) - 카메라→ego 이동 벡터
- `intrins`: (B, N, 3, 3) - 카메라 내부 파라미터
- `post_rots`: (B, N, 3, 3) - 후처리 회전 행렬
- `post_trans`: (B, N, 3) - 후처리 이동 벡터
- `binimg`: (B, 1, H_bev, W_bev) - BEV 세그멘테이션 맵

여기서:
- B = 배치 크기
- N = 카메라 개수 (보통 6)
- H, W = 이미지 높이, 너비
- H_bev, W_bev = BEV 맵 높이, 너비

### VizData
시각화용 데이터셋. SegmentationData와 동일하지만 `lidar_data`가 추가됩니다:

- ... (위와 동일)
- `lidar_data`: (3, 0) - 빈 텐서 (SimBEV는 raw LiDAR 포인트를 포함하지 않음)

## CVT 데이터로더와의 차이점

이 데이터로더는 Cross View Transformer (CVT)의 SimBEV 데이터로더를 참고하여 LSS 모델에 맞게 수정되었습니다:

1. **반환 형식**: CVT는 Sample 객체를 반환하지만, LSS는 튜플을 반환
2. **Extrinsics 처리**: ego→camera 변환을 camera→ego로 변환
3. **데이터 증강**: LSS 스타일의 증강 파이프라인 사용
4. **BEV 라벨**: LSS 형식에 맞게 처리

## 주의사항

1. **카메라 순서**: 이미지, intrinsics, extrinsics는 모두 동일한 순서를 따라야 합니다
   - 순서: front_left, front, front_right, back_left, back, back_right

2. **Extrinsics**: ego→camera 변환 행렬로 저장되어야 합니다 (코드에서 자동으로 역변환)

3. **이미지 경로**: meta.json의 이미지 경로는 meta.json이 있는 디렉토리 기준 상대 경로

4. **BEV 라벨**: numpy 배열(.npy) 또는 이미지 파일로 저장 가능

## 테스트

데이터로더를 테스트하려면:

```bash
python test_simbev_loader.py
```

테스트 스크립트를 실행하기 전에 `test_simbev_loader.py`의 `dataroot` 경로를 실제 데이터셋 경로로 수정하세요.
