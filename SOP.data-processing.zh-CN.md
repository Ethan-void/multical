# 多相机数据整理与像素转 3D SOP

本文说明以下三个脚本的基本使用流程：

1. `merge_camera_sessions.py`：合并多次采集并统一帧编号。
2. `prepare_sparse_camera_dataset.py`：为缺失的相机帧生成纯灰占位图。
3. `pixel_to_3d.py`：将同步的多相机像素坐标重建为世界坐标。

建议始终在命令行中显式传入路径和相机名，不依赖脚本顶部的本地默认配置。

## 1. 合并多次采集

输入目录示例：

```text
raw_sessions/
├── capture1/
│   ├── cam0/
│   ├── cam1/
│   ├── cam4/
│   └── cam5/
└── capture2/
    ├── cam0/
    ├── cam1/
    ├── cam4/
    └── cam5/
```

同一次采集中，同步图片应在各相机目录中使用相同的相对路径和文件名。

先预检，不复制文件：

```bash
uv run python scripts/merge_camera_sessions.py raw_sessions \
  --output data/merged \
  --sessions capture1 capture2 \
  --cameras cam2 cam3 \
  --dry-run
```

确认统计数量正确后执行合并：

```bash
uv run python scripts/merge_camera_sessions.py raw_sessions \
  --output data/merged \
  --sessions capture1 capture2 \
  --cameras cam0 cam1 cam4 cam5
```

输出示例：

```text
data/merged/
├── cam0/000000.jpg
├── cam1/000000.jpg
├── cam4/000000.jpg
├── cam5/000000.jpg
└── merge_manifest.csv
```

注意：输出目录必须为空。`merge_manifest.csv` 记录每张输出图片的原始来源。若要求每帧在所有相机中都存在，可增加 `--require-complete`；如果下一步准备补帧，则不要增加该参数。

## 2. 补齐缺失帧

先检查缺失帧，不写入文件：

```bash
uv run python scripts/prepare_sparse_camera_dataset.py \
  --image_path data/merged \
  --cameras cam0 cam1 cam4 cam5 \
  --dry-run
```

执行补帧：

```bash
uv run python scripts/prepare_sparse_camera_dataset.py \
  --image_path data/merged \
  --cameras cam0 cam1 cam4 cam5
```

脚本会根据各相机文件名的并集，在缺帧位置创建灰度值为 127 的纯灰图片，并生成：

```text
data/merged/placeholder_manifest.json
```

占位图只用于保持多相机文件名和帧序号一致，不包含真实图像信息，也不会产生有效标定点或目标观测。

需要撤销补帧时执行：

```bash
uv run python scripts/prepare_sparse_camera_dataset.py \
  --image_path data/merged \
  --remove
```

脚本只删除清单中记录且内容未被修改的占位图，不会删除真实图片。

## 3. 多相机像素转世界 3D

执行前需要：

- 相机内参文件，例如 `intrinsic.json`。
- 世界外参文件，例如 `world_extrinsic.json`。
- 至少两个相机对同一时刻、同一目标的像素坐标。

### 单个点

```bash
uv run python scripts/pixel_to_3d.py \
  --world-extrinsics DATASET/world/world_extrinsic.json \
  --calibration DATASET/intrinsic/intrinsic.json \
  --pixel cam0 641.2 358.7 \
  --pixel cam1 522.8 361.1
```

成功时输出 JSON，其中主要字段为：

- `point_world`：世界坐标 `[X, Y, Z]`。
- `cameras_used`：实际参与重建的相机。
- `cameras_rejected`：因误差过大被剔除的相机。
- `reprojection_rms_px`：重投影 RMS，单位为像素。
- `max_ray_angle_deg`：最大三角化射线夹角。

### 单条轨迹

轨迹目录格式：

```text
traj_0001/
├── cam0/ball.csv
├── cam1/ball.csv
├── cam2/ball.csv
└── cam3/ball.csv
```

每个 CSV 至少包含：

```text
Frame,Visibility,X,Y
```

其中 `Frame` 是同步帧号，`Visibility` 表示目标是否可见，`X`、`Y` 是目标中心像素坐标。

执行：

```bash
uv run python scripts/pixel_to_3d.py \
  --world-extrinsics DATASET/world/world_extrinsic.json \
  --calibration DATASET/intrinsic/intrinsic.json \
  --trajectory-dir trajectories/traj_0001 \
  --camera-map cam2 cam4 \
  --camera-map cam3 cam5 \
  --output-txt results/traj_0001/points_3d.txt
```

只有输入目录相机名和标定文件相机名不一致时才需要 `--camera-map`。

### 批量处理多条轨迹

```bash
uv run python scripts/pixel_to_3d.py \
  --world-extrinsics DATASET/world/world_extrinsic.json \
  --calibration DATASET/intrinsic/intrinsic.json \
  --dataset-dir trajectories \
  --output-dir results \
  --camera-map cam2 cam4 \
  --camera-map cam3 cam5
```

脚本会扫描 `trajectories/traj_*/cam*/ball.csv`，并为每条轨迹生成 `points_3d.txt`。输出中的 `status=failed` 表示该帧可用相机不足、重投影误差过大、深度无效或射线夹角过小；此类帧的坐标为 `nan`，不应作为有效 3D 点使用。
