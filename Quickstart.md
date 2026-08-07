数据准备

内外参独立标定

```text
20260731/
├── boards.yaml
├── intrinsic/
│   ├── CamA/000001.jpg
│   └── CamB/000001.jpg
└── extrinsic/
    ├── CamA/000001.jpg
    └── CamB/000001.jpg
```

内外参一起标定

```text
20260731/
├── CamA/
│   ├── 000001.jpg
│   └── ...
├── CamB/
└── CamC/
```

世界外参：world_markers.yaml


标定独立内参
uv run multical intrinsic \
--image_path 20260731/intrinsic \
--boards 20260731/boards.yaml \
--cameras cam0 cam1 cam2 cam3 \
--limit_intrinsic 30

limit_intrinsic：用于内参求解的最大图片数量


标定独立外参
uv run multical calibrate \
--image_path 20260731/extrinsic \
--boards 20260731/boards.yaml \
--cameras cam0 cam1 cam2 cam3 \
--calibration 20260731/intrinsic/intrinsic.json \
--fix_intrinsic \
--master cam0 \
--loss soft_l1 \
--iter 3

loss soft_l1：使用鲁棒损失，降低大误差点对优化的影响
iter：联合优化、异常点筛选和重新优化的轮数


验证集评估内参
uv run multical calibrate --image_path validation_intrinsic_cam0 --boards cali_split_20260727/boards.yaml --cameras cam0 --calibration 20260729/intrinsic.json --fix_intrinsic --master cam0 --loss soft_l1 --iter 1 --output_path 20260729/validation_results/intrinsic_cam0 --name validation

验证集评估外参
uv run multical calibrate --image_path validation_extrinsic --boards cali_split_20260727/boards.yaml --cameras cam0 cam1 cam2 cam3 --calibration 20260729/calibration.json --fix_intrinsic --fix_camera_poses --master cam0 --loss soft_l1 --iter 3 --output_path 20260729/validation_results/entrinsic --name validation


整理内外参评估结果
uv run python scripts/analyze_calibration.py \
  --intrinsic 20260731/intrinsic/intrinsic.json \
  --extrinsic 20260731/extrinsic/calibration.json \
  --output 20260731_calibration_analysis.xlsx

  
标定世界外参 (world_markers.yaml,多相机用worldmulti)
uv run multical world \
--calibration 20260731/extrinsic/calibration.json \
--correspondences 20260731/world_markers_multicam.yaml \
--ransac_threshold 2.0

ransac_threshold：判断控制点是否为内点的像素阈值
loss soft_l1：


点击提取像素 脚本，之后用检测模型
uv run multical observe \
--image_path 20260731/measured_points \
--cameras cam0 cam1 cam2 cam3 \
--output 20260731/measured_observations.yaml \
--frame 000000.jpg \
--columns 2 \
--tile_width 640


3D重建 
uv run multical triangulate \
--calibration 20260731/calibration.json \
--world_extrinsics 20260731/world_extrinsics_multicam.json \
--observations 20260731/measured_observations.yaml \
--output 20260731/triangulation.json \
--reprojection_threshold 1.5 \
--min_ray_angle_deg 8.0

reprojection_threshold：3D重建时，某相机观测允许的最大重投影误差
min_ray_angle_deg：三角化时，两条相机射线允许的最小夹角


评估3D重建
uv run multical evaluate3d \
--reconstruction 20260731/triangulation.json \
--ground_truth 20260731/measured_world_points.yaml \
--output 20260731/evaluation3d.json