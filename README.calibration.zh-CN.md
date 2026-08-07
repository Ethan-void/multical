# 四相机完整标定与 3D 验收流程

本文档记录一套可重复执行的四相机标定流程，覆盖数据采集、内参、相机间外参、世界坐标外参、三角化和 3D 精度验收。

适用相机为 `cam0`、`cam1`、`cam2`、`cam3`。所有长度统一使用米，图像坐标使用像素。

## 1. 流程总览

```text
准备标定板和测量基准
        ↓
采集各相机内参图片
        ↓
intrinsic：计算内参和畸变
        ↓
采集同步的多相机外参图片
        ↓
calibrate：固定内参，计算相机间外参
        ↓
在已知世界坐标位置采集标记图片
        ↓
worldmulti：联合计算相机组到世界坐标系的变换
        ↓
采集独立的 3D 验收点并标注像素
        ↓
triangulate：重建世界三维坐标
        ↓
evaluate3d：与实测坐标比较并生成验收报告
```

内参、相机间外参和世界外参是三个不同问题：

- 内参描述单台相机的焦距、主点和畸变。
- 相机间外参描述四台相机之间的固定相对位姿。
- 世界外参把整个相机组对齐到现场定义的世界坐标系。

## 2. 推荐目录结构

每次新采集使用一个独立的数据目录，例如 `DATASET/`：

```text
DATASET/
├── intrinsic/
│   ├── cam0/
│   ├── cam1/
│   ├── cam2/
│   └── cam3/
├── extrinsic/
│   ├── cam0/
│   ├── cam1/
│   ├── cam2/
│   └── cam3/
├── test_intrinsic/
│   └── cam0/
├── test_extrinsic/
│   ├── cam0/
│   ├── cam1/
│   ├── cam2/
│   └── cam3/
├── world/
│   ├── world_images/
│   │   ├── cam0/
│   │   ├── cam1/
│   │   ├── cam2/
│   │   └── cam3/
│   └── world_markers.yaml
├── observe/
│   ├── measured_points/
│   │   ├── cam0/
│   │   ├── cam1/
│   │   ├── cam2/
│   │   └── cam3/
│   └── measured_observations.yaml
├── measured_world_points.yaml
├── intrinsic/               # 同时也是内参输出目录
├── extrinsic/               # 同时也是相机间外参输出目录
├── triangulation/
└── reports/
```

如果不希望原图和输出放在同一级目录，可把配置中的 `dataset` 和 `output_root` 设为不同路径。

## 3. 标定板配置

本流程使用 ChArUco 标定板，配置文件为：

```yaml
boards:
  charuco_1600x1200:
    _type_: charuco
    size: [8, 6]
    aruco_dict: 4X4_1000
    aruco_offset: 0
    square_length: 0.180
    marker_length: 0.135
    min_rows: 3
    min_points: 20
```

打印或制作完成后，应实测方格边长。`square_length` 和 `marker_length` 必须填写实际尺寸，不能只使用设计尺寸。板面必须平整、刚性固定，避免翘曲。

正式采集前可检查单张图片的检测效果：

```bash
uv run multical boards \
  --boards boards/charuco_1600x1200.yaml \
  --detect DATASET/intrinsic/cam0/000001.jpg
```

## 4. 采集内参图片

内参图片由每台相机独立使用，不要求四个相机的文件名或帧数一致。

每台相机建议采集至少 30～40 张有效图片，并满足：

- 标定板覆盖画面中心、四角和边缘。
- 包含近、中、远不同距离。
- 包含水平、俯仰、偏航等不同姿态。
- 标定板清晰、无运动模糊、不过曝。
- 避免大量几乎相同的连续图片。
- 画面边缘也要有足够观测，否则畸变只能在中心区域拟合得好。

执行内参标定：

```bash
./pipeline --config configs/pipeline.DATASET.yaml stage intrinsic
```

主要输出：

```text
DATASET/intrinsic/intrinsic.json
DATASET/intrinsic/distortion_check/
```

检查项目：

- 每台相机 RMS 和逐图误差是否稳定。
- 是否有少数图片明显高于其他图片。
- `distortion_check/` 中直线去畸变后是否自然。
- 标定板是否覆盖整个有效画面，而不只是中心。

## 5. 采集并标定相机间外参

外参数据用于求四台相机之间的固定相对位姿。相机必须与后续正式使用时保持完全相同的位置、焦距和对焦状态。

### 5.1 图片命名规则

同一次同步采集在不同相机目录中必须使用相同文件名：

```text
extrinsic/cam0/000015.jpg
extrinsic/cam1/000015.jpg
extrinsic/cam2/000015.jpg
extrinsic/cam3/000015.jpg
```

程序根据文件名认定这些图片属于同一时刻。缺少某台相机的图片时，该帧只使用实际存在的相机；但整个相机观测关系必须连通。

### 5.2 采集要求

- 多相机图片必须同步。
- 相机固定，标定支架移动。
- 共同观测的相机必须看到同一个刚性标定目标的同一姿态。
- 标定目标要覆盖共同视域的不同位置、距离、高度和角度。
- 不要只沿一条直线移动；空间分布应形成有宽度、有高度变化的结构。
- 每个相机对需要多组有效共同观测，不能只依赖一两帧桥接。
- 四台相机的观测图必须连通，否则会形成互不相关的坐标子系统。

执行：

```bash
./pipeline --config configs/pipeline.DATASET.yaml stage extrinsic
```

该阶段固定已经得到的内参，仅优化相机位姿：

```yaml
calibration: "{output_root}/intrinsic/intrinsic.json"
fix_intrinsic: true
master: cam2
loss: soft_l1
iter: 3
```

`master` 只定义相机组内部坐标基准，不等于最终世界坐标原点。最终世界坐标由 `worldmulti` 决定。

主要输出：

```text
DATASET/extrinsic/calibration.json
DATASET/extrinsic/calibration.pkl
DATASET/extrinsic/calibration.txt
DATASET/extrinsic/distortion_check/
```

## 6. 独立验证内参和相机间外参

验证图片不应与正式标定图片重复。

```bash
./pipeline --config configs/pipeline.DATASET.yaml stage validate_intrinsic
./pipeline --config configs/pipeline.DATASET.yaml stage validate_extrinsic
```

外参验证阶段固定内参和相机位姿，只估计标定板姿态：

```yaml
fix_intrinsic: true
fix_camera_poses: true
```

因此验证误差反映的是新图片与固定相机模型的一致性，不会通过重新优化相机参数把问题掩盖掉。

## 7. 采集世界坐标控制点

### 7.1 世界标定支架

世界坐标阶段使用两个编码标记固定在同一个刚性支架上：

- 上标记中心高度：`1.700 m`
- 下标记中心高度：`0.500 m`
- 两个中心应位于同一条铅垂线上。

每次放置必须确认支架竖直。支架倾斜时，上下中心的 XY 不再相同，会直接引入错误的世界坐标约束。

世界点的 XY 应测量标记中心铅垂投影的位置，不能测量底座边缘后直接当作标记中心。场地标线的设计坐标也不能替代现场实测。

### 7.2 控制点分布

推荐控制点满足：

- 覆盖整个有效工作区域，不只集中在中心。
- 同一相机组观测点形成二维分布，避免全部共线。
- 同时包含纵向和横向跨度。
- 上下标记提供高度差，多个地面位置提供 XY 分布。
- 近端、中部和远端均有控制点。
- 每个桥接相机对至少保留 3～4 个质量良好的位置。
- 整个观测关系必须连接四台相机。

当前布局使用四组共同观测关系：

```text
cam0—cam1
cam2—cam3
cam1—cam3
cam0—cam2
```

它们构成一个连通且有冗余回路的相机图。若现场共同视域发生变化，应按真实可见关系调整，不能仅修改组名。

### 7.3 世界图片规则

默认根据 `capture.name` 在以下位置查找图片：

```text
world/world_images/cam0/捕获名称.jpg
world/world_images/cam1/捕获名称.jpg
world/world_images/cam2/捕获名称.jpg
world/world_images/cam3/捕获名称.jpg
```

世界坐标图片不要求四个相机目录完全一致。某个位置只有实际能看到标记的相机需要放图片，找不到的相机会记录为 `image_not_found`，其余图片仍然正常使用。

同一世界位置如果由不同相机分开拍摄，只有在支架中心位置和高度完全不动时才能写为同一个世界坐标。旋转标定板时必须绕标记中心轴旋转；如果移动了支架，应作为新的 `capture` 并重新测量坐标。能够同步拍摄时优先同步拍摄。

### 7.4 `world_markers.yaml` 示例

```yaml
world_units: meters
cameras: [cam0, cam1, cam2, cam3]
marker_family: 6X6_250
image_path: world_images

marker_quality:
  mode: reject
  min_edge_px: 15
  warn_edge_px: 25
  min_side_ratio: 0.20
  min_area_ratio: 0.15
  max_view_angle_deg: 60
  warn_view_angle_deg: 45

captures:
  - name: "01_0"
    markers:
      - marker_id: 23
        occurrence: upper
        world_point: [0.000, 3.000, 1.700]
      - marker_id: 23
        occurrence: lower
        world_point: [0.000, 3.000, 0.500]
```

同一个 `marker_id` 在画面中出现两次时，`occurrence: upper` 和 `occurrence: lower` 用于区分上下两个标记。

质量参数含义：

| 参数 | 作用 |
|---|---|
| `mode: reject` | 自动拒绝不满足硬阈值的单次标记观测 |
| `min_edge_px` | 标记最短边低于该像素数时拒绝 |
| `warn_edge_px` | 标记偏小时给出警告，但仍可参与计算 |
| `min_side_ratio` | 四边长度差异过大时拒绝 |
| `min_area_ratio` | 四边形面积相对边长过小时拒绝 |
| `max_view_angle_deg` | 斜视角超过该值时拒绝 |
| `warn_view_angle_deg` | 斜视角偏大时给出警告 |

质量预筛只作用于世界坐标标记观测，不会改变此前的内参或相机间外参标定。

## 8. 标定世界外参

执行：

```bash
./pipeline --config configs/pipeline.DATASET.yaml stage world
```

关键参数：

```yaml
command: worldmulti
args:
  calibration: "{output_root}/extrinsic/calibration.json"
  correspondences: "{dataset}/world/world_markers.yaml"
  output: "{output_root}/world/world_extrinsic.json"
  ransac_threshold: 3.0
  loss: soft_l1
```

`ransac_threshold` 的单位是像素：

- `2.0 px` 更严格，适合数据质量很高、约束充足时做最终检查。
- `3.0 px` 对实际采集的小幅检测和测量误差更稳健，适合作为当前流程的标定阈值。
- 提高阈值会增加内点数量，但不会让原本错误的世界坐标变正确。

主要输出：

```text
DATASET/world/world_extrinsic.json
DATASET/world/check/
```

`check/` 按相机输出标记中心、重投影位置、误差和质量状态，是判断哪些照片需要重拍的首要依据。

修改 `world_markers.yaml` 后，如果流水线提示输出仍是最新状态，可强制重跑：

```bash
./pipeline --config configs/pipeline.DATASET.yaml stage world --force
```

不要为了让指标好看而直接删除所有外点。先判断原因：

- 模糊、标记过小、严重斜视：重拍图片。
- 支架倾斜：扶正后重拍。
- 上下残差方向一致：优先复测该位置的 XY。
- 同一相机对多个位置都呈系统偏差：检查相机间外参和畸变覆盖。
- 桥接组有效内点不足：补拍新的桥接位置，不能只删除失败位置。

## 9. 采集独立 3D 验收点

3D 验收点必须独立于世界外参控制点，否则只能证明模型能拟合参与标定的数据，不能证明现场泛化精度。

推荐覆盖：

- 两端底线和边线附近。
- 两个半场中部。
- 四相机交叠区域。
- 不同高度，包括地面、腰部和较高位置。

实测真值示例：

```yaml
coordinate_frame: world
world_units: meters

points:
  P01: [0.000, 4.115, 0.000]
  P02: [0.000, -4.115, 0.000]
  P03: [5.485, 0.000, 0.000]
  P04: [14.051, 0.000, 0.700]
```

同一验收点必须由至少两台相机在目标静止时同步观测。测量真值和图像点击必须对应目标的同一个物理中心。

## 10. 标注像素观测

开启配置中的 `observe` 阶段，或单独执行：

```bash
./pipeline --config configs/pipeline.DATASET.yaml stage observe
```

配置示例：

```yaml
observe:
  enabled: false
  interactive: true
  command: observe
  args:
    image_path: "{dataset}/observe/measured_points"
    cameras: [cam0, cam1, cam2, cam3]
    output: "{dataset}/observe/measured_observations.yaml"
    frame: 000000.jpg
    columns: 2
    tile_width: 640
```

`enabled: false` 仅表示执行 `all` 时跳过交互窗口，仍然可以用 `stage observe` 单独运行。

点击要求：

- 每个 P 点在所有可见相机中点击同一个物理位置。
- 看不见或被遮挡的相机不要猜测点击。
- 每个点至少需要两个有效相机观测。
- 放大确认目标中心，避免一个相机点底部、另一个相机点顶部。

## 11. 三角化

执行：

```bash
./pipeline --config configs/pipeline.DATASET.yaml stage triangulate
```

当前参数：

```yaml
reprojection_threshold: 1.5
min_ray_angle_deg: 8.0
refine: true
refine_loss: soft_l1
```

- `reprojection_threshold` 用于拒绝与最终三维点不一致的相机观测。
- `min_ray_angle_deg` 防止使用夹角太小、深度极不稳定的相机射线。
- `refine` 只优化目标点 XYZ，不会修改标定好的相机参数。

输出：

```text
DATASET/triangulation/triangulation.json
```

低重投影误差不必然代表世界坐标误差小。远距离或射线夹角不理想时，即使重投影只有约 1 px，也可能产生数厘米甚至更大的三维误差。

## 12. 3D 精度验收

执行：

```bash
./pipeline --config configs/pipeline.DATASET.yaml stage evaluate3d
```

输出：

```text
DATASET/reports/evaluation3d.json
DATASET/reports/evaluation3d.xlsx
```

如果不设置阈值，报告只能给出统计值，不能把“通过”当作正式验收结论。应根据项目要求在配置中明确填写：

```yaml
evaluate3d:
  args:
    max_mean_error: 0.05
    max_p95_error: 0.08
    max_error: 0.12
```

以上只是示例。若合同要求“每个点都小于 5 cm”，应设置 `max_error: 0.05`，不能只设置平均误差为 5 cm。

当前流程的一组参考结果为：

| 指标 | 结果 |
|---|---:|
| 实测点 | 15 |
| 成功重建 | 15 |
| 平均三维误差 | 3.16 cm |
| RMS | 4.20 cm |
| 中位数 | 2.61 cm |
| P95 | 7.33 cm |
| 最大误差 | 11.24 cm |
| 不超过 5 cm | 12/15 |

该结果表示平均精度达到 5 cm，但尚未达到全场所有点均不超过 5 cm。中心交叠区域可达到约 1 cm，较大误差主要出现在远端或边缘位置，因此验收必须同时查看平均值、P95、最大值和空间分布。

## 13. 生成标定分析报告

```bash
./pipeline --config configs/pipeline.DATASET.yaml analyze intrinsic
./pipeline --config configs/pipeline.DATASET.yaml analyze extrinsic
./pipeline --config configs/pipeline.DATASET.yaml analyze all
```

输出位于：

```text
DATASET/reports/intrinsic_analysis.xlsx
DATASET/reports/extrinsic_analysis.xlsx
DATASET/reports/calibration_analysis.xlsx
```

## 14. 一键运行和断点续跑

先检查配置和即将执行的命令：

```bash
./pipeline --config configs/pipeline.DATASET.yaml dry-run
./pipeline --config configs/pipeline.DATASET.yaml list
```

执行所有已启用阶段：

```bash
./pipeline --config configs/pipeline.DATASET.yaml all
```

流水线会记录：

- 上次阶段是否成功。
- 实际命令和参数。
- 主要输入文件和图片目录的状态。
- 预期输出是否仍然存在。

输入、参数和输出都没有变化时会自动跳过。常用命令：

```bash
# 强制重跑单个阶段
./pipeline --config configs/pipeline.DATASET.yaml stage world --force

# 从外参开始运行，到世界外参结束
./pipeline --config configs/pipeline.DATASET.yaml \
  --stage all --from-stage extrinsic --to-stage world --resume

# 只执行三角化和验收
./pipeline --config configs/pipeline.DATASET.yaml \
  --stage triangulate,evaluate3d --resume

# 同时补齐所选阶段的前置依赖
./pipeline --config configs/pipeline.DATASET.yaml \
  --stage evaluate3d --with-deps --resume
```

## 15. 通用 Pipeline 配置模板

将下面内容保存为 `configs/pipeline.DATASET.yaml`，然后修改 `dataset` 和 `output_root`：

```yaml
variables:
  dataset: DATASET
  boards: boards/charuco_1600x1200.yaml
  cameras: &cameras [cam0, cam1, cam2, cam3]
  output_root: DATASET

state_file: "{output_root}/pipeline_state.json"

env:
  OPENCV_OPENCL_RUNTIME: disabled

stages:
  intrinsic:
    command: intrinsic
    group: calibration
    args:
      image_path: "{dataset}/intrinsic"
      boards: "{boards}"
      cameras: *cameras
      limit_intrinsic: 40
      intrinsic_error_limit: 0.5
      output_path: "{output_root}/intrinsic"
      name: intrinsic

  extrinsic:
    command: calibrate
    group: calibration
    needs: [intrinsic]
    args:
      image_path: "{dataset}/extrinsic"
      boards: "{boards}"
      cameras: *cameras
      calibration: "{output_root}/intrinsic/intrinsic.json"
      fix_intrinsic: true
      master: cam2
      loss: soft_l1
      iter: 3
      output_path: "{output_root}/extrinsic"
      name: calibration

  validate_intrinsic:
    command: calibrate
    group: validation
    needs: [intrinsic]
    args:
      image_path: "{dataset}/test_intrinsic"
      boards: "{boards}"
      cameras: [cam0]
      calibration: "{output_root}/intrinsic/intrinsic.json"
      fix_intrinsic: true
      master: cam0
      loss: soft_l1
      iter: 1
      output_path: "{output_root}/test_intrinsic"
      name: validation_intrinsic

  validate_extrinsic:
    command: calibrate
    group: extrinsic_validation
    needs: [extrinsic]
    args:
      image_path: "{dataset}/test_extrinsic"
      boards: "{boards}"
      cameras: *cameras
      calibration: "{output_root}/extrinsic/calibration.json"
      fix_intrinsic: true
      fix_camera_poses: true
      master: cam0
      loss: soft_l1
      iter: 3
      output_path: "{output_root}/test_extrinsic"
      name: validation_extrinsic

  world:
    command: worldmulti
    group: reconstruction
    needs: [extrinsic]
    args:
      calibration: "{output_root}/extrinsic/calibration.json"
      correspondences: "{dataset}/world/world_markers.yaml"
      output: "{output_root}/world/world_extrinsic.json"
      ransac_threshold: 3.0
      loss: soft_l1

  observe:
    enabled: false
    interactive: true
    command: observe
    group: reconstruction
    args:
      image_path: "{dataset}/observe/measured_points"
      cameras: *cameras
      output: "{dataset}/observe/measured_observations.yaml"
      frame: 000000.jpg
      columns: 2
      tile_width: 640

  triangulate:
    command: triangulate
    group: reconstruction
    needs: [world]
    args:
      calibration: "{output_root}/extrinsic/calibration.json"
      world_extrinsics: "{output_root}/world/world_extrinsic.json"
      observations: "{dataset}/observe/measured_observations.yaml"
      output: "{output_root}/triangulation/triangulation.json"
      reprojection_threshold: 1.5
      min_ray_angle_deg: 8.0
      refine: true
      refine_loss: soft_l1

  evaluate3d:
    command: evaluate3d
    group: report
    needs: [triangulate]
    args:
      reconstruction: "{output_root}/triangulation/triangulation.json"
      ground_truth: "{dataset}/measured_world_points.yaml"
      output: "{output_root}/reports/evaluation3d.json"
      # 按项目验收标准设置：
      # max_mean_error: 0.05
      # max_p95_error: 0.08
      # max_error: 0.12

  analyze_intrinsic:
    enabled: false
    command: analyze
    group: intrinsic_analysis
    args:
      intrinsic: "{output_root}/intrinsic/intrinsic.json"
      output: "{output_root}/reports/intrinsic_analysis.xlsx"

  analyze_extrinsic:
    enabled: false
    command: analyze
    group: extrinsic_analysis
    args:
      extrinsic: "{output_root}/extrinsic/calibration.json"
      workspace: "{output_root}/extrinsic/calibration.pkl"
      output: "{output_root}/reports/extrinsic_analysis.xlsx"

  analysis:
    command: analyze
    group: report
    needs: [intrinsic, extrinsic]
    args:
      intrinsic: "{output_root}/intrinsic/intrinsic.json"
      extrinsic: "{output_root}/extrinsic/calibration.json"
      workspace: "{output_root}/extrinsic/calibration.pkl"
      output: "{output_root}/reports/calibration_analysis.xlsx"
```

## 16. 最终验收检查表

### 采集前

- [ ] 四台相机安装牢固，焦距、对焦和分辨率锁定。
- [ ] 标定板尺寸经过实测，板面平整。
- [ ] 世界原点、轴方向、单位和测量基准明确。
- [ ] 世界支架上下标记中心竖直对齐。

### 内参

- [ ] 每台相机都有近、中、远及画面边缘图片。
- [ ] 已检查高误差图片和去畸变预览。
- [ ] 验证图片未参与内参标定。

### 相机间外参

- [ ] 多相机帧严格同步并使用相同文件名。
- [ ] 共同观测图连接全部四台相机。
- [ ] 每条关键连接都有多组、非共线、不同距离的观测。
- [ ] 相机在之后没有移动或重新对焦。

### 世界外参

- [ ] 控制点坐标为现场实测值。
- [ ] 控制点覆盖全场并包含高度差。
- [ ] 支架竖直，图像清晰，标记尺寸和斜视角合格。
- [ ] 已检查 `world/check/` 中的拒绝、警告和残差方向。
- [ ] 桥接相机组有效内点数量充足。

### 3D 验收

- [ ] 验收点未用于世界外参标定。
- [ ] 每个点至少由两台相机同步观测。
- [ ] 验收点覆盖中心、边缘、近端、远端和不同高度。
- [ ] 已明确平均、P95 和最大误差阈值。
- [ ] 不只查看重投影像素误差，同时检查实际三维误差。

## 17. 常见问题

### 修改 YAML 后为什么没有重新运行？

先确认修改的是当前 `--config` 指向的数据文件。如果仍提示输出最新，使用 `--force` 强制重跑对应阶段。

### 世界图片是否必须四个相机完全一致？

不需要。只使用实际存在并成功检测到标记的图片。但每个世界位置至少要提供有效观测，整个四相机关系也必须连通。

### 标记严重斜视时程序会自动处理吗？

会根据 `marker_quality` 检查尺寸、形状和视角。`mode: reject` 会拒绝超过硬阈值的单次观测；警告项仍会保留，需要人工结合 `check/` 图片判断。

### 被拒绝的图片可以直接删除吗？

可以删除文件或从 YAML 中移除对应捕获，但删除不会增加约束。若该点属于桥接组或边缘区域，应先补拍替代位置，再删除无效图片。

### 为什么重投影误差小，3D 误差仍可能较大？

三角化深度还取决于距离和相机射线夹角。远端、夹角小或世界坐标实测不准时，小于 1～2 px 的重投影误差仍可能对应数厘米的三维偏差。

### 什么时候必须重新做全部标定？

相机位置、镜头焦距、对焦、分辨率或裁剪方式改变后，应至少重新做内参和相机间外参；相机组整体位置发生变化但内部相对位置不变时，可保留内参与相机间外参，重新标定世界外参并重新验收。
