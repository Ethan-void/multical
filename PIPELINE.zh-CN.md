# Multical 四相机标定 Pipeline 使用说明

本文档对应以下两份配置：

- `configs/pipeline.20260804.yaml`：四台相机可通过同一批标定板观测建立完整外参关系时使用。
- `configs/pipeline.worldgroups.20260804.yaml`：四台相机只能分成两个局部双目组，再通过同一世界坐标系合并时使用。

两套流程都覆盖内参标定、相机外参、世界坐标定位、三维重建和精度报告。主要区别在于四台相机的外参如何建立。

## 1. 功能概览

### 1.1 全局四相机流程

适用配置：`configs/pipeline.20260804.yaml`

```text
四相机内参
    ↓
四相机联合外参
    ├── 内参独立验证
    ├── 外参独立验证
    ↓
世界控制点联合定位
    ↓
多视角三角化
    ↓
3D 精度验收 + 标定分析报告
```

适用条件：外参数据能够让 `cam0`、`cam1`、`cam2`、`cam3` 形成连通的共同观测图。四台相机不必在每一帧中同时看到标定板，但不同相机之间必须存在足够的同步共同观测，最终把四台相机连接成一个整体。

### 1.2 分组世界坐标流程

适用配置：`configs/pipeline.worldgroups.20260804.yaml`

```text
四相机内参
    ├── cam0 + cam1 局部双目标定
    └── cam2 + cam3 局部双目标定
             ↓
两个相机组分别定位到同一世界坐标系
             ↓
初始组合 worldgroups
             ↓
带局部双目先验的联合优化 worldgroupba
             ↓
跨组多视角三角化
             ↓
3D 精度验收 + 分组质量报告
```

适用条件：`cam0/cam1` 与 `cam2/cam3` 之间难以拍到可靠的共同标定板，但两个相机组都能观测到已测量的世界控制点。两个组必须使用完全相同的世界坐标定义和长度单位。


## 2. 环境和快速开始

在仓库根目录执行命令。项目使用 `uv` 启动 Python 环境；首次使用可创建虚拟环境并安装本仓库：

```bash
uv venv
uv pip install -e .
```

如需使用 `vis` 等可视化功能，可安装交互依赖：`uv pip install -e '.[interactive]'`。

先查看阶段并检查实际命令：

```bash
# pipeline 脚本默认使用分组配置
./pipeline list
./pipeline dry-run
```

运行完整流程：

```bash
# 分组世界坐标流程；这是 ./pipeline 的默认配置
./pipeline all
```

`all` 默认带 `--resume`。已成功且命令、输入、输出均未变化的阶段会自动跳过，可以安全地在失败或中断后重新执行。


## 3. 数据目录

### 3.1 全局四相机流程

默认数据目录和输出目录都是 `20260804/`：

```text
20260804/
├── intrinsic/
│   ├── cam0/ ... cam3/           # 各相机内参图片
│   └── intrinsic.json            # 生成的内参
├── extrinsic/
│   ├── cam0/ ... cam3/           # 同步外参图片
│   ├── calibration.json          # 生成的四相机标定
│   └── calibration.pkl           # 完整标定工作区
├── test_intrinsic/cam0/          # 独立内参验证图片
├── test_extrinsic/cam0/ ... cam3/# 独立外参验证图片
├── world/
│   ├── world_images/cam0/ ... cam3/
│   ├── world_markers.yaml        # 世界点与图片对应关系
│   └── world_extrinsic.json      # 生成的世界外参
├── observe/
│   ├── measured_points/cam0/ ... cam3/
│   └── measured_observations.yaml
├── measured_world_points.yaml    # 3D 验收点实测坐标
├── triangulation/
├── reports/
└── pipeline_state.json
```

相机目录中的同名图片被视为同一时刻，例如：

```text
extrinsic/cam0/000015.jpg
extrinsic/cam1/000015.jpg
```

内参图片不要求不同相机之间同名或同步；外参图片必须按同步帧使用相同文件名。

## 4. 全局四相机流程

### 4.1 `intrinsic`：计算内参和畸变

```bash
./pipeline stage intrinsic
```
对四台相机分别进行内参标定。每台相机最多选取 40 张图片，`intrinsic_error_limit: 0.5` 用于控制内参图像的误差筛选。

主要输出：

```text
20260804/intrinsic/intrinsic.json
```

### 4.2 `extrinsic`：联合计算四相机外参

```bash
./pipeline stage extrinsic --with-deps
```

该阶段读取四台相机的同步图片，固定内参，只优化相机和标定板位姿。`cam2` 是相机组内部基准，`soft_l1` 用于降低异常观测的影响，优化执行 3 轮。

主要输出：

```text
20260804/extrinsic/calibration.json
20260804/extrinsic/calibration.pkl
```

`master: cam2` 只定义相机组内部坐标基准，不是最终世界原点。

### 4.3 独立验证

```bash
./pipeline --config configs/pipeline.20260804.yaml stage validate_intrinsic
./pipeline --config configs/pipeline.20260804.yaml stage validate_extrinsic
```

- `validate_intrinsic` 使用 `test_intrinsic/cam0` 检查固定内参在新图片上的表现。
- `validate_extrinsic` 固定内参和相机位姿，只估计新图中的标定板位姿，用来检查四相机模型是否能解释独立数据。

验证图片不应与标定图片重复，否则不能反映泛化误差。

### 4.4 `world`：将相机组定位到世界坐标系

```bash
./pipeline --config configs/pipeline.20260804.yaml stage world --with-deps
```

`worldmulti` 联合使用 `world/world_markers.yaml` 中的多相机控制点，将整个相机组对齐到现场世界坐标系。RANSAC 阈值为 3 px，最终使用 `soft_l1` 优化。

主要输出：

```text
20260804/world/world_extrinsic.json
```

### 4.5 `triangulate`：重建世界三维坐标

```bash
./pipeline --config configs/pipeline.20260804.yaml stage triangulate --with-deps
```

读取固定相机参数、世界外参和人工像素观测。当前配置要求：

- 至少有满足条件的多相机射线；
- 最小射线夹角为 `8°`；
- 单个观测的重投影筛选阈值为 `1.5 px`；
- 使用 `soft_l1` 对每个三维点做非线性精修；
- 精修只改变目标点 XYZ，不改变相机参数。

输出：`20260804/triangulation/triangulation.json`。

### 4.6 `evaluate3d` 和 `analysis`：验收与报告

```bash
./pipeline --config configs/pipeline.20260804.yaml \
  --stage evaluate3d --with-deps --resume

./pipeline --config configs/pipeline.20260804.yaml stage analysis
```

`evaluate3d` 将三角化结果与 `measured_world_points.yaml` 对比，生成：

```text
20260804/reports/evaluation3d.json
20260804/reports/evaluation3d.xlsx
```

此配置中的平均、P95 和最大误差验收阈值仍为注释状态，因此默认只统计、不作正式通过/失败判定。需要按项目要求取消注释并填写 `max_mean_error`、`max_p95_error` 和 `max_error`。

`analysis` 汇总内参、外参和工作区中的逐观测质量，输出 `20260804/reports/calibration_analysis.xlsx`。

## 5. 分组世界坐标流程

### 5.1 `intrinsic`：四相机内参

```bash
./pipeline stage intrinsic
```

功能和全局流程相同，输出为 `20260804_worldgroups/intrinsic/intrinsic.json`。

### 5.2 `extrinsic_01`、`extrinsic_23`：局部双目标定

```bash
./pipeline stage local_stereo --with-deps
```

一条命令会按配置顺序执行同属 `local_stereo` 组的两个阶段：

| 阶段 | 相机 | 局部基准 | 输出目录 |
|---|---|---|---|
| `extrinsic_01` | `cam0`, `cam1` | `cam0` | `extrinsic_01/` |
| `extrinsic_23` | `cam2`, `cam3` | `cam2` | `extrinsic_23/` |

两组都固定内参并使用 `soft_l1`。`warmup_before_outlier_rejection: true` 会先做一次预优化，再根据残差拒绝异常观测，避免初始位姿较差时过早删除有效点。

执行器会从四相机 `intrinsic.json` 自动生成相应的双相机内参子集，保存在各输出目录的 `.pipeline_inputs/`，不会修改原始内参文件。

### 5.3 `world_01`、`world_23`：各组独立世界定位

```bash
./pipeline stage world_control --with-deps
```

两个局部相机组分别使用：

- `world/world_markers_01.yaml`：定位 `cam0/cam1`；
- `world/world_markers_23.yaml`：定位 `cam2/cam3`。

两份控制点文件必须使用相同的世界坐标系、原点、轴方向和 `world_units`。建议至少保留 4 个由两个组共同测量的世界控制点，并让控制点覆盖有效空间，不要集中在一条直线或一个小区域。

输出：

```text
20260804_worldgroups/world/group01.json
20260804_worldgroups/world/group23.json
```

### 5.4 `worldgroups`：生成初始全局结果

```bash
./pipeline stage worldgroups --with-deps
```

该阶段将两个互不重叠的局部相机组按各自的世界外参直接合并，并检查：

- 每台相机只属于一个组；
- 两个组覆盖内参文件中的全部相机；
- 组内相对位姿与局部标定结果一致；
- 相机模型和世界单位一致。

输出：

```text
20260804_worldgroups/world/world_extrinsic.initial.json
20260804_worldgroups/extrinsic/calibration.initial.json
```

这只是初始合并结果，正式使用应继续执行 `worldgroupba`。

### 5.5 `worldgroupba`：带组内先验的联合优化

```bash
./pipeline stage worldgroupba --with-deps
```

该阶段同时使用局部标定工作区、世界控制点和初始合并结果做约束 Bundle Adjustment。目标是在改善两个组对世界控制网拟合的同时，避免破坏已可靠标定的组内双目几何。

关键参数：

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `relative_rotation_sigma_deg` | `0.1` | 组内相对旋转先验尺度 |
| `relative_translation_sigma` | `0.02` | 组内相对平移先验尺度，单位同世界坐标 |
| `relative_prior_weight` | `5.0` | 未单独指定时的默认组内先验权重 |
| `relative_prior_weights` | `[2.0, 5.0]` | 与 `group01, group23` 对齐的分组权重 |
| `ransac_threshold` | `3.0 px` | 世界控制点质量诊断阈值 |
| `final_recheck_iterations` | `0` | 不执行额外的最终硬剔除轮次 |

`group01` 的权重较低，允许世界控制网对其做更多修正；`group23` 权重较高，更保守地保持局部双目关系。所有世界控制点仍以 `soft_l1` 参与优化，3 px 在当前设置下主要用于质量诊断。

正式输出：

```text
20260804_worldgroups/world/world_extrinsic.json
20260804_worldgroups/extrinsic/calibration.json
```

### 5.6 三角化、3D 验收和分组报告

```bash
./pipeline stage triangulate --with-deps
./pipeline stage evaluate3d --with-deps
./pipeline stage analyze_worldgroups --with-deps
```

分组配置的 `triangulate` 从 `intrinsic.json` 读取镜头参数，从最终 `world_extrinsic.json` 读取每台相机的世界位姿。这是有意设计，不需要把 `calibration.json` 作为三角化输入。

`evaluate3d` 的当前验收阈值为：

| 指标 | 阈值 |
|---|---:|
| 平均 3D 误差 | `≤ 0.05 m` |
| P95 3D 误差 | `≤ 0.10 m` |
| 最大 3D 误差 | `≤ 0.20 m` |

`analyze_worldgroups` 进一步检查：

- 两组局部外参 RMS；
- 世界控制点内点率和重投影 RMS；
- 两组共享世界控制点的一致性；
- 至少 3 个真正使用了跨组相机射线的独立 3D 验收点；
- 跨组 3D 的平均、P95 和最大误差。

报告输出：

```text
20260804_worldgroups/reports/evaluation3d.json
20260804_worldgroups/reports/evaluation3d.xlsx
20260804_worldgroups/reports/worldgroups_analysis.json
20260804_worldgroups/reports/worldgroups_analysis.xlsx
```

仅有较低的组内重投影误差，不能证明两个相机组已经正确合并；共享控制点和跨组独立 3D 点是必要的验收证据。

## 6. 世界控制点文件

两种流程的 `world_markers*.yaml` 使用相同格式：

```yaml
world_units: meters
cameras: [cam0, cam1]
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
  - name: "01_0_e"
    markers:
      - marker_id: 23
        occurrence: upper
        world_point: [0.000, 3.000, 1.700]
      - marker_id: 23
        occurrence: lower
        world_point: [0.000, 3.000, 0.500]
```

图片默认位于控制点文件同级的 `world_images/<camera>/<capture.name>.jpg`。缺少某台相机的图片时，该相机观测会被跳过，其他存在的图片仍可参与计算。

使用上下两个同 ID 标记时，`occurrence: upper/lower` 用于区分它们。`world_point` 必须是标记中心的实测世界坐标；支架倾斜、坐标抄错或把底座位置当成标记中心都会直接污染世界外参。

## 7. 人工标注和 3D 验收数据

`observe` 是交互阶段，两份配置都默认 `enabled: false`，因为仓库中已经存在 `measured_observations.yaml`。需要重新点击像素时单独运行：

```bash
# 分组配置
./pipeline stage observe

# 全局四相机配置
./pipeline --config configs/pipeline.20260804.yaml stage observe
```

默认打开四相机的 `000000.jpg`，连续保存为 `P01`、`P02`……。常用操作：

- 左键：设置或细调当前相机的点；
- 右键：删除当前相机的点；
- 方向键或 WASD：移动 1 px；
- Enter 或空格：保存当前点；
- `N`：跳过，`B`：返回，`R`：重置；
- `X`：删除当前点；
- `E`、`Q` 或 Esc：保存并退出。

每个 3D 点至少需要两个相机观测。分组质量验收还需要部分点同时使用两个不同相机组，例如一个点同时被 `cam1` 和 `cam3` 观测。

像素观测文件示例：

```yaml
frames:
  - frame: P01
    observations:
      cam0: [1792.45, 622.20]
      cam1: [919.39, 602.44]
```

实测世界坐标文件示例：

```yaml
coordinate_frame: world
world_units: meters
points:
  P01: [0.000, 4.115, 0.000]
```

两个文件中的点名必须一致。验收点应独立于世界标定控制点，否则只能证明模型拟合了训练数据，不能证明实际 3D 精度。

## 8. Pipeline 常用命令

```bash
# 列出阶段
./pipeline list

# 只打印命令，不执行
./pipeline dry-run

# 执行单个阶段；默认断点续跑
./pipeline stage worldgroupba

# 执行同一 group 下的多个阶段
./pipeline stage local_stereo

# 自动补齐前置依赖
./pipeline stage evaluate3d --with-deps

# 强制重跑，不使用已有成功记录
./pipeline stage worldgroupba --force

# 从指定阶段运行到指定阶段
./pipeline --stage all \
  --from-stage extrinsic_01 \
  --to-stage worldgroupba \
  --resume

# 一次选择多个阶段
./pipeline --stage triangulate,evaluate3d --resume

# 单独生成报告
./pipeline analyze intrinsic
./pipeline analyze extrinsic
./pipeline analyze worldgroups
```

`stage` 默认只执行选中的阶段，不会自动运行依赖。输入尚未生成时增加 `--with-deps`；输入已存在且只想重跑当前阶段时不要增加。

### 8.1 断点续跑规则

`--resume` 仅在以下条件全部满足时跳过阶段：

- 上次状态为成功；
- 实际命令和参数未变化；
- 主要输入文件或输入图片未变化；
- 所有预期输出仍然存在；
- 当前阶段不是交互阶段。

状态分别保存在：

```text
20260804/pipeline_state.json
20260804_worldgroups/pipeline_state.json
```

修改配置、输入图片或控制点后，输入指纹变化的阶段会重新运行。需要无条件重跑时使用 `--force`。

## 9. 修改为新数据集

先复制与现场相机关系匹配的配置：

```bash
# 四相机全局外参
cp configs/pipeline.20260804.yaml configs/pipeline.NEW.yaml

# 两个局部相机组
cp configs/pipeline.worldgroups.20260804.yaml \
  configs/pipeline.worldgroups.NEW.yaml
```

优先修改顶部变量：

```yaml
variables:
  dataset: NEW_DATASET
  boards: boards/charuco_1600x1200.yaml
  cameras: [cam0, cam1, cam2, cam3]
  output_root: outputs/NEW_DATASET
```

如果改变相机分组，还必须同步修改：

- `extrinsic_*` 的 `cameras` 和 `master`；
- `world_*` 使用的局部标定和控制点文件；
- `worldgroups`、`worldgroupba` 中的 `calibrations`、`workspaces`、`correspondences` 和 `group_names`；
- `relative_prior_weights`，其顺序必须与 `group_names` 一致；
- 世界控制点文件中的 `cameras`。

标定板配置 `boards/charuco_1600x1200.yaml` 当前使用 8×6 ChArUco、`4X4_1000` 字典、0.180 m 方格和 0.135 m 标记。实际板尺寸与配置不一致会造成整体尺度错误，换板后必须实测并更新配置。

## 10. 常见问题

### 输入不存在

Pipeline 会在执行前检查主要输入。确认数据目录和配置顶部的 `dataset`、`output_root`，或使用 `--with-deps` 先生成前置结果。

### 输出存在但阶段仍重跑

命令参数、输入文件时间戳、图片集合或状态记录发生变化时，`--resume` 会重新执行。这是正常行为。可以先用 `dry-run` 查看实际命令。

### 修改控制点后没有得到预期结果

强制重跑该阶段及其下游阶段，例如：

```bash
./pipeline stage world_01 --force
./pipeline --stage worldgroups,worldgroupba,triangulate,evaluate3d \
  --resume
```

### 重投影误差很低，但 3D 误差仍然较大

低重投影误差只表示像素观测与当前几何模型一致。射线夹角过小、控制点覆盖不足、世界坐标测量偏差或两个组之间的系统偏移，仍会产生明显 3D 误差。应同时查看 `evaluation3d` 和 `worldgroups_analysis`，尤其关注跨组 3D 点。

### macOS/OpenCV 出现 OpenCL 缓存并发错误

两份配置都设置了：

```yaml
env:
  OPENCV_OPENCL_RUNTIME: disabled
```

通过 Pipeline 执行时会自动生效。
