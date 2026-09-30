# 多相机分组世界对齐标定 SOP

本文档记录一套可重复执行的多相机标定流程，覆盖数据采集、内参、相机间外参、世界坐标外参、三角化和 3D 精度验收。

所有长度统一使用米，图像坐标使用像素。

## 快速开始

编辑 `configs/pipeline.init.yaml`，按本次采集设置初始化参数。例如，四相机分成 `01`、`23` 两组：

```yaml
dataset: DATASET
# output_root: DATASET  # 可选，默认与 dataset 相同
boards: boards/extrinsic/charuco_1600x1200/charuco_1600x1200.yaml
world_board: boards/world/aruco_600x600/aruco_600x600.yaml
auto_analyze: true
court: badminton
groups:
  "01": [cam0, cam1]
  "23": [cam2, cam3]
```

- `dataset`：本次采集的数据目录；`output_root` 可另设为结果目录。
- `boards`：内外参标定板配置；`world_board`：世界标记的 ID、中心高度等配置，均须与实际标定板一致。
- `court`：`tennis`（网球场）或 `badminton`（羽毛球场），省略时默认 `tennis`。
- `groups`：相机分组；相机列表由各组自动推导，组名用于生成阶段名及文件名。
- `auto_analyze: true`：生成内外参标定完成后自动执行诊断报告的配置。

初始化目录和实验配置：

```bash
./pipeline init
```

该命令创建数据目录结构及 `configs/pipeline.worldgroups.DATASET.yaml`，并将其设为 `./pipeline` 的默认配置。

默认配置保存在项目根目录的 `pipeline` 脚本顶部，`CONFIG=${MULTICAL_PIPELINE_CONFIG:-...}` 中的 `...` 即默认配置文件路径。可直接修改该路径，也可通过以下命令切换默认配置：

```bash
./pipeline use configs/pipeline.worldgroups.DATASET.yaml
```

一般指令可在子命令前加 `--config`（简写 `-c`）指定本次使用的配置文件，不会修改默认配置，例如：

```bash
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml list
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage intrinsic
./pipeline -c configs/pipeline.worldgroups.DATASET.yaml all
```

省略 `--config` 时使用默认配置；若设置了环境变量 `MULTICAL_PIPELINE_CONFIG`，则优先使用该变量指定的配置。优先级为：命令行 `--config` > 环境变量 > 脚本内默认路径。

`DATASET` 对应 `dataset` 路径的最后一级目录名。重复初始化会补齐目录并保留已有实验配置；只有需要重新生成配置时才使用 `./pipeline init --force`，该选项会覆盖已有实验配置。

以上述 `dataset: DATASET` 为例，初始化后创建：

```text
项目根目录/
├── configs/
│   └── pipeline.worldgroups.DATASET.yaml  # 本次实验的完整流水线配置
└── DATASET/
    ├── intrinsic/                       # 各相机内参采集图片
    ├── extrinsic/                       # 各组相机同步外参图片
    ├── world/
    │   └── world_images/                # 世界控制点图片
    └── observe/
        └── measured_points/             # 独立 3D 验收点图片
```

初始化只创建以上基础目录和实验配置。放入图片时，在各图片目录下按实际相机名创建 `cam0/`、`cam1/` 等子目录，例如 `DATASET/intrinsic/cam0/000001.jpg`。世界点 YAML 通过后文的世界坐标编辑器保存；标定结果、`reports/` 等由对应阶段执行后生成，完整输入输出结构见第 2 节。

将采集图片放入对应目录，检查生成配置中的路径、分组及参数，再预览并开始内参标定：

```bash
./pipeline list
./pipeline dry-run
./pipeline stage intrinsic
```

后续按正文逐阶段执行。本文命令统一显式指定 `configs/pipeline.worldgroups.DATASET.yaml`；使用初始化后选定的默认配置时，可省略 `--config`。仓库当前初始化配置使用 `01 / 24 / 35` 三组，正文使用 `01 / 23` 两组举例，请按实际分组替换阶段后缀和相机名。

## 1. 流程总览

```text
布置相机，准备ChArUco标定板
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
worldgroups：合并各相机组，生成统一世界外参初值
        ↓
worldgroupba：联合优化各组相机位姿和世界控制点误差
        ↓
observe：采集独立的 3D 验收点并标注像素
        ↓
triangulate：重建世界三维坐标
        ↓
evaluate3d：与验收点实测坐标比较并生成验收报告
```

内参、相机间外参和世界外参是三个不同问题：

- 内参描述单台相机的焦距、主点和畸变。
- 相机间外参描述相机之间的固定相对位姿。
- 世界外参把整个相机组对齐到现场定义的世界坐标系。

## 2. 推荐目录结构

每次新采集使用一个独立的数据目录，例如 `DATASET/`：

```text
DATASET/
├── intrinsic/                         # 内参输入及输出
│   ├── cam0/
│   ├── cam1/
│   ├── cam2/
│   ├── cam3/
│   ├── intrinsic.json                 # intrinsic 输出
│   └── distortion_check/
├── extrinsic/                         # 分组外参输入及最终合并输出
│   ├── cam0/
│   ├── cam1/
│   ├── cam2/
│   ├── cam3/
│   ├── calibration.initial.json       # worldgroups 输出
│   └── calibration.json               # worldgroupba 最终输出
├── extrinsic_01/                      # cam0-cam1 分组标定输出
│   ├── calibration.json
│   ├── calibration.pkl
│   └── distortion_check/
├── extrinsic_23/                      # cam2-cam3 分组标定输出
│   ├── calibration.json
│   ├── calibration.pkl
│   └── distortion_check/
├── world/                             # 世界控制点输入及世界外参输出
│   ├── world_images/
│   │   ├── cam0/
│   │   ├── cam1/
│   │   ├── cam2/
│   │   └── cam3/
│   ├── world_markers_01.yaml
│   ├── world_markers_23.yaml
│   ├── group01.json                   # world_01 输出
│   ├── group23.json                   # world_23 输出
│   ├── world_extrinsic.initial.json   # worldgroups 输出
│   ├── world_extrinsic.json           # worldgroupba 最终输出
│   └── check/
├── observe/                           # 独立验收点图片和像素标注
│   ├── measured_points/
│   │   ├── cam0/
│   │   ├── cam1/
│   │   ├── cam2/
│   │   └── cam3/
│   └── measured_observations.yaml
├── test_intrinsic/                    # 可选：独立内参验证图片
├── test_extrinsic/                    # 可选：独立外参验证图片
├── measured_world_points.yaml         # 3D 验收点实测真值
├── triangulation/
│   └── triangulation.json
├── reports/
│   ├── intrinsic_analysis.xlsx
│   ├── extrinsic_analysis_01.xlsx
│   ├── extrinsic_analysis_23.xlsx
│   ├── worldgroups_analysis.xlsx
│   └── evaluation3d.json
└── pipeline_state.json                # Pipeline 断点续跑状态
```

如果不希望原图和输出放在同一级目录，可把配置中的 `dataset` 和 `output_root` 设为不同路径。

## 3. 标定板配置

本流程使用 ChArUco 标定板( boards/charuco_1600x1200.yaml )，配置文件为：

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

正式采集前可检查单张图片的检测效果：

```bash
uv run multical boards \
  --boards boards/charuco_1600x1200.yaml \
  --detect DATASET/intrinsic/cam0/000001.jpg
```

## 4. 采集并标定相机内参

内参图片由每台相机独立使用，不要求四个相机的文件名或帧数一致。

每台相机建议采集至少 30  张有效图片，并满足：

- 标定板覆盖画面中心、四角和边缘。
- 包含近、中、远不同距离。
- 包含水平、俯仰、偏航等不同姿态。
- 标定板清晰、无运动模糊、不过曝。
- 避免大量几乎相同的连续图片。
- 画面边缘也要有足够观测，否则畸变只能在中心区域拟合得好。

pipeline 参数：

```yaml
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
```

执行内参标定：

```bash
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage intrinsic
```

也可用配置文件名中的唯一关键词临时选择配置，例如 `./pipeline 0924 --stage intrinsic`。
先加上 `--dry-run` 可预览命令；关键词匹配多个配置时会报错并列出候选文件。

主要输出：

```text
DATASET/intrinsic/intrinsic.json
DATASET/intrinsic/distortion_check/
```

生成指标诊断报告：

已自动生成报告时无需重复执行；需要单独生成或重新生成时执行：

```bash
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage analyze_intrinsic
```

检查项目：

- 整体 RMS 和平均 RMS 小于 3 px。
- 垂直水平覆盖率大于 80%。
- `distortion_check/` 中直线去畸变后是否自然。

## 5. 采集并标定相机间外参

分成若干组双目，按组分别采集组内共同帧，假设有四目，分成两组，每组独立，

```
组1：
cam0—cam1

组2：
cam2—cam3
```

### 5.1 图片命名规则

同组内相机必须同步采集，同组内不同相机目录中必须使用相同文件名：

```text
cam0-cam1双目组：
extrinsic/cam0/000015.jpg
extrinsic/cam1/000015.jpg

cam2-cam3双目组：
extrinsic/cam2/000019.jpg
extrinsic/cam3/000019.jpg
```

### 5.2 采集要求

每组相机对建议采集至少 25 张有效图片，并满足：

- 图片必须同步。
- 标定目标要覆盖共同视域的不同位置、距离、高度和角度。
- 不要只沿一条直线移动；空间分布应形成有宽度、有高度变化的结构。
- 标定板距离不能太远

pipeline 参数：

```yaml
extrinsic_01:
    command: calibrate
    group: local_stereo
    needs: [intrinsic]
    args:
      image_path: "{dataset}/extrinsic"
      boards: "{boards}"
      cameras: [cam0, cam1]
      calibration: "{output_root}/intrinsic/intrinsic.json"
      fix_intrinsic: true
      master: cam0
      loss: soft_l1
      warmup_before_outlier_rejection: true
      iter: 3
      output_path: "{output_root}/extrinsic_01"
      name: calibration

extrinsic_23:
    command: calibrate
    group: local_stereo
    needs: [intrinsic]
    args:
      image_path: "{dataset}/extrinsic"
      boards: "{boards}"
      cameras: [cam2, cam3]
      calibration: "{output_root}/intrinsic/intrinsic.json"
      fix_intrinsic: true
      master: cam2
      loss: soft_l1
      warmup_before_outlier_rejection: true
      iter: 3
      output_path: "{output_root}/extrinsic_23"
      name: calibration
```

`master` 只定义相机组内部坐标基准，不等于最终世界坐标原点。最终世界坐标由 `worldmulti` 决定。

分组执行：

```bash
# cam0-cam1的外参标定
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage extrinsic_01

# cam2-cam3的外参标定
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage extrinsic_23
```

主要输出：

```text
cam0-cam1：
DATASET/extrinsic_01/calibration.json
DATASET/extrinsic_01/calibration.pkl
DATASET/extrinsic_01/calibration.txt
DATASET/extrinsic_01/distortion_check/
```

生成指标诊断报告：

已自动生成报告时无需重复执行；需要单独生成或重新生成时执行：

```bash
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage analyze_extrinsic_01

./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage analyze_extrinsic_23
```

检查项目：

* 内点 RMS 小于 0.3 px，共同位姿大于 25 帧
* 旋转离散度小于 0.5，平移离散度小于 0.15

## 6. 采集世界坐标控制点

以四目为例，按 cam0-cam1 和 cam2-cam3 相机组分别采集共同帧。

### 6.1 世界标定支架

世界坐标阶段使用两个编码标记固定在同一个刚性支架上：

- 上标记中心高度：`1.700 m`
- 下标记中心高度：`0.500 m`
- 两个中心应位于同一条铅垂线上。

世界点的 XY 应测量标记中心铅垂投影的位置，不能测量底座边缘后直接当作标记中心。场地标线的设计坐标也不能替代现场实测。

### 6.2 控制点分布

推荐控制点满足：

- 覆盖整个有效工作区域。
- 同一相机组观测点形成二维分布，避免全部共线。
- 同时包含纵向和横向跨度。
- 上下标记提供高度差，多个地面位置提供 XY 分布。
- 近端、中部和远端均有控制点。
- 每组相机间尽量有更多的共享世界控制点

### 6.3  `world_markers.yaml` 准备

`世界外参的图片路径由配置文件提供，所以先准备配置文件，这里以 cam0-cam1 组 world_markers_01.yaml示例`

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
  - name: "000001"
    markers:
      - marker_id: 23
        occurrence: upper
        world_point: [0.000, 3.000, 1.700]
      - marker_id: 23
        occurrence: lower
        world_point: [0.000, 3.000, 0.500]

  - name: "000002"
    markers:
      - marker_id: 23
        occurrence: upper
        world_point: [0.000, -3.000, 1.700]
      - marker_id: 23
        occurrence: lower
        world_point: [0.000, -3.000, 0.500]
```

`以 name: "000001" 为例，默认根据 capture.name 在以下位置查找图片：`

```
world/world_images/cam0/000001.jpg
world/world_images/cam1/000001.jpg
```

* 世界坐标图片不要求四个相机目录完全一致。某个位置只有实际能看到标记的相机需要放图片，找不到的相机会记录为 `image_not_found`，其余图片仍然正常使用。
* 同一世界位置如果由不同相机分开拍摄，只有在支架中心位置和高度完全不动时才能写为同一个世界坐标。旋转标定板时必须绕标记中心轴旋转；如果移动了支架，应作为新的 `capture` 并重新测量坐标。能够同步拍摄时优先同步拍摄。

同一个 `marker_id` 在画面中出现两次时，`occurrence: upper` 和 `occurrence: lower` 用于区分上下两个标记。

质量参数含义：

| 参数                    | 作用                               |
| ----------------------- | ---------------------------------- |
| `mode: reject`        | 自动拒绝不满足硬阈值的单次标记观测 |
| `min_edge_px`         | 标记最短边低于该像素数时拒绝       |
| `warn_edge_px`        | 标记偏小时给出警告，但仍可参与计算 |
| `min_side_ratio`      | 四边长度差异过大时拒绝             |
| `min_area_ratio`      | 四边形面积相对边长过小时拒绝       |
| `max_view_angle_deg`  | 斜视角超过该值时拒绝               |
| `warn_view_angle_deg` | 斜视角偏大时给出警告               |

质量预筛只作用于世界坐标标记观测，不会改变此前的内参或相机间外参标定。

### 6.4 世界坐标编辑器：`worldpoints_marker_01`

可用世界坐标编辑器录入、修改第 6.3 节的控制点文件：

```bash
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage worldpoints_marker_01
```

该阶段使用 `mode: markers`，读取 `world/world_images` 中的图片和 `world_board` 定义的标记 ID、中心高度，在场地视图中编辑各采集位置的世界坐标，保存到 `world/world_markers_01.yaml`。检查采集名称与图片文件名一致，坐标以现场实测值为准；场地线和吸附点仅辅助定位。

其他相机组使用对应后缀，例如 `worldpoints_marker_23`。初始化会按 `groups` 自动生成这些阶段；它们设置为 `enabled: false`、`interactive: true`，执行 `all` 时跳过，需要用 `stage` 显式打开。

编辑器通过 `court` 切换网球场或羽毛球场，坐标原点位于近端底线中心，X 沿场地长度方向、Y 沿宽度方向、Z 向上，单位为米。切换场地不会自动换算已保存坐标。

## 7. 标定世界外参

pipeline 参数：

```yaml
world_01:
    command: worldmulti
    group: world_control
    needs: [extrinsic_01]
    args:
      calibration: "{output_root}/extrinsic_01/calibration.json"
      correspondences: "{dataset}/world/world_markers_01.yaml"
      output: "{output_root}/world/group01.json"
      ransac_threshold: 3.0
      loss: soft_l1
```

`ransac_threshold` 的单位是像素：

- `2.0 px` 更严格，适合数据质量很高、约束充足时做最终检查。
- `3.0 px` 对实际采集的小幅检测和测量误差更稳健，适合作为当前流程的标定阈值。
- 提高阈值会增加内点数量，但不会让原本错误的世界坐标变正确。

执行：

```bash
# cam0-cam1
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage world_01

# cam2-cam3
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage world_23
```

主要输出：

```text
DATASET/world/group01.json
DATASET/world/group23.json
DATASET/world/check/
```

`check/` 按相机输出标记中心、重投影位置、误差和质量状态，是判断哪些照片需要重拍的首要依据。

### 7.1 `worldgroups`：合并各相机组

`worldmulti` 只得到每个相机组各自的世界外参。`worldgroups` 将两个组的结果合并，生成包含全部相机的统一世界外参初值和相机间外参初值。

pipeline 参数：

```yaml
worldgroups:
  command: worldgroups
  group: reconstruction
  needs: [world_01, world_23]
  args:
    intrinsic: "{output_root}/intrinsic/intrinsic.json"
    calibrations:
      - "{output_root}/extrinsic_01/calibration.json"
      - "{output_root}/extrinsic_23/calibration.json"
    world_extrinsics:
      - "{output_root}/world/group01.json"
      - "{output_root}/world/group23.json"
    group_names: [group01, group23]
    master: cam0
    output: "{output_root}/world/world_extrinsic.initial.json"
    calibration_output: "{output_root}/extrinsic/calibration.initial.json"
```

执行：

```bash
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage worldgroups
```

主要输出：

```text
DATASET/world/world_extrinsic.initial.json
DATASET/extrinsic/calibration.initial.json
```

这些文件只是联合优化的初始值，不建议直接作为最终生产标定结果。

### 7.2 `worldgroupba`：分组联合优化

`worldgroupba` 同时使用各组的外参观测、世界控制点和组内相机相对位姿，对全部相机进行联合优化。相对位姿先验用于限制优化偏离已经标定好的组内结构。

对于 `20260907_badminton`，配置在 `01 / 24 / 35` 基础上增加了 `45`（cam4、cam5）：

- `extrinsic_45` 从独立目录 `extrinsic_images_45/cam4`、`extrinsic_images_45/cam5` 读取两台相机的同步外参图像（不混入原 `extrinsic` 目录），`world_45` 使用 `world/world_markers_45.yaml`；标注入口为 `worldpoints_marker_45`。
- `worldgroups.args.allow_overlap: true` 允许相机跨组出现，并保留列表中首次出现的相机位姿作为初值。因此将冗余组放在原三组之后；该输出仅用于初始化。
- `worldgroupba` 遇到重叠组时，每台相机只优化一套世界位姿，各组仍使用自己的标定板帧位姿和相对位姿先验。新增组的 calibration、workspace、correspondences、group_names 和 relative_prior_weights 必须一一对应。
- `45` 将 `24` 与 `35` 连接起来；`01` 仍通过世界控制点定位。需要有效的共同可见标定板观测，不能仅凭增加组数认定精度提高。重复使用的世界点标注也不代表新增独立测量。

补齐各组世界点标注后，运行 `./pipeline all`。比较增加冗余组前后独立测量点的 `evaluate3d` 误差，判断是否保留该约束。

pipeline 参数：

```yaml
worldgroupba:
  command: worldgroupba
  group: reconstruction
  needs: [worldgroups]
  args:
    intrinsic: "{output_root}/intrinsic/intrinsic.json"
    initial_world_extrinsics: "{output_root}/world/world_extrinsic.initial.json"
    calibrations:
      - "{output_root}/extrinsic_01/calibration.json"
      - "{output_root}/extrinsic_23/calibration.json"
    workspaces:
      - "{output_root}/extrinsic_01/calibration.pkl"
      - "{output_root}/extrinsic_23/calibration.pkl"
    correspondences:
      - "{dataset}/world/world_markers_01.yaml"
      - "{dataset}/world/world_markers_23.yaml"
    group_names: [group01, group23]
    master: cam0
    ransac_threshold: 3.0
    loss: soft_l1
    relative_rotation_sigma_deg: 0.1
    relative_translation_sigma: 0.02
    relative_prior_weight: 5.0
    relative_prior_weights: [2.0, 2.0]
    final_recheck_iterations: 0
    output: "{output_root}/world/world_extrinsic.json"
    calibration_output: "{output_root}/extrinsic/calibration.json"
```

执行：

```bash
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage worldgroupba
```

最终输出：

```text
DATASET/world/world_extrinsic.json
DATASET/extrinsic/calibration.json
DATASET/world/check/final/
```

后续三角化必须使用 `worldgroupba` 输出的最终 `world_extrinsic.json`，不要使用 `group01.json`、`group23.json` 或 `world_extrinsic.initial.json`。

## 8. 采集独立 3D 验收点

3D 验收点必须独立于世界外参控制点，否则只能证明模型能拟合参与标定的数据，不能证明现场泛化精度。

推荐覆盖：

- 两端底线和边线附近。
- 两个半场中部。
- 四相机交叠区域。
- 不同高度，包括地面、腰部和较高位置。

“单图多点” ：一张图里有多个3D验收点，measured_world_points.yaml 真值文件示例：

```yaml
coordinate_frame: world
world_units: meters

points:
  P01: [0.000, 4.115, 0.000]
  P02: [0.000, -4.115, 0.000]
  P03: [5.485, 0.000, 0.000]
  P04: [14.051, 0.000, 0.700]
```

“多图单点” ：每张图里只有一个3D验收点，measured_world_points.yaml 真值文件示例：

```yaml
coordinate_frame: world
world_units: meters

points:
  "000000.jpg": [0.000, 0.2, 0.7]
  "000001.jpg": [5.485, 4.315, 0.7]
  "000002.jpg": [8.485, 0.2, 0.7]
  "000003.jpg": [6.485, -3.915, 0.7]
  "000004.jpg": [18.298, -3.915, 0.7]
```

## 9. 标注像素观测

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

pipeline 配置：

`frame: 000000.jpg `单图多点模式，注释后开启多图单点模式

`enabled: false` 仅表示执行 `all` 时跳过交互窗口，仍然可以用 `stage observe` 单独运行。

执行：

```bash
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage observe
```

点击要求：

- 每个 P 点在所有可见相机中点击同一个物理位置。
- 看不见或被遮挡的相机不要猜测点击。
- 每个点至少需要两个有效相机观测。
- 放大确认目标中心，避免一个相机点底部、另一个相机点顶部。

### 9.1 世界坐标编辑器：`worldpoints_measured`

完成 `observe` 像素标注后，可通过编辑器录入“单图多点”或“多图单点”验收点的实测世界坐标。两种模式均使用同一个阶段 `worldpoints_measured`，编辑器根据观测文件中的点名及来源图片自动识别：

```bash
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage worldpoints_measured
```

该阶段使用 `mode: measured`，读取 `observe/measured_observations.yaml` 中的点名及观测，选择验收点、核对对应图片中的像素标注，再录入实测 X、Y、Z，保存到 `measured_world_points.yaml`，供 `evaluate3d` 作为真值使用。点名必须与观测文件一致；这里编辑的是世界坐标，像素位置仍通过 `observe` 标注。

两种模式的操作方式：

- **单图多点**：在 `observe.args` 中设置 `frame: 000000.jpg`，使用各相机的同名图片标注多个验收点（如 `P01`、`P02`）。保存像素观测后打开上述编辑器，逐个选择点名，核对同一帧中的对应标注并填写各点实测坐标。真值文件以 `P01`、`P02` 等点名为键，示例见第 8 节。
- **多图单点**：省略或注释 `observe.args.frame`，逐帧标注每张图片中的一个验收点。保存后打开上述编辑器，逐个选择图片对应的点并填写实测坐标；默认以 `000000.jpg`、`000001.jpg` 等帧名为真值文件的键。

模式由 `observe` 的采集与标注方式决定，无须为世界坐标编辑器另设模式参数；两种情况均保持 `mode: measured`。

该交互阶段同样不会随 `all` 自动打开，应在执行最终验收前单独完成。验收点须保持独立，不要将世界标定控制点或三角化结果作为实测真值。

## 10. 三角化

```yaml
triangulate:
    command: triangulate
    group: reconstruction
    needs: [worldgroupba]
    args:
      calibration: "{output_root}/intrinsic/intrinsic.json"
      world_extrinsics: "{output_root}/world/world_extrinsic.json"
      observations: "{dataset}/observe/measured_observations.yaml"
      output: "{output_root}/triangulation/triangulation.json"
      reprojection_threshold: 3.0
      min_ray_angle_deg: 8.0
      refine: true
      refine_loss: soft_l1
```

pipeline 参数：

- `reprojection_threshold` 用于拒绝与最终三维点不一致的相机观测。
- `min_ray_angle_deg` 防止使用夹角太小、深度极不稳定的相机射线。
- `refine` 只优化目标点 XYZ，不会修改标定好的相机参数。

执行：

```bash
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage triangulate
```

输出：

```text
DATASET/triangulation/triangulation.json
```

低重投影误差不必然代表世界坐标误差小。远距离或射线夹角不理想时，即使重投影只有约 1 px，也可能产生数厘米甚至更大的三维误差。

## 11.最终 3D 精度验收

pipeline 参数：

```yaml
evaluate3d:
    command: evaluate3d
    group: report
    needs: [triangulate]
    args:
      reconstruction: "{output_root}/triangulation/triangulation.json"
      ground_truth: "{dataset}/measured_world_points.yaml"
      output: "{output_root}/reports/evaluation3d.json"
      max_mean_error: 0.05
      max_p95_error: 0.10
      max_error: 0.20
```

执行：

```bash
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage evaluate3d
```

输出：

```text
DATASET/reports/evaluation3d.json
DATASET/reports/evaluation3d.xlsx
```

当前流程的一组参考结果为：

| 指标         |     结果 |
| ------------ | -------: |
| 实测点       |       15 |
| 成功重建     |       15 |
| 平均三维误差 |  3.16 cm |
| RMS          |  4.20 cm |
| 中位数       |  2.61 cm |
| P95          |  7.33 cm |
| 最大误差     | 11.24 cm |
| 不超过 5 cm  |    12/15 |

## 12. 生成标定分析报告

```bash
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml analyze intrinsic
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml analyze extrinsic
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml analyze all
```

输出位于：

```text
DATASET/reports/intrinsic_analysis.xlsx
DATASET/reports/extrinsic_analysis.xlsx
DATASET/reports/calibration_analysis.xlsx
```

## 13. 验证集验证内外参

验证图片不应与正式标定图片重复。

```bash
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage validate_intrinsic
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage validate_extrinsic
```

外参验证阶段固定内参和相机位姿，只估计标定板姿态：

```yaml
fix_intrinsic: true
fix_camera_poses: true
```

因此验证误差反映的是新图片与固定相机模型的一致性，不会通过重新优化相机参数把问题掩盖掉。

## 14. 一键运行和断点续跑

先检查配置和即将执行的命令：

```bash
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml dry-run
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml list
```

执行所有已启用阶段：

```bash
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml all
```

流水线会记录：

- 上次阶段是否成功。
- 实际命令和参数。
- 主要输入文件和图片目录的状态。
- 预期输出是否仍然存在。

输入、参数和输出都没有变化时会自动跳过。常用命令：

```bash
# 强制重跑单个阶段
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml stage world_01 --force

# 从外参开始运行，到世界外参结束
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml \
  --stage all --from-stage extrinsic_01 --to-stage world_01 --resume

# 只执行三角化和验收
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml \
  --stage triangulate,evaluate3d --resume

# 同时补齐所选阶段的前置依赖
./pipeline --config configs/pipeline.worldgroups.DATASET.yaml \
  --stage evaluate3d --with-deps --resume
```

* 不加 `--resume`：选中的阶段全部重新运行。
* 加 `--resume`：跳过已成功且结果仍有效的阶段，从未完成或已变化的位置继续。#
