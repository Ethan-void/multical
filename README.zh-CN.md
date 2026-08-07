# multical

使用一个或多个标定板进行多相机标定。

快速了解主要功能和流程，请参阅 [精简指南](README.quickstart.zh-CN.md)。

从数据采集、内参、相机间外参、世界外参到双目/多目世界三维重建的通用流程，请参阅 [端到端中文指南](README.end-to-end.zh-CN.md)。

![image](https://raw.githubusercontent.com/saulzar/multical/master/screenshots/image_view.png)
![image](https://raw.githubusercontent.com/saulzar/multical/master/screenshots/3d_view.png)

[更新日志](https://github.com/saulzar/multical/tree/master/example_boards)

## 安装

这个软件以 Python 库的形式发布，可以从 PyPI 安装：

```bash
pip install multical
```

安装后会提供一个同名命令行脚本：

```bash
multical
```

如果是在本仓库源码目录中开发或调试，也可以使用：

```bash
uv pip install -e .
```

推荐使用 `uv` 时，可以按下面方式创建虚拟环境并安装本项目：

```bash
cd /path/to/multical
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

如果遇到 OpenCV 与 NumPy 版本不兼容，可以固定一组较稳的依赖版本：

```bash
uv pip install -e . \
  "numpy==1.26.4" \
  "scipy==1.12.0" \
  "opencv-contrib-python==4.6.0.66"
```

## 运行 multical 应用

`multical` 脚本是运行标定应用的主要入口。它包装了几个子命令：

```text
usage: multical [-h] {calibrate,intrinsic,boards,show} ...
```

命令行参数可以通过子命令的帮助查看，例如：

```bash
multical calibrate --help
```

```bash
# uv
uv run multical --help
uv run multical calibrate --help
```

### 输入格式

默认输入方式是：每个相机单独一个文件夹，不同相机中对应图像使用相同文件名。例如：

```text
  - cam1
    - image01.jpg
    - image02.jpg
  - cam2
    - image01.jpg
    - image02.jpg
```

也可以手动指定相机名，并可选地指定目录结构模式：

```bash
multical calibrate --camera_pattern "{camera}/extrinsic" --cameras cam1 cam2 cam3
```

```bash
# uv
uv run multical calibrate --camera_pattern "{camera}/extrinsic" --cameras cam1 cam2 cam3
```

默认会搜索当前目录，也可以通过 `--image_path` 指定图像路径。

其中 `{camera}` 会被替换为相机名：

```text
  - cam1
    - intrinsic
       - image01.jpg
       - image02.jpg
    - extrinsic
       - image01.jpg
       - image02.jpg
  - cam2
    - intrinsic
       - image01.jpg
       - image02.jpg
    - extrinsic
       - image01.jpg
       - image02.jpg
  - cam3
    ...
```

初始内参标定时会选择固定数量的图像。可以通过 `--limit_intrinsic` 增加数量，以换取更高精度，但运行时间也会增加。

### 输出

`multical calibrate` 的输出会写入 `--output_path` 指定的目录；如果没有指定，默认写入图像路径。输出名称通过 `--name` 指定，默认是 `calibration`。

主要输出文件：

* `calibration.json`：相机摘要、内参、相机相对位姿、相机 rig 位姿以及最终联合优化质量。`quality` 中包含最终内点重投影误差 `RMS`（像素）、包含全部观测的 `RMS_all` 和对应的观测数量。
* `calibration.log`：标定过程日志。
* `calibration.detections.pkl`：缓存的标定板检测结果，让重复标定更快。
* `calibration.pkl`：序列化后的 workspace，包含可视化、恢复标定等所需的完整细节。

单独运行 `multical intrinsic` 时，`intrinsic.json` 中的每台相机还会包含
`quality`：记录 OpenCV 内参标定的整体 `RMS`、每视图平均/最大 RMS、实际使用
的视图数、图像数、角点观测数以及逐视图明细。后续将该文件传给
`multical calibrate --calibration ... --fix_intrinsic` 时，这些附加字段会被安全
忽略，不会影响内参或外参标定。

### 标定目标

当前支持的标定目标包括普通棋盘格、Charuco 标定板和 AprilGrid 标定板（Kalibr 使用的格式）。标定目标通过配置文件指定，参数为 `--boards`，示例可以在源码目录的 [example_boards](https://github.com/saulzar/multical/tree/master/example_boards) 中找到。普通棋盘格更适合单面单板；L 形或多面标定架应使用各面 ID 不重复的 Charuco 或 AprilGrid。

### 合成多相机标定数据集

项目提供多相机端到端数据集生成器，支持普通棋盘格、Charuco 和 AprilGrid：

```bash
uv run python tests/generate_calibration_simulation.py \
  --output-dir charuco_l_split \
  --board-type charuco \
  --board-style l \
  --dataset-mode split \
  --intrinsic-frames 20 \
  --extrinsic-frames 50
```

`--dataset-mode combined` 会把同步外参帧和逐相机内参增强帧放在同一数据集中，用于内外参同时标定；`split` 会分别生成 `intrinsic/` 和 `extrinsic/`。带编码的标定板支持 `plane`、`l` 和 `triangle`，普通棋盘格因为无法区分不同面，只支持 `plane`。每次生成结束后，数据集根目录还会自动创建 `overviews/` 文件夹并在其中生成分页概览图；每一行依次显示同一帧的各相机图片，全部图片都会按文件名顺序包含在概览中。缩略图尺寸为 `320×180`，每页显示10个同步帧。

标定完成后可执行 `commands.txt` 中附带的评估命令。评估脚本会比较 `calibration.json` 和仿真真值，在终端打印结果，并自动在 `ground_truth.json` 同级目录生成 `evaluation.json`。如需指定其他位置，可增加 `--output path/to/report.json`。

### 立体校正

可以针对一组或多组相机生成 OpenCV 立体校正参数和重映射表：

```bash
uv run multical rectify --calibration calibration.json --pairs CamA:CamB CamB:CamC --image_path images --frame frame_0000.jpg
```

默认输出到 `rectification/`：其中 `rectification.json` 保存每对相机的 `E/F/R1/R2/P1/P2/Q`、基线和有效区域，根级 `quality` 会从 `calibration.json` 带入最终标定 RMS。`E/F` 遵循文件中声明的左相机到右相机变换约定；这里的 RMS 是多相机全局联合优化的重投影 RMS，不是另行调用 `stereoCalibrate` 得到的双目对 RMS。每对相机还会生成一个压缩的 `*_maps.npz` 映射文件。如果提供 `--image_path`，还会输出左右校正样例图和带水平极线的并排预览图。默认 `--alpha -1` 让 OpenCV 自动选择缩放，`--alpha 0` 会裁掉无效黑边，`--alpha 1` 会尽量保留完整视场。运行时加载映射文件中的四个数组并传给 `cv2.remap` 即可。

### 世界坐标外参

Multical 默认得到相对于主相机的统一相机组位姿。若要将相机组对齐到用户定义的世界坐标，需要准备一个 JSON 或 YAML 控制点文件：

```yaml
camera: CamA
world_units: meters
world_points:
  - [0.0, 0.0, 0.0]
  - [10.97, 0.0, 0.0]
  - [0.0, 23.77, 0.0]
  - [10.97, 23.77, 0.0]
  - [5.485, 0.0, 0.0]
  - [5.485, 23.77, 0.0]
image_points:
  - [u0, v0]
  - [u1, v1]
  - [u2, v2]
  - [u3, v3]
  - [u4, v4]
  - [u5, v5]
```

`world_points` 和 `image_points` 必须逐行对应，然后运行：

```bash
uv run multical world --calibration calibration.json --correspondences world_points.yaml
```

脚本默认生成 `world_extrinsics.json`，其中每台相机都有 `world_to_camera`、`camera_to_world` 和 `position_world`，同时记录 PnP 内点数及重投影误差。最少需要4组点，实际建议使用8～20个分布在整个画面内、经过准确测量的控制点。

如果希望多台相机共同约束同一个世界变换，可使用独立命令
`worldmulti`。它不会调用或覆盖原有 `world` 流程，默认输出
`world_extrinsics_multicam.json`。相机内参和相机间外参保持固定，优化的
未知量只有一个公共的 `world_to_rig` 刚体变换；每个控制点只需被可见的
相机观测，不要求全部相机同时看见。

手工像素对应点配置示例：

```yaml
world_units: meters
observations:
  - camera: C1
    world_point: [0.0, 0.0, 0.5]
    image_point: [812.4, 603.1]
  - camera: C2
    world_point: [0.0, 0.0, 0.5]
    image_point: [421.7, 598.6]
  - camera: C1
    world_point: [1.0, 0.0, 0.5]
    image_point: [965.2, 601.8]
  - camera: C2
    world_point: [1.0, 0.0, 0.5]
    image_point: [573.6, 597.9]
```

自动检测多相机 ArUco 中心时，每次放置可以显式列出可用图像：

```yaml
world_units: meters
cameras: [C1, C2]
marker_family: 6X6_250
# 默认 reject。也可设为 warn（只报告）或 off（关闭）。
marker_quality:
  mode: reject
  min_edge_px: 15
  warn_edge_px: 25
  min_side_ratio: 0.20
  min_area_ratio: 0.15
  max_view_angle_deg: 60
  warn_view_angle_deg: 45
captures:
  - name: position_001
    images:
      C1: world_images/C1/position_001.jpg
      C2: world_images/C2/position_001.jpg
    markers:
      - marker_id: 23
        occurrence: upper
        world_point: [0.0, 0.0, 1.5]
      - marker_id: 23
        occurrence: lower
        world_point: [0.0, 0.0, 0.5]
```

未列出的相机不会参与该位置；图中没检测到的标签会记录在
`joint.skipped_observations`，不会使整批处理失败。执行：

```bash
uv run multical worldmulti --calibration calibration.json --correspondences world_multicam.yaml --ransac_threshold 3.0 --loss soft_l1
```

标记质量检查在世界外参优化前对每个上、下标记独立执行。硬阈值失败的
观测以 `marker_quality_rejected` 记录在 `joint.skipped_observations`；边界
观测保留并记为 `warning`。最终结果的 `joint.marker_quality` 汇总配置、
剔除数和警告数，验证图中的每个点也带有对应质量指标。命令行可用
`--marker_quality`、`--marker_min_edge_px`、`--marker_min_side_ratio`、
`--marker_min_area_ratio` 和 `--marker_max_view_angle` 覆盖 YAML 设置。

输出中的 `cameras` 与单相机 `world` 输出兼容，可直接传给
`multical triangulate --world_extrinsics world_extrinsics_multicam.json`。

### 世界3D坐标重建

生成 `world_extrinsics.json` 后，准备同一目标在同步相机图像中的像素观测：

```yaml
sequence: tennis-ball-001
frames:
  - frame: frame_0000
    timestamp: 0.000
    observations:
      CamA: [641.2, 358.7]
      CamB: [522.8, 361.1]
      CamC:
        point: [703.4, 349.8]
        confidence: 0.94
  - frame: frame_0001
    timestamp: 0.0083
    observations:
      CamA: [643.0, 355.2]
      CamB: [525.1, 357.6]
```

执行世界坐标三角化：

```bash
uv run multical triangulate --calibration calibration.json --world_extrinsics world_extrinsics.json --observations ball_observations.yaml
```

默认生成 `triangulation.json`。每个成功帧包含世界坐标 `[X,Y,Z]`、采用和剔除的相机、各相机重投影误差、重投影 RMS 以及最大视线夹角。每帧至少需要两台同步相机观测；有三台以上相机时，会根据 `--reprojection_threshold` 自动剔除不一致观测。观测不足、点落在相机后方、几何条件过差或视线夹角小于 `--min_ray_angle_deg` 的帧会明确标记为失败，不会输出不可靠坐标。

每个数据集都会包含：

* `boards.yaml`：可直接用于 Multical 的标定板配置；
* `ground_truth.json`：仿真真实内外参和多面板固定关系；
* `commands.txt`：可直接运行的标定命令。

AprilGrid 检测使用 OpenCV AprilTag 36h11 后端，不再依赖仅支持 Linux 的 `apriltags2-ethz`。

在正式标定前，建议先用一张图检查配置是否符合预期：

```bash
multical boards --boards my_board.yaml --detect my_image.jpeg
```

```bash
# uv
uv run multical boards --boards my_board.yaml --detect my_image.jpeg
```

### 输出可视化

如果需要运行可视化，需要安装额外依赖，主要包括 `qtpy` 和 `pyvistaqt`。可以安装 `interactive` 选项：

```bash
pip install "multical[interactive]"
```

```bash
# uv
uv pip install -e ".[interactive]"
```

也可以根据需要单独安装这些依赖，例如使用 conda。

可视化运行方式：

```bash
multical vis --workspace_file calibration.pkl
```

```bash
# uv
uv run multical vis --workspace_file calibration.pkl
```

## 库结构

Multical 在 `multical.workspace` 中提供了一个方便的高层接口，涵盖了多数常见用法，包括查找图像、加载图像、初始单相机标定、提取标定板位姿、位姿初始化、bundle adjustment 优化以及数据导出。

这个模块也是了解底层库功能如何使用的最好文档。

## 非重叠相机场景

非重叠相机场景指的是：多个相机之间视野完全没有重叠。

非常感谢 Tasnim Tabassum Nova 贡献了手眼标定模式，可以通过 `is_non_overlapping` 选项配置。

她提供了代码以及一个包含 6 个非重叠视野相机的数据集。该项目的数据集可在[这里](https://zenodo.org/records/13294455)获取。

<img src="https://github.com/user-attachments/assets/2b58c4cb-cdc6-49ba-b1d0-63af123a6473" width="300" height="300">
<img src="https://github.com/user-attachments/assets/9256dff5-e3c8-463b-a43d-588e74d33182" width="300" height="300">

该贡献包括：

1. 一种用于处理非重叠相机场景的手眼标定方法。
2. 一种用于计算内参的迭代方法；它在大数据集上更鲁棒，并减少了手动挑选内参标定图像的需要。
3. 支持剔除异常位姿，避免其参与后续计算。

相机外参初始估计和最终估计的完整可视化可在[这里](https://chart-studio.plotly.com/~Tabassum_Nova/7/#/plot)查看。

![ex_viz1](https://github.com/user-attachments/assets/f9b0bba8-6e36-42f2-be33-d7ca540b2795)

## FAQ

### 如何制作实体标定板？

下面是生成标定板图片的一个流程：

```bash
multical boards --boards example_boards/charuco_16x22.yaml --paper_size A2 --pixels_mm 10 --write my_images
```

```bash
# uv
uv run multical boards --boards example_boards/charuco_16x22.yaml --paper_size A2 --pixels_mm 10 --write my_images
```

输出示例：

```text
Using boards:
charuco_16x22 CharucoBoard {type='charuco', aruco_dict='4X4_1000', aruco_offset=0, size=(16, 22), marker_length=0.01875, square_length=0.025, aruco_params={}}
Wrote my_images/charuco_16x22.png
```

然后在 GIMP 中打开 `my_images/charuco_16x22.png`，使用“打印到文件（PDF）”，将边距设为 0，纸张大小设为 A2。之后可以打印 PDF，或发给打印店。

### multical 可以同时标定相机内参和外参吗？

可以。这是默认行为，同一组图像会同时用于内参和外参标定。如何分开进行内参和外参标定见下一节。

### multical 可以单独标定内参吗？

可以。先执行单独的内参标定；这些图像不需要在不同相机之间一一对应。每个相机会在 `intrinsic.json` 中生成一份标定结果。

```bash
multical intrinsic --image_path intrinsic_images
```

```bash
# uv
uv run multical intrinsic --image_path intrinsic_images
```

之后可以使用已知内参进行仅外参标定。用 `--calibration` 指定已有内参标定文件，并结合 `--fix_intrinsic`，避免继续调整内参：

```bash
multical calibrate --image_path extrinsic_images --calibration intrinsic_images/intrinsic.json --fix_intrinsic
```

```bash
# uv
uv run multical calibrate --image_path extrinsic_images --calibration intrinsic_images/intrinsic.json --fix_intrinsic
```

注意：原英文 README 的部分示例写作 `--input_path`；当前代码中的路径参数是 `--image_path`。

### 如何诊断不好的标定结果？

* 检查标定板是否按预期被检测到。见上面的“标定目标”部分，尤其要确认标定板尺寸是否正确，例如 `8x6` 和 `6x8` 是否写反。
* 确保使用了正确的相机模型。注意，当前没有鱼眼相机模型；欢迎添加。
* 可视化标定结果，并检查是否符合预期。观察误差是否有特定模式，例如是否只集中在某些帧或某些相机。检查初始化结果，确认问题来自标定板检测，还是来自 bundle adjustment 优化阶段。
* 确保输入图像正确同步。不同相机的图像如果不是同一时刻采集，标定效果会很差。或者应采取措施保持相机静止，例如使用三脚架。

### 如何评估标定精度？

* 重投影误差用于衡量模型拟合数据的程度。如果输入图像数量足够、视角变化充分，较低的重投影误差通常意味着较好的标定结果。如果输入太少，相机模型约束不足，即使误差低也未必可靠。
* 可以用同一组相机采集的另一组图像来对比标定结果。固定相机参数和位姿，用另一组图像重新标定：

```bash
multical calibrate --image_path alternative_images --calibration calibration.json --fix_intrinsic --fix_camera_poses
```

```bash
# uv
uv run multical calibrate --image_path alternative_images --calibration calibration.json --fix_intrinsic --fix_camera_poses
```

如果相机参数和位姿无法很好匹配另一组图像，重投影误差会较高。

## 致谢

Multical 从 [CALICO](https://github.com/amy-tabb/calico) 应用中获得了很多灵感。CALICO 主要实现了论文 “Calibration of Asynchronous Camera Networks: CALICO” 中提出的算法。

感谢 Tasnim Tabassum Nova 对非重叠相机场景的支持以及鲁棒性改进。

[Anipose lib](https://github.com/lambdaloop/aniposelib) 中的一些抽象和思路非常有用，并在本项目中得到了扩展。项目中使用了少量代码片段来初始化相对位姿；相比 CALICO 中使用的最小二乘方法，这在大多数情况下更鲁棒。

与 aniposelib 类似，本项目使用 scipy 的非线性优化器 [scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) 作为 bundle adjustment 算法基础。

OpenCV 提供了许多有用算法，本项目大量使用 OpenCV 来检测标定板、初始化相机参数、求解手眼标定以及建模相机镜头畸变。

## 作者

Oliver Batchelor  
oliver.batchelor@canterbury.ac.nz

Tasnim Tabassum Nova  
@TabassumNova
