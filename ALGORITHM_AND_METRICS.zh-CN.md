# Multical 算法逻辑与评价指标说明

本文面向使用者，简要说明本项目从多相机标定到世界坐标三维重建的处理逻辑，并重点解释各输出文件中的质量指标。

## 1. 一句话理解整条链路

```text
标定板图像
  -> 角点检测
  -> 各相机内参标定
  -> 标定板位姿估计与相机重叠关系建图
  -> 多相机联合优化（Bundle Adjustment）
  -> 相机组对齐到世界坐标
  -> 多视角三角化
  -> 与实测 3D 点比较
```

其中存在三类不同的“误差”，不可混用：

1. **标定重投影误差，单位 px**：标定板角点在图像上的拟合误差；
2. **世界坐标对齐重投影误差，单位 px**：已知世界点投影到图像后的拟合误差；
3. **三维重建误差，单位为世界坐标单位**：重建点与实测点之间的空间距离，这才是最终 3D 精度。

像素误差低是必要条件，但不是 3D 精度高的充分条件。相机基线、视线夹角、世界坐标对齐、控制点测量误差以及相机间外参误差都会影响最终 3D 精度。

## 2. 主标定算法

### 2.1 输入与角点检测

输入包括：

- 每台相机的标定图像；
- `boards.yaml` 中的标定板类型、尺寸和角点坐标；
- 可选的已有内参文件。

程序在每张图中检测棋盘格、ChArUco 或 AprilGrid 角点，并整理为“相机 × 帧 × 标定板 × 角点”的观测表。相同文件名表示同一个同步帧。

### 2.2 单相机内参标定

每台相机独立执行以下步骤：

1. 丢弃标定板可见角点比例低于 `intrinsic_min_board_coverage` 的图像；
2. 若图像数超过 `limit_intrinsic`，优先选择角点在画面中覆盖范围较大的图像；
3. 使用 OpenCV 标定相机矩阵 `K` 和畸变参数 `dist`；
4. 根据逐视图重投影误差进行稳健离群视图筛选；
5. 删除离群视图后重新标定，直到收敛、达到迭代上限或触发最少视图保护。

离群阈值为：

```text
robust_threshold = median_RMS + view_mad_scale × 1.4826 × MAD_RMS
threshold = max(view_error_limit, robust_threshold)
```

只有误差大于 `threshold` 的视图才是候选离群视图；每轮删除数量还受 `intrinsic_max_reject_fraction` 和 `intrinsic_min_views` 限制。

### 2.3 多相机外参初始化

对每个“相机—同步帧—标定板”组合，通过 PnP 估计标定板相对相机的位姿。误差过大的 PnP 位姿会被排除。

随后统计相机两两之间的共同有效观测，构建相机重叠图：

- 边的权重来自共同可见的标定板角点数量；
- 自动选择主相机和一组连接全部相机的初始化边；
- 用共同帧中的多组位姿稳健估计相机间初始刚体变换。

如果相机图不连通，就无法可靠地把全部相机放入同一个坐标系。

### 2.4 多相机联合优化

程序使用非线性最小二乘联合最小化所有有效角点的二维重投影残差。默认主要优化：

- 相机之间的固定外参；
- 每个同步帧的运动位姿；
- 标定板之间的固定位姿。

相机内参和标定板几何参数是否参与优化由命令参数决定；使用 `--fix_intrinsic` 时内参保持不变。

优化期间会迭代执行：

1. 根据角点重投影误差选择离群点；
2. 若某个相机帧中离群点比例过高，则整帧剔除；
3. 在保留的内点上重新进行 Bundle Adjustment；
4. 最后再次检查内外点分类，并在分类变化时重新拟合。

默认点级离群阈值的基本形式是：

```text
threshold = clamp(Q75(error) × outlier_threshold,
                  outlier_min_threshold,
                  outlier_max_threshold)
```

最终生成 `calibration.json`，保存 `K`、`dist`、相机位姿以及质量诊断。

## 3. 世界坐标对齐与三维重建

### 3.1 世界坐标对齐

Multical 标定得到的是相机组内部的相对坐标。要输出业务所需的世界坐标，还需要已知的“世界 3D 点—图像 2D 点”对应关系。

- `world`：使用单台锚定相机，通过 PnP 求世界坐标到该相机的变换，再传播到整个相机组；
- `worldmulti`：固定已标定的相机内外参，只优化一个公共的 `world_to_rig` 刚体变换，让多台相机共同约束世界坐标对齐。

`worldmulti` 先从具备至少 4 个不同世界点的相机产生初值，再用稳健损失联合优化全部观测；以 `ransac_threshold_px` 区分内点和外点，并在内点上完成最终拟合。

### 3.2 多视角三角化

对每个同步目标点：

1. 对像素点去畸变并转换为归一化相机坐标；
2. 枚举相机对，用加权 DLT 产生 3D 候选点，观测置信度以平方根权重进入方程；
3. 计算候选点在所有相机中的深度和重投影误差；
4. 优先选择内点相机最多、置信度最高、重投影误差最低的候选；
5. 使用全部内点相机重新三角化，并迭代剔除误差超限或负深度的观测；
6. 可选地用非线性最小二乘进一步最小化像素重投影残差；
7. 检查视线夹角，夹角小于 `min_ray_angle_deg` 时拒绝该结果。

每帧至少需要两台有效相机。

### 3.3 3D 真值评估

`evaluate3d` 按帧名匹配重建点和实测世界点，先统一长度单位，再计算：

```text
delta = reconstructed_world - measured_world
error_3d = sqrt(delta_x² + delta_y² + delta_z²)
```

最后汇总整体距离误差、各坐标轴误差和验收结果。

## 4. 输出评价指标详解

### 4.1 `intrinsic.json`：单相机内参质量

路径：`cameras.<camera>.quality`

| 指标 | 含义 | 方向与注意事项 |
| --- | --- | --- |
| `RMS` | OpenCV 内参标定的整体重投影 RMS，单位 px | 越小越好；需结合视图数量和画面覆盖判断 |
| `mean_view_RMS` | 所有最终采用视图的逐视图 RMS 算术平均 | 越小越好 |
| `max_view_RMS` | 最差采用视图的 RMS | 越小越好；明显高于均值时应检查对应图像 |
| `view_count` | 最终采用的“图像—标定板”视图数 | 不是越多越好，关键是姿态与画面覆盖充分 |
| `image_count` | 最终采用的不同图像数 | 单图可能包含多块板，因此可与 `view_count` 不同 |
| `input_image_count` | 输入图像总数 | 数据规模指标 |
| `detected_image_count` | 至少成功检测到一块有效标定板的图像数 | 与输入数差距大时应检查曝光、清晰度和板配置 |
| `detection_failed_image_count` | 未成功检测的图像数 | 越少越好 |
| `quality_rejected_image_count` | 因标定板可见比例不足被剔除的图像数 | 多时说明遮挡或板未完整入镜 |
| `excluded_by_limit_image_count` | 检测合格但因 `limit_intrinsic` 未被选中的图像数 | 属于主动限量，不等同于坏图 |
| `rejected_image_count` | 候选图像中因重投影离群而剔除的图像数 | 多时需检查模糊、误检或模型不匹配 |
| `observation_count` | 最终采用的角点数 | 约束数量指标 |
| `filter_converged` | 是否已没有可安全删除的离群视图 | `true` 较理想 |
| `filter_limit_reached` | 是否因迭代上限停止且仍存在候选离群视图 | `true` 时应检查 `filter_history` |
| `overall_error_limit_met` | 最终整体 RMS 是否低于配置目标 | 仅表示达到该像素目标，不代表 3D 精度达标 |
| `image_board_coverage` | 每张图最佳标定板的已检测角点数 / 板总角点数 | 越接近 1 越完整；不是“画面面积覆盖率” |
| `views[].RMS` | 某一最终采用视图的重投影 RMS | 用于定位具体问题图像 |

注意：整体 `RMS` 是按全部角点残差汇总的结果，不一定等于 `mean_view_RMS`。

### 4.2 `calibration.json.quality`：联合标定总体质量

程序先对每个角点计算二维欧氏误差：

```text
e_i = sqrt((u_projected-u_observed)² + (v_projected-v_observed)²)
RMS = sqrt(mean(e_i²))
```

| 指标 | 含义 | 方向与注意事项 |
| --- | --- | --- |
| `RMS` | 最终内点的全局重投影 RMS，单位 px | 越小越好；这是主要拟合指标 |
| `RMS_all` | 包含被判为外点的全部观测 RMS | 越小越好；与 `RMS` 差距大说明存在明显离群观测 |
| `inlier_observation_count` | 最终内点角点数 | 与总数结合看保留比例 |
| `observation_count` | 全部有效角点数 | 数据规模指标 |
| `outlier_filter_applied` | 是否应用了离群点筛选 | 为 `true` 时应同时查看 `RMS_all` 和内点比例 |

推荐额外计算：

```text
inlier_ratio = inlier_observation_count / observation_count
```

不能只追求低 `RMS`：过少、过于相似或只覆盖画面中心的图像也可能得到很低的训练内误差，却无法良好约束焦距、畸变和外参。

### 4.3 `calibration.json.extrinsic_quality`：外参质量诊断

该节点标记为 `diagnostic_only=true`，表示它用于风险提示，不参与最终优化结果的求解。

#### 每台相机 `cameras.<camera>`

| 指标 | 含义 |
| --- | --- |
| `observation_count` / `inlier_count` / `outlier_count` | 该相机全部角点、内点和外点数 |
| `inlier_ratio` | 内点占全部观测比例，越高越好；低于 0.9 会产生警告 |
| `detected_frame_count` | 有有效角点的帧数 |
| `inlier_frame_count` | 至少保留一个内点的帧数 |
| `rejected_frame_count` | 检测有效但整帧没有内点的帧数 |
| `reprojection_RMS_px` | 该相机内点的重投影 RMS |
| `all_points_RMS_px` | 该相机全部有效角点的重投影 RMS |

#### 每个相机对 `pairs.<cameraA>:<cameraB>`

| 指标 | 含义 | 方向 |
| --- | --- | --- |
| `selected_initialization_edge` | 是否被选为连接相机图的初始化边 | 初始化关键边为弱边时风险较高 |
| `common_frame_count` | 两台相机共同看到标定板的同步帧数 | 通常越多越稳 |
| `common_corner_count` | 共同可见角点数 | 通常越多越稳 |
| `common_inlier_corner_count` | 联合优化后仍为内点的共同角点数 | 越多越好 |
| `pair_reprojection_RMS_px` | 两台相机共同内点的合并重投影 RMS | 越小越好 |
| `common_pose_count` | 可用于逐帧相对位姿统计的共同位姿数 | 越多越稳 |
| `pose_inlier_count` | 稳健位姿对齐保留的共同位姿数 | 与 `common_pose_count` 结合判断 |
| `rotation_scatter_deg` | 各共同帧推导出的相机间旋转相对稳健中心的 RMS 离散，单位度 | 越小越好 |
| `translation_scatter` | 各共同帧推导出的平移相对稳健中心的 RMS 离散 | 越小越好；单位与标定板尺寸单位相同 |
| `frobenius_scatter` | 4×4 变换矩阵残差的 Frobenius 范数 RMS | 越小越好，但物理意义不如旋转/平移直观 |
| `final_edge_rotation_residual_deg` | 最终联合外参与该相机对稳健中心之间的旋转差 | 越小越好 |
| `final_edge_translation_residual` | 最终联合外参与该相机对稳健中心之间的平移差 | 越小越好 |
| `final_rotation_residual_rms_deg` | 最终外参解释所有共同帧时的旋转残差 RMS | 越小越好 |
| `final_translation_residual_rms` | 最终外参解释所有共同帧时的平移残差 RMS | 越小越好 |
| `status` | `good`、`caution`、`weak` 或 `no_overlap` | 由共同帧数及旋转/平移离散阈值判定 |

当前代码中的诊断阈值为：

| 状态 | 共同帧数 | 旋转离散 | 平移离散 |
| --- | ---: | ---: | ---: |
| `good` | ≥ 30 | ≤ 0.2° | ≤ 0.05 |
| `caution` | ≥ 20 | ≤ 0.5° | ≤ 0.15 |
| `weak` | 未满足上述条件 |  |  |
| `no_overlap` | 共同帧数为 0 |  |  |

这些是项目内置的经验诊断阈值，平移阈值会随标定板所用长度单位变化，不能脱离单位直接套用。

#### 图结构 `graph`

| 指标 | 含义 |
| --- | --- |
| `connected` | 初始化图是否用 `相机数-1` 条边连通全部相机；应为 `true` |
| `initialization_master` | 自动选择的初始化主相机 |
| `selected_initialization_edges` | 用于传播初始外参的边 |
| `weak_selected_edges` | 被选中的初始化边中质量为 `weak` 的边；应重点检查 |
| `redundant_edge_consistency` | 未被选中的直接边与初始化树间接路径之间的闭环误差 |
| `rotation_closure_error_deg` | 闭环旋转不一致，越小越好 |
| `translation_closure_error` | 闭环平移不一致，越小越好 |

### 4.4 `world_extrinsics_multicam.json.joint`：世界坐标对齐质量

| 指标 | 含义 | 方向与注意事项 |
| --- | --- | --- |
| `observation_count` | 参与求解的 2D—3D 对应观测数 | 数据规模指标 |
| `distinct_world_point_count` | 不同世界控制点数量 | 应具有足够数量和空间分布 |
| `participating_cameras` | 实际参与世界对齐的相机 | 应符合预期 |
| `inlier_count` | 误差不超过 `ransac_threshold_px` 的观测数 | 应结合总数看比例 |
| `reprojection_rms_px` | 内点重投影 RMS | 越小越好 |
| `reprojection_mean_px` | 内点重投影误差均值 | 越小越好 |
| `reprojection_max_px` | 内点最大重投影误差 | 正常情况下不应明显超过筛选阈值 |
| `all_points_rms_px` | 包含外点的全部观测 RMS | 与内点 RMS 差距大说明数据中有大量不一致点 |
| `all_points_max_px` | 全部观测的最大误差 | 用于定位最差点 |
| `optimization_success` | 数值优化器是否正常结束 | 只代表求解成功，不代表解准确 |
| `camera_statistics` | 按相机拆分的观测数、内点数和误差 | 用于定位某一相机或对应关系的问题 |
| `skipped_observations` | 因图像或标记缺失而未进入求解的观测 | 用于检查数据完整性 |

建议同时计算 `inlier_count / observation_count`。若内点 RMS 较低但内点比例很低，说明求解只解释了少量数据，世界对齐仍不可信。

### 4.5 `triangulation.json`：逐帧三角化质量

#### `summary`

| 指标 | 含义 |
| --- | --- |
| `frame_count` | 输入帧总数 |
| `reconstructed_count` | 成功输出 3D 点的帧数 |
| `failed_count` | 因观测不足、负深度、误差超限或视线夹角不足而失败的帧数 |
| `refined_count` | 非线性优化实际改善并被接受的帧数 |
| `mean_reprojection_rms_px` | 所有成功帧的逐帧 RMS 的算术平均 |
| `max_reprojection_error_px` | 所有成功帧、所用相机中的最大单相机像素误差 |
| `min_ray_angle_deg` | 所有成功帧的 `max_ray_angle_deg` 最小值 |

#### `frames[]`

| 指标 | 含义 | 方向与注意事项 |
| --- | --- | --- |
| `status` | `ok` 或 `failed` | 失败时查看 `reason` |
| `point_world` | 重建的世界坐标 `[X,Y,Z]` | 单位见根节点 `world_units` |
| `cameras_used` | 最终作为内点参与三角化的相机 | 至少 2 台 |
| `cameras_rejected` | 因未标定、负深度或误差超限被排除的相机 | 多时需检查同步和检测 |
| `reprojection_errors_px` | 重建点在每台相机上的像素误差 | 越小越好 |
| `reprojection_rms_px` | 所用相机像素误差的 RMS | 越小越好 |
| `reprojection_max_px` | 所用相机中的最大像素误差 | 应不超过配置阈值 |
| `max_ray_angle_deg` | 所用相机两两视线夹角的最大值 | 一般越大几何条件越好；过小会放大深度误差 |
| `refinement.initial_reprojection_rms_px` | DLT 初值的像素 RMS | 与最终值比较 |
| `refinement.final_reprojection_rms_px` | 非线性优化后的像素 RMS | 不应高于初值 |
| `refinement.improvement_px` | 优化前后 RMS 的下降量 | 大于等于 0 |

注意：当前实现输出的是相机对中的**最大**视线夹角，并以它做最小角度检查；这不代表所有被采用的相机对都具有同样好的夹角。

### 4.6 `evaluation3d.json`：最终 3D 精度

这是最接近业务效果的评价文件。

#### 数据完整性

| 指标 | 含义 |
| --- | --- |
| `ground_truth_count` | 实测真值点数量 |
| `matched_count` | 成功匹配且成功重建的点数 |
| `failed_reconstruction_count` | 找到对应帧但重建失败的真值点数 |
| `missing_reconstruction_count` | 在重建文件中找不到对应帧的真值点数 |
| `unmeasured_reconstruction_count` | 有重建结果但没有真值的点数 |

#### 整体空间距离 `summary.error_3d`

对每个成功匹配点，`error_3d` 是重建点与实测点的欧氏距离。

| 指标 | 含义 | 特点 |
| --- | --- | --- |
| `mean` | 3D 距离误差算术平均 | 表示平均精度，易受大误差影响 |
| `RMS` | 3D 距离误差均方根 | 对大误差惩罚更强，通常不小于 mean |
| `median` | 3D 距离误差中位数 | 表示典型水平，对少量异常值更稳健 |
| `p95` | 95% 的成功匹配点误差不超过该值 | 适合描述尾部风险和服务质量 |
| `max` | 最大 3D 距离误差 | 表示最坏样本 |

#### 分轴误差 `summary.axis_error.X/Y/Z`

| 指标 | 含义 | 解读 |
| --- | --- | --- |
| `bias` | 有符号坐标误差的均值 | 接近 0 较好；正负号表示系统偏移方向 |
| `MAE` | 绝对坐标误差均值 | 该轴典型绝对偏差，越小越好 |
| `RMS` | 该轴误差均方根 | 对大偏差更敏感，越小越好 |
| `max_abs` | 该轴最大绝对误差 | 该方向的最坏偏差 |

若某一轴 `|bias|` 接近该轴 `MAE`，说明误差大多朝同一方向，通常提示世界坐标对齐、尺度或系统性外参偏差，而不只是随机噪声。

#### 验收结果 `acceptance`

- `thresholds`：命令行传入的 `max_mean_error`、`max_p95_error`、`max_error`；
- `passed`：没有缺失/失败的真值点，并且所有已配置阈值都通过；
- `failures`：未通过原因。

**重要：若没有配置任何精度阈值，`passed=true` 只说明全部真值点均成功重建，不代表 3D 误差满足业务要求。** 应根据实际业务单位和容差显式设置三个阈值。

## 5. 建议的判读顺序

建议按以下顺序排查，避免只盯一个 RMS：

1. **内参数据是否健康**：检测成功率、画面覆盖、视图数量、`max_view_RMS`；
2. **联合标定是否稳定**：全局 `RMS`、`RMS_all`、内点比例；
3. **外参链路是否可靠**：`graph.connected`、`weak_selected_edges`、相机对旋转/平移离散和闭环误差；
4. **世界对齐是否可靠**：控制点空间分布、内点比例、按相机误差和验证叠加图；
5. **三角化几何是否良好**：成功率、被拒相机、重投影误差、视线夹角；
6. **最终看 3D 真值误差**：`mean`、`p95`、`max` 和各轴 `bias`。

## 6. 当前 `20260731` 样例的简要解读

以下仅用于展示如何读指标：

- 内参 `RMS` 约为 `0.16～0.20 px`，且示例相机的筛选已收敛，单相机角点拟合较好；
- 联合标定全局内点 `RMS=0.861 px`，`RMS_all=0.950 px`，角点内点率约为 `99.64%`；
- 但外参初始化所用的三条边全部为 `weak`，部分相机对的旋转和平移离散明显超过内置阈值，说明“全局像素 RMS 尚可”不能消除外参一致性风险；
- 世界坐标联合对齐示例只有 `23/76` 个内点，内点 RMS 为 `1.714 px`，全部点 RMS 为 `8.077 px`，表明大量世界控制点观测与同一个刚体变换不一致；
- 三角化成功 `7/10`，成功帧平均重投影 RMS 仅 `0.452 px`，最小视线夹角约 `35.95°`；
- 但最终 3D 误差 mean 为 `0.459 m`、p95 为 `0.748 m`，且 X 轴 bias 为 `+0.355 m`。这说明主要问题更像世界坐标对齐或跨相机外参的系统偏差，而不是三角化点自身的像素拟合问题。

因此，对该样例不能仅凭低像素重投影误差判定系统精度良好，应优先检查外参弱连接、世界控制点对应关系、控制点测量尺度以及多相机世界对齐中的低内点比例。
