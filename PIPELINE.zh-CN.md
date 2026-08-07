# Multical 一键流程

`scripts/run_pipeline.py` 把内参、外参、验证、世界坐标、三角测量和报告命令集中到一个 YAML 中，并记录每一步是否成功。

## 直接使用 20260729 配置

最短用法：

```bash
./pipeline dry-run          # 检查全部命令，不执行
./pipeline all              # 断点续跑完整流程
./pipeline validation cam3  # 只验证 cam3
./pipeline analyze intrinsic # 只分析内参
./pipeline analyze extrinsic # 只分析外参
./pipeline analyze all       # 内外参联合分析
./pipeline list             # 查看阶段
./pipeline stage world      # 只运行一个阶段
```

使用其他配置：

```bash
./pipeline --config configs/pipeline.NEW.yaml all
```

也可以直接调用 Python 执行器。先检查即将执行的命令，不会创建或修改文件：

```bash
uv run python scripts/run_pipeline.py \
  --config configs/pipeline.20260729.yaml \
  --dry-run
```

运行完整流程：

```bash
uv run python scripts/run_pipeline.py \
  --config configs/pipeline.20260729.yaml \
  --stage all \
  --resume
```

只验证 cam3 内参：

```bash
uv run python scripts/run_pipeline.py \
  --config configs/pipeline.20260729.yaml \
  --stage validation \
  --camera cam3 \
  --resume
```

只生成三维重建和报告：

```bash
uv run python scripts/run_pipeline.py \
  --config configs/pipeline.20260729.yaml \
  --stage triangulate,evaluate3d
```

从外参开始运行，到世界坐标完成后停止：

```bash
uv run python scripts/run_pipeline.py \
  --config configs/pipeline.20260729.yaml \
  --stage all \
  --from-stage extrinsic \
  --to-stage world \
  --resume
```

查看配置中的阶段：

```bash
uv run python scripts/run_pipeline.py \
  --config configs/pipeline.20260729.yaml \
  --list-stages
```

## 断点续跑规则

`--resume` 仅在以下条件全部满足时跳过阶段：

- 上次状态为成功；
- 实际命令和参数没有变化；
- 主要输入文件没有变化；
- 预期输出文件仍然存在。

修改参数或输入后，相应阶段会自动重新执行。使用 `--force` 可以强制重跑所选阶段。

默认严格只执行所选阶段，不检查或运行前置步骤；如果所需输入不存在，会直接报错。确实需要串联前置步骤时可显式增加 `--with-deps`。

状态记录默认保存在配置的 `state_file`，20260729 配置使用：

```text
outputs/pipeline_20260729/pipeline_state.json
```

## 配置说明

- `variables`：集中保存数据目录、标定板、相机列表和输出根目录。
- `env`：所有阶段使用的环境变量。
- `stages`：按书写顺序执行。
- `enabled: false`：默认跳过阶段，适合 `observe` 这类交互步骤。
- `command`：Multical 子命令，也支持 `analyze` 和通用 `python`。
- `group`：允许一条 `--stage validation` 选择同组验证阶段。
- `args`：直接对应原命令行参数，不需要写 `--`。
- `outputs`：通常可以自动推导；自定义 Python 阶段可显式填写。

单相机验证引用多相机内参 JSON 时，执行器会在输出目录的 `.pipeline_inputs/` 下自动生成临时的单相机内参，不修改原始内参文件。

`enabled: false` 只表示不参与 `all`；仍可使用 `./pipeline stage observe` 单独运行交互阶段。

交互阶段可设置 `interactive: true`，这样即使输出文件已经存在，`--resume` 也不会跳过窗口。对同一组同步图片连续标注多个点时，在 `observe.args` 中设置 `frame: 000000.jpg`；保存后会自动生成 P01、P02 等编号。

## 新数据集

复制配置并优先修改顶部变量：

```bash
cp configs/pipeline.20260729.yaml configs/pipeline.NEW.yaml
```

通常只需修改：

```yaml
variables:
  dataset: 新数据目录
  boards: 新标定板配置.yaml
  cameras: [cam0, cam1, cam2, cam3]
  output_root: outputs/新任务名称
```

如目录结构或阈值不同，再修改对应阶段的 `args`。
