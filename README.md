# MiniMind-EmbodiedFlow

一个极简的 2D 具身 AI 玩具演示：

```text
图像 + 指令
        -> MiniMind-V Task Parser v2
        -> 结构化 JSON
        -> 颜色规则视觉锚定
        -> 起点 / 终点 / 障碍物
        -> Flow Matching
        -> 2D 避障轨迹
```

本包仅包含最终的演示运行时。不包含 MiniMind-V 官方仓库或 MiniMind-V VLM 模型权重。

## 包含的文件

```text
core/
  flow_model.py
  flow_sampler.py
  minimind_parser.py
  visual_grounding.py
  metrics.py
  visualization.py
  io_utils.py

scripts/
  check_env.py
  run_final_demo.py

datasets/multimodal_task_parser_v2/
  test_labels.jsonl
  images/                  # 200 张演示/测试图像
  preview_grid.png
  stats.json

checkpoints/
  two_obstacle_flow_mlp.pt
```

## 外部 MiniMind-V 依赖

需要单独准备 MiniMind-V。预期的外部目录结构如下：

```text
path/to/minimind-v-master/
  model/
  model/siglip2-base-p32-256-ve/
  out/task_parser_sft_vlm_v2_768.pth
```

演示需要以下两个参数：

```text
--minimind_v_root      path/to/minimind-v-master
--vlm_checkpoint       path/to/task_parser_sft_vlm_v2_768.pth
```

MiniMind-V SFT v2 权重文件有意未包含在本包中。请将其放置在 MiniMind-V 的 `out/` 目录下，或显式传入其路径。

## 环境配置

推荐配置：

```text
Python conda 环境: minimind-v
PyTorch: CUDA 版本
所需包: transformers, torchvision, pillow, numpy
```

安装轻量依赖：

```bash
pip install -r requirements_minimind_v_inference.txt
```

环境检查：

```bash
python scripts/check_env.py \
  --minimind_v_root E:/Minimind-v+paligemma+flowmatching/minimind-v-master \
  --vlm_checkpoint E:/Minimind-v+paligemma+flowmatching/minimind-v-master/out/task_parser_sft_vlm_v2_768.pth \
  --flow_checkpoint checkpoints/two_obstacle_flow_mlp.pt \
  --labels datasets/multimodal_task_parser_v2/test_labels.jsonl
```

## 运行演示

```bash
python scripts/run_final_demo.py \
  --labels datasets/multimodal_task_parser_v2/test_labels.jsonl \
  --minimind_v_root E:/Minimind-v+paligemma+flowmatching/minimind-v-master \
  --vlm_checkpoint E:/Minimind-v+paligemma+flowmatching/minimind-v-master/out/task_parser_sft_vlm_v2_768.pth \
  --flow_checkpoint checkpoints/two_obstacle_flow_mlp.pt \
  --output_dir outputs/final_demo \
  --num_samples 16 \
  --device auto
```

预期输出：

```text
outputs/final_demo/
  metrics.json
  results.jsonl
  raw_outputs.jsonl
  error_cases.jsonl
  summary_grid.png
  overlays/
```

## 预期指标

在本包附带的人工合成测试图像上，pipeline 应接近以下指标：

```text
parser.exact_match: 1.0
visual_grounding.grounding_success_rate: 1.0
trajectory.success_rate: 1.0
trajectory.collision_rate: 0.0
```

## 说明

- MiniMind-V 仅输出任务 JSON。
- MiniMind-V 不输出坐标、障碍物边界框或轨迹。
- 颜色规则视觉锚定从合成图像中提取起点、终点和障碍物。
- Flow Matching 仅接收几何条件。
- 这是一个合成 2D 玩具项目，而非真实的机器人系统。

## 局限性

- 视觉锚定依赖干净的人工合成颜色。
- 形状分类器基于启发式规则。
- 真实照片不在适用范围之内。
- 相同颜色和形状的多个物体可能存在歧义。
- 训练过程、消融实验、规则解析器基线及历史实验均不包含在此精简包中。
