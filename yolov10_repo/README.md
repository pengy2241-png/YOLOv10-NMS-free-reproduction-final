# YOLOv10 NMS-free Reproduction and Integrated AT4 Submission

本仓库是累积式大作业（AT3）与 AT4 期末整合报告的完整提交版本，包含文献综述、YOLOv10 NMS-free 技术复现、离线 Agent 工作流和最终论文。

- 作者：彭芷仪
- 学号：d26092110368
- 最终论文：[paper.pdf](paper.pdf)

## AT3 与 AT4 对应关系

| 累积式大作业部分 | 仓库位置 | AT4 论文章节 |
|---|---|---|
| ① 文献综述 | `literature_review/` | Related Work |
| ② 技术复现 | `src/`, `results/`, `docs/reproduction/` | Method 与 Experiments |
| ③ 离线 Agent | `agent_task/` | Method 与 Experiments |
| 最终整合报告 | `paper.pdf` | 全文 |

## 仓库结构

```text
.
├── README.md
├── paper.pdf
├── requirements.txt
├── src/
│   ├── assigner.py
│   ├── data.py
│   ├── loss.py
│   ├── metrics.py
│   └── model.py
├── results/
│   ├── metrics.json
│   ├── metrics.csv
│   ├── ablation.json
│   ├── train_log*.json
│   ├── logs/
│   └── screenshots/
├── tests/
├── scripts/
├── docs/reproduction/
├── literature_review/
└── agent_task/
```

## 环境

主要实验环境记录如下。

- Python 3.12.14
- PyTorch 2.15.0
- CPU 执行
- 训练使用 4 个 CPU 线程
- 延迟基准使用 1 个 CPU 线程

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## YOLOv10 NMS-free 复现

### 方法

复现 consistent dual assignment 的核心机制。共享 backbone 同时产生 one-to-many 与 one-to-one 两个检测头。one-to-many 使用 `top-k=3`，one-to-one 使用 `top-k=1`。推理时只使用 one-to-one 头，因此不需要执行 NMS。

### 数据

Shapes-DET 由脚本即时生成。

- 图像大小：128 × 128
- 类别：circle、square、triangle
- 每张图包含 1 至 3 个目标
- 训练集：768 张
- 验证集：192 张
- 输入随机种子：42
- 验证随机种子：10042

### 训练配置

| 项目 | 配置 |
|---|---|
| 模型 | 单尺度 stride-8 tiny CNN |
| 模型宽度 | 24 |
| 可训练参数 | 604,264 |
| 优化器 | AdamW |
| Batch size | 32 |
| Weight decay | 0.0001 |
| 学习率调度 | Cosine schedule |
| 阶段 1 | 40 epochs，LR 0.002，one-to-one weight 1.0 |
| 阶段 2 | 20 epochs，LR 0.0003，one-to-one weight 2.5 |
| 阶段 3 | 10 epochs，LR 0.0001，one-to-one weight 2.5，hard-negative weight 0.1 |

### 一键运行

```bash
bash scripts/run_all.sh
```

### 分阶段运行

```bash
python train.py --epochs 40 --train-size 768 --val-size 192 --out-dir results/stage1

python train.py --epochs 20 --train-size 768 --val-size 192 --lr 3e-4 \
  --resume results/stage1/best.pt --o2o-weight 2.5 --out-dir results/stage2

python train.py --epochs 10 --train-size 768 --val-size 192 --lr 1e-4 \
  --resume results/stage2/best.pt --o2o-weight 2.5 --hard-negative-weight 0.1 \
  --out-dir results

python evaluate.py --checkpoint results/best.pt --conf-threshold 0.2
python make_figures.py
```

### 主要结果

| 模式 | mAP50 | mAP50 to 95 | Precision | Recall | F1 | CPU latency |
|---|---:|---:|---:|---:|---:|---:|
| One-to-one NMS-free | 0.9065 | 0.7775 | 0.7824 | 0.8226 | 0.8022 | 3.56 ms |
| One-to-one + NMS | 0.9944 | 0.8234 | 0.9969 | 0.8226 | 0.9014 | 4.45 ms |
| One-to-many + NMS | 0.9974 | 0.8628 | 0.8220 | 0.9974 | 0.9013 | 4.68 ms |

NMS-free 模式比 one-to-one + NMS 快约 20.0%，比 one-to-many + NMS 快约 23.9%。所有结果来自最终 checkpoint 的一次确定性评估。延迟是单线程 CPU 上预热 20 次、测量 200 次、重复 3 次后取中位数。

### 测试

```bash
python -m unittest discover -s tests -v
```

## 离线 Agent 工作流

Agent 使用四个工具完成离线来源整理：

1. `search_sources`
2. `read_source`
3. `write_final_brief`
4. `validate_final_state`

运行：

```bash
cd agent_task
python run_experiment.py
```

受控比较只改变 tool budget。

- `max_steps=3`：在读取第三个来源前停止，validator 失败。
- `max_steps=6`：完成 search、三次 read、write 与 validate，validator 通过。

Agent 不使用网络、API、模型或 GPU。fixture 明确标记为 `student_defined_offline_fixture`。

## 提交材料

- 最终论文：`paper.pdf`
- 文献综述：`literature_review/`
- 复现报告：`docs/reproduction/复现报告.pdf`
- Agent 提交文件与分析：`agent_task/`
- 指标与训练记录：`results/`
- 可运行源码：`src/`, `train.py`, `evaluate.py`, `infer.py`
