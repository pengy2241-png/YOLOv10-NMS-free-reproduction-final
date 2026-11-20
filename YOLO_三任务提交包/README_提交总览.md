# YOLO 累積式大作業：三任務提交包

整理日期：2026-11-13

## 目錄

### 01_任务一_文献综述

- `YOLO任务一_提交版.pdf`：可直接提交的 2 页版本，第 1 页知识图谱、第 2 页一页摘要。
- `YOLO任务一_完整报告.pdf`：完整综述、对比表、批判性分析、研究缺口与参考文献。
- `YOLO任务一_文献综述.md`：可编辑文字稿。
- `YOLO任务一_知识图谱.svg` / `YOLO任务一_知识图谱.pdf`：知识图谱的可编辑矢量版与 PDF 版（以当前导出的文件为准）。
- `YOLO任务一_一页摘要.pdf`：单独的一页摘要。

核心文献：5 篇正式会议论文，其中 3 篇为 2024+ 顶会论文（ECCV 2024、CVPR 2024、NeurIPS 2024）。

### 02_任务二_技术复现

GitHub 倉庫：https://github.com/pengy2241-png/YOLOv10-NMS-free-reproduction

YOLOv10 NMS-free 核心方法复现，包含：

- 可运行代码、单元测试、README、requirements；
- 三阶段训练与评估脚本；
- 模型权重 `results/best.pt`；
- 指标 `metrics.json` / `metrics.csv`；
- 失败实验记录 `ablation.json`；
- 4 张结果截图；
- 复现报告 `reports/复现报告.pdf`。

最终结果：NMS-free mAP50 为 0.9065，CPU 延迟 3.56 ms；相較 NMS 基準約快 20-24%。报告已說明精度差距與簡化範圍。

### 03_任务三_Agent工作

- `final-brief.md`：一页含 citation 的研究简报；
- `task-contract.json`、`tool-schema.json`；
- `trajectory.jsonl`、`tool-results.jsonl`；
- `evaluation.csv`、`error-note.md`；
- `provenance.json`、`validator-result.json`；
- `analysis.md`：两页以内分析；
- `runs/max_steps_3/` 与 `runs/max_steps_6/`：完整受控比较。

本地受控比较：`max_steps=3` 失败，`max_steps=6` 通过；本地 rubric 分数 45/100 与 100/100。

## Fixture 说明

任务三的 fixture 由学生自行构建，目的是满足作业规定的离线约束：**不使用 API、模型、GPU、网络或大型套件**。目录中标记为：

```text
student_defined_offline_fixture
```

该 fixture 包含指定的离线 source pack、来源清单、deterministic Agent policy 和独立 final-state validator。若课程另行发布指定来源包或 validator，可替换对应文件后重跑。

## 建议提交顺序

1. 任务一：提交 `01_任务一_文献综述/YOLO任务一_提交版.pdf`；如需附录，再附完整报告。
2. 任务二：提交 GitHub 倉庫 https://github.com/pengy2241-png/YOLOv10-NMS-free-reproduction；报告在 `reports/复现报告.pdf`。
3. 任务三：提交 `03_任务三_Agent工作/` 中的 `final-brief.md`、JSON/JSONL/CSV/Markdown 文件与分析。
