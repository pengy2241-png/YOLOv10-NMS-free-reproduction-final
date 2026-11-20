# YOLOv10 NMS-free Core Reproduction

这是一个面向课程「累積式大作業 ②：技術復現」的**簡化版 YOLOv10 核心方法復現**。目標不是重現完整 YOLOv10 在 COCO 上的絕對分數，而是驗證其最核心的部署思想：

> 使用一致的匹配代價同時訓練 one-to-many 與 one-to-one 兩個檢測頭；推理時直接使用 one-to-one 頭，不再執行 NMS。

原論文：

> Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., Han, J., & Ding, G. (2024). **YOLOv10: Real-Time End-to-End Object Detection**. NeurIPS 2024. DOI: 10.52202/079017-3429. arXiv: 2405.14458.

## 1. 復現範圍

### 已復現

- one-to-many 與 one-to-one 雙檢測頭；
- 兩個頭共享相同的候選 cell 生成與匹配代價；
- one-to-many 使用 `top-k=3`，one-to-one 使用 `top-k=1`；
- one-to-one head 的 NMS-free 推理；
- one-to-many + NMS、one-to-one + NMS 作為對照；
- mAP50、mAP50-95、Precision、Recall、F1 與 CPU latency 評估；
- 損失曲線、預測框視覺化、終端輸出截圖。

### 刻意簡化

| 原論文 | 本復現 |
|---|---|
| YOLOv10 完整 CNN 架構與多尺度特徵 | 單尺度 stride-8 小型 CNN |
| COCO 大規模資料與長訓練 | 768/192 張 Shapes-DET 合成圖，40 epochs |
| 完整 consistent dual assignments 與效率模組 | 共享匹配代價 + top-k 差異的最小實現 |
| GPU 上的端到端延遲 | CPU 單張圖延遲 |
| 完整消融實驗 | NMS-free 與 NMS 的受控比較 |

因此，本專案復現的是**機制與趨勢**，而不是原論文 COCO 指標的數值本身。這個限制會在報告中明確說明。

## 2. 方法

```text
image
  -> shared tiny backbone (stride 8)
  -> one-to-many head (top-k = 3)  -> training supervision (dense)
  -> one-to-one head  (top-k = 1)  -> deployment head (NMS-free)

matching cost = 2 * classification cost
              + 1 * objectness cost
              + 3 * L1 box cost
              + 2 * (1 - IoU)
```

兩個 head 使用同一組候選 cell 與同一個 cost，只改變每個 GT 可以匹配的 cell 數量。這是最小化但可運行的 consistent dual-assignment 復現。

## 3. 快速開始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m unittest discover -s tests -v
python train.py --epochs 40 --train-size 768 --val-size 192 --out-dir results/stage1
python train.py --epochs 20 --train-size 768 --val-size 192 --lr 3e-4 \
  --resume results/stage1/best.pt --o2o-weight 2.5 --out-dir results/stage2
python train.py --epochs 10 --train-size 768 --val-size 192 --lr 1e-4 \
  --resume results/stage2/best.pt --o2o-weight 2.5 --hard-negative-weight 0.1 --out-dir results
python evaluate.py --checkpoint results/best.pt --conf-threshold 0.2
python make_figures.py
```

一鍵執行：

```bash
bash scripts/run_all.sh
```

## 4. 目錄結構

```text
.
├── src/
│   ├── assigner.py      # consistent dual assignment
│   ├── data.py          # Shapes-DET synthetic dataset
│   ├── loss.py          # focal / classification / box / GIoU loss
│   ├── metrics.py       # AP, precision/recall, NMS
│   └── model.py         # shared backbone + dual heads
├── tests/test_core.py
├── train.py
├── evaluate.py
├── make_figures.py
├── results/
│   ├── best.pt
│   ├── best_stage1.pt
│   ├── metrics.json
│   ├── metrics.csv
│   ├── ablation.json
│   ├── predictions_val.pt
│   └── screenshots/
└── scripts/run_all.sh
```

## 5. 資料集

`Shapes-DET` 由腳本即時生成，不需要下載外部資料：

- 圖像大小：128 × 128；
- 類別：circle、square、triangle；
- 每張圖 1-3 個物件；
- 隨機背景、顏色、尺寸與輕微噪聲；
- 固定隨機種子，保證可重跑。

它是一個受控基準，目的是隔離「NMS-free 雙重分配」這個變量，而不是替代 COCO。

## 6. 結果

最終模型採用三階段訓練：40 epochs 基礎訓練、20 epochs 加強 one-to-one 監督、10 epochs 溫和重複預測抑制。總計 70 epochs。驗證集固定為 192 張未見合成圖。

| 模式 | mAP50 | mAP50-95 | Precision | Recall | F1 | CPU latency |
|---|---:|---:|---:|---:|---:|---:|
| one-to-one NMS-free（最終） | 0.9065 | 0.7775 | 0.7824 | 0.8226 | 0.8022 | 3.56 ms |
| one-to-one + NMS | 0.9944 | 0.8234 | 0.9969 | 0.8226 | 0.9014 | 4.45 ms |
| one-to-many + NMS | 0.9974 | 0.8628 | 0.8220 | 0.9974 | 0.9013 | 4.68 ms |

核心觀察：

1. **延遲方向與原論文一致。** one-to-one NMS-free 為 3.56 ms，比 one-to-one + NMS 少約 20.0%，比 one-to-many + NMS 少約 23.9%。
2. **精度仍有差距。** NMS-free 的 mAP50 為 0.9065，低於 one-to-many + NMS 的 0.9974。主要原因是簡化模型仍存在部分重複預測，而 NMS 能消除它們。
3. **失敗實驗被保留。** 若把鄰域 hard-negative 權重提高到 1.5，one-to-one 頭會過度保守，Recall 降到 0.0051。最終改用 0.1，避免模型不再輸出物件。此結果記錄在 `results/ablation.json`。

值得注意的是，表中 mAP 是與置信度閾值無關的指標；Precision / Recall 則使用統一的 0.20 閾值。所有模式共用同一組模型權重、驗證集與 NMS IoU。

結果圖：

- `results/screenshots/training_curves.png`
- `results/screenshots/metrics_dashboard.png`
- `results/screenshots/qualitative_predictions.png`
- `results/screenshots/terminal_output.png`

## 7. 判讀方式

- **mAP50 / mAP50-95**：檢測精度；
- **Precision / Recall / F1**：固定置信度閾值下的操作點；
- **latency / FPS**：CPU 單張圖的端到端推論，包含 decode，NMS 模式另含 NMS；
- 比較時三個模式的模型權重、驗證集、分數閾值與 NMS IoU 均保持不變。

預期趨勢是：one-to-one NMS-free 的精度接近 one-to-many + NMS，但省去 NMS 後具有更低延遲。若精度差距過大，則說明簡化架構下的 one-to-one 監督不足，需要更多訓練或更成熟的標籤分配策略。

## 8. 與原論文對比

原論文報告 YOLOv10-S 相較 RT-DETR-R18 約 1.8 倍速度、參數與 FLOPs 約少 2.8 倍，並以 one-to-one 頭實現 NMS-free。本專案不試圖複製這些 COCO 絕對數值，而是檢查相同機制在小型受控資料上是否呈現相同方向：**NMS-free 分支能否在精度可接受的前提下降低後處理延遲**。

## 9. 限制

- 合成資料與 COCO 的長尾、遮擋與小物件分佈不同；
- 單尺度特徵圖不具備完整多尺度檢測能力；
- CPU 延遲不能直接推廣到 GPU 或邊緣裝置；
- `top-k` 與損失權重是簡化設定，未做完整超參數搜索；
- 未復現 YOLOv10 的完整效率模組與大規模預訓練。

## 10. 授權

本復現代碼使用 MIT License。原論文與其官方代碼的權利歸原作者所有。
