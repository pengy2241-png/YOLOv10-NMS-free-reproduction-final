#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m unittest discover -s tests -v
python train.py --epochs 40 --train-size 768 --val-size 192 --out-dir results/stage1
python train.py --epochs 20 --train-size 768 --val-size 192 --lr 3e-4 \
  --resume results/stage1/best.pt --o2o-weight 2.5 --out-dir results/stage2
cp results/stage1/train_log.json results/train_log_stage1.json
cp results/stage1/best.pt results/best_stage1.pt
cp results/stage2/train_log.json results/train_log_stage2.json
cp results/stage2/best.pt results/best_stage2.pt
python train.py --epochs 10 --train-size 768 --val-size 192 --lr 1e-4 \
  --resume results/stage2/best.pt --o2o-weight 2.5 --hard-negative-weight 0.1 --out-dir results
python evaluate.py --checkpoint results/best.pt --conf-threshold 0.2
python make_figures.py
