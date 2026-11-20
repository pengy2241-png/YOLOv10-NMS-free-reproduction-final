"""Evaluate the NMS-free one-to-one head and two NMS baselines."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import CLASS_NAMES, ShapesDetectionDataset, detection_collate
from src.metrics import class_aware_nms, summary_metrics
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/best.pt")
    parser.add_argument("--val-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--benchmark-iters", type=int, default=200)
    parser.add_argument("--num-threads", type=int, default=1)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    args = ckpt.get("args", {})
    model = build_model(
        num_classes=len(CLASS_NAMES),
        width=int(args.get("width", 24)),
        image_size=int(args.get("image_size", 128)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def collect_predictions(model, loader, device, mode: str, nms_iou: float):
    predictions = []
    gt_boxes, gt_labels = [], []
    with torch.inference_mode():
        for images, boxes, labels in loader:
            outputs = model(images.to(device))
            raw = model.decode_head(outputs["o2o" if mode.startswith("o2o") else "o2m"], conf_threshold=0.001, topk=100)
            for pred, b, l in zip(raw, boxes, labels):
                if mode.endswith("nms"):
                    pred = class_aware_nms(pred, iou_threshold=nms_iou)
                predictions.append({k: v.detach().cpu() for k, v in pred.items()})
                gt_boxes.append(b.cpu())
                gt_labels.append(l.cpu())
    return predictions, gt_boxes, gt_labels


def benchmark(model, images, device, mode: str, nms_iou: float, iters: int = 100) -> float:
    images = images.to(device)
    timings = []
    with torch.inference_mode():
        for _ in range(20):
            outputs = model(images)
            raw = model.decode_head(outputs["o2o" if mode.startswith("o2o") else "o2m"], 0.001, 100)
            if mode.endswith("nms"):
                for p in raw:
                    class_aware_nms(p, nms_iou)
        # Best-effort device synchronization if a non-CPU backend is used.
        if hasattr(torch, "mps") and device.type == "mps":
            torch.mps.synchronize()
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(iters):
                outputs = model(images)
                raw = model.decode_head(outputs["o2o" if mode.startswith("o2o") else "o2m"], 0.001, 100)
                if mode.endswith("nms"):
                    for p in raw:
                        class_aware_nms(p, nms_iou)
            if hasattr(torch, "mps") and device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            timings.append(1000.0 * elapsed / iters)
    timings.sort()
    return timings[len(timings) // 2]


def main():
    args = parse_args()
    torch.set_num_threads(max(1, args.num_threads))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, ckpt = load_checkpoint(Path(args.checkpoint), device)
    val_ds = ShapesDetectionDataset(args.val_size, size=int(ckpt.get("args", {}).get("image_size", 128)), seed=args.seed + 10000, train=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=detection_collate)
    sample_images, _, _ = next(iter(val_loader))

    modes = ["o2o_nmsfree", "o2o_nms", "o2m_nms"]
    records = []
    details = {}
    for mode in modes:
        preds, gt_boxes, gt_labels = collect_predictions(model, val_loader, device, mode, args.nms_iou)
        metrics = summary_metrics(
            preds,
            gt_boxes,
            gt_labels,
            num_classes=len(CLASS_NAMES),
            conf_threshold=args.conf_threshold,
            iou_threshold=0.5,
        )
        latency_ms = benchmark(model, sample_images[:1], device, mode, args.nms_iou, args.benchmark_iters)
        record = {
            "mode": mode,
            "mAP50": metrics["mAP50"],
            "mAP50_95": metrics["mAP50_95"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "latency_ms": latency_ms,
            "fps": 1000.0 / max(latency_ms, 1e-9),
        }
        records.append(record)
        details[mode] = {"metrics": metrics, "latency_ms": latency_ms}
        print(
            f"{mode:12s} | mAP50 {record['mAP50']:.4f} | mAP50-95 {record['mAP50_95']:.4f} | "
            f"P {record['precision']:.4f} | R {record['recall']:.4f} | {record['latency_ms']:.2f} ms",
            flush=True,
        )

    (out_dir / "metrics.json").write_text(json.dumps({"checkpoint": str(args.checkpoint), "results": records, "details": details}, indent=2))
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Persist qualitative predictions and corresponding GT for later figures.
    preds_o2o, gt_boxes, gt_labels = collect_predictions(model, val_loader, device, "o2o_nmsfree", args.nms_iou)
    preds_o2m, _, _ = collect_predictions(model, val_loader, device, "o2m_nms", args.nms_iou)
    torch.save(
        {
            "o2o_nmsfree": preds_o2o,
            "o2m_nms": preds_o2m,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
        },
        out_dir / "predictions_val.pt",
    )
    print(f"saved: {out_dir / 'metrics.json'} | {out_dir / 'predictions_val.pt'}", flush=True)


if __name__ == "__main__":
    main()
