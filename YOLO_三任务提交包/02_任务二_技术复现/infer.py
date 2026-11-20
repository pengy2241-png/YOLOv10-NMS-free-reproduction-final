"""Run one validation sample and save a GT/prediction visualization."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from evaluate import load_checkpoint
from src.data import CLASS_NAMES, ShapesDetectionDataset
from src.metrics import class_aware_nms


def to_pixels(box, size):
    return tuple(int(float(v) * size) for v in box)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/best.pt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--head", choices=["o2o", "o2m"], default="o2o")
    parser.add_argument("--nms", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", default="results/inference_sample.png")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, ckpt = load_checkpoint(Path(args.checkpoint), device)
    size = int(ckpt.get("args", {}).get("image_size", 128))
    ds = ShapesDetectionDataset(192, size=size, seed=42 + 10000, train=False)
    image, gt_boxes, gt_labels = ds[args.index]
    with torch.inference_mode():
        output = model(image[None].to(device))
        pred = model.decode_head(output[args.head], conf_threshold=0.001, topk=100)[0]
    if args.nms:
        pred = class_aware_nms(pred, 0.5)
    keep = pred["scores"] >= args.conf
    canvas = ((image * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    pil = Image.fromarray(canvas).resize((size * 4, size * 4), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(pil)
    scale = size * 4
    for b, l in zip(gt_boxes, gt_labels):
        cx, cy, bw, bh = [float(v) for v in b]
        gt_box = to_pixels(torch.tensor([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]), scale)
        draw.rectangle(gt_box, outline=(40, 180, 80), width=4)
        draw.text((gt_box[0] + 3, gt_box[1] + 2), CLASS_NAMES[int(l)], fill=(40, 180, 80))
    for b, score, label in zip(pred["boxes"][keep], pred["scores"][keep], pred["labels"][keep]):
        draw.rectangle(to_pixels(b, scale), outline=(40, 100, 230), width=3)
        draw.text(
            (to_pixels(b, scale)[0] + 3, to_pixels(b, scale)[3] - 13),
            f"{CLASS_NAMES[int(label)]} {float(score):.2f}",
            fill=(40, 100, 230),
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pil.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
