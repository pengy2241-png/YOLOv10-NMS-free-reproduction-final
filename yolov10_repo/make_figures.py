"""Create report figures and a terminal-style run screenshot."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from src.data import CLASS_NAMES, ShapesDetectionDataset

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SHOTS = RESULTS / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)

FONT_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_REG = "/System/Library/Fonts/STHeiti Light.ttc"
FONT_MONO = "/System/Library/Fonts/SFNSMono.ttf"

NAVY = (23, 42, 70)
TEAL = (14, 124, 123)
ORANGE = (215, 106, 44)
BLUE = (45, 91, 154)
GREEN = (46, 125, 50)
RED = (190, 54, 54)
GRAY = (91, 101, 115)
LIGHT = (246, 243, 236)
WHITE = (255, 255, 255)


def font(size: int, bold: bool = False, mono: bool = False):
    path = FONT_MONO if mono else (FONT_BOLD if bold else FONT_REG)
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def text(draw, xy, s, size=24, fill=NAVY, bold=False, anchor=None, mono=False):
    draw.text(xy, s, font=font(size, bold, mono), fill=fill, anchor=anchor)


def rect(draw, box, outline, width=3):
    draw.rectangle(box, outline=outline, width=width)


def load_history(*paths):
    for path in paths:
        if path.exists():
            return json.loads(path.read_text())["history"]
    raise FileNotFoundError(f"no history found in {paths}")


def draw_training_curve():
    stage1 = load_history(RESULTS / "train_log_stage1.json", RESULTS / "stage1" / "train_log.json")
    stage2 = load_history(RESULTS / "train_log_stage2.json", RESULTS / "stage2" / "train_log.json")
    stage3 = load_history(RESULTS / "train_log.json")
    hist = []
    for offset, stage in [(0, stage1), (len(stage1), stage2), (len(stage1) + len(stage2), stage3)]:
        for h in stage:
            hist.append({**h, "epoch": h["epoch"] + offset})
    epochs = [h["epoch"] for h in hist]
    tr = [h.get("train_total", 0) for h in hist]
    va = [h.get("val_total", 0) for h in hist]
    W, H = 1400, 760
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 95), fill=NAVY)
    text(d, (55, 27), "Training history | 40 + 20 + 10 epochs", 38, WHITE, True)
    text(d, (55, 67), "stage 1: dual-head base | stage 2: stronger O2O | stage 3: soft duplicate suppression", 18, (214, 226, 241))
    left, top, right, bottom = 100, 155, W - 55, H - 85
    d.rectangle((left, top, right, bottom), fill=(250, 250, 250), outline=(210, 215, 222))
    ymax = max(max(tr), max(va), 1e-6) * 1.08
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        d.line((left, y, right, y), fill=(225, 228, 233), width=1)
        text(d, (left - 20, y), f"{ymax * i / 5:.2f}", 19, GRAY, anchor="rm")
    n = max(len(epochs) - 1, 1)
    for values, color in [(tr, TEAL), (va, ORANGE)]:
        pts = []
        for x, y in zip(epochs, values):
            px = left + (right - left) * (x - epochs[0]) / n
            py = bottom - (bottom - top) * min(y / ymax, 1.0)
            pts.append((px, py))
        d.line(pts, fill=color, width=5, joint="curve")
    for boundary in (len(stage1), len(stage1) + len(stage2)):
        x = left + (right - left) * (boundary - epochs[0]) / n
        d.line((x, top, x, bottom), fill=(150, 160, 175), width=2)
        text(d, (x + 8, top + 12), f"stage boundary", 16, GRAY)
    text(d, (left, top - 38), "train_total", 22, TEAL, True)
    text(d, (left + 185, top - 38), "val_total", 22, ORANGE, True)
    text(d, ((left + right) / 2, H - 38), "cumulative epoch", 22, GRAY, anchor="mm")
    img.save(SHOTS / "training_curves.png")


def draw_metrics_dashboard():
    data = json.loads((RESULTS / "metrics.json").read_text())
    rows = data["results"]
    W, H = 1500, 820
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 100), fill=NAVY)
    text(d, (50, 25), "Controlled comparison | one-to-one NMS-free vs one-to-many + NMS", 34, WHITE, True)
    # two panels
    panel1 = (50, 140, 720, 760)
    panel2 = (780, 140, 1450, 760)
    d.rounded_rectangle(panel1, radius=24, fill=(249, 249, 249), outline=(215, 220, 228), width=2)
    d.rounded_rectangle(panel2, radius=24, fill=(249, 249, 249), outline=(215, 220, 228), width=2)
    text(d, (85, 175), "COCO-style mAP@0.50", 27, NAVY, True)
    text(d, (815, 175), "End-to-end latency (ms, CPU batch=1)", 27, NAVY, True)

    colors = [BLUE, ORANGE, TEAL]
    labels = ["O2O\nNMS-free", "O2O\n+ NMS", "O2M\n+ NMS"]
    values = [r["mAP50"] for r in rows]
    maxv = max(max(values), 0.01) * 1.15
    base_y = 700
    for i, (label, val, color) in enumerate(zip(labels, values, colors)):
        x = 120 + i * 190
        height = (base_y - 260) * val / maxv
        d.rectangle((x, base_y - height, x + 110, base_y), fill=color)
        text(d, (x + 55, base_y - height - 25), f"{val:.3f}", 24, NAVY, True, anchor="mm")
        for j, line in enumerate(label.split("\n")):
            text(d, (x + 55, base_y + 25 + j * 27), line, 20, GRAY, anchor="mm")
    d.line((90, base_y, 680, base_y), fill=(170, 178, 188), width=2)

    lat = [r["latency_ms"] for r in rows]
    maxl = max(max(lat), 0.01) * 1.15
    for i, (label, val, color) in enumerate(zip(labels, lat, colors)):
        x = 850 + i * 190
        height = (base_y - 260) * val / maxl
        d.rectangle((x, base_y - height, x + 110, base_y), fill=color)
        text(d, (x + 55, base_y - height - 25), f"{val:.2f}", 24, NAVY, True, anchor="mm")
        for j, line in enumerate(label.split("\n")):
            text(d, (x + 55, base_y + 25 + j * 27), line, 20, GRAY, anchor="mm")
    d.line((820, base_y, 1410, base_y), fill=(170, 178, 188), width=2)
    img.save(SHOTS / "metrics_dashboard.png")


def _to_pixel(box, size):
    x1, y1, x2, y2 = [float(v) * size for v in box]
    return (int(x1), int(y1), int(x2), int(y2))


def draw_qualitative():
    saved = torch.load(RESULTS / "predictions_val.pt", map_location="cpu", weights_only=False)
    ds = ShapesDetectionDataset(192, size=128, seed=42 + 10000, train=False)
    indices = [0, 1, 2, 3]
    tile = 520
    W, H = tile * 2, tile * 2 + 100
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 82), fill=NAVY)
    text(d, (36, 20), "Qualitative validation outputs", 35, WHITE, True)
    text(d, (36, 58), "green: GT | blue: O2O NMS-free | orange: O2M + NMS", 20, (214, 226, 241))
    for n, idx in enumerate(indices):
        image = ds.items[idx][0]
        arr = ((image * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
        pil = Image.fromarray(arr).resize((tile, tile), Image.Resampling.BILINEAR).convert("RGB")
        x0 = (n % 2) * tile
        y0 = 100 + (n // 2) * tile
        img.paste(pil, (x0, y0))
        od = ImageDraw.Draw(img)
        gt_b = saved["gt_boxes"][idx]
        gt_l = saved["gt_labels"][idx]
        for b, l in zip(gt_b, gt_l):
            cx, cy, bw, bh = [float(v) for v in b]
            gt_xyxy = torch.tensor([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            gt_box = _to_pixel(gt_xyxy, tile)
            od.rectangle(gt_box, outline=GREEN, width=5)
            text(od, (gt_box[0] + 4, gt_box[1] + 3), CLASS_NAMES[int(l)], 15, GREEN, True)
        for key, color, prefix in [("o2o_nmsfree", BLUE, "O2O"), ("o2m_nms", ORANGE, "O2M")]:
            pred = saved[key][idx]
            keep = pred["scores"] >= 0.20
            for b, s, l in zip(pred["boxes"][keep][:8], pred["scores"][keep][:8], pred["labels"][keep][:8]):
                box = _to_pixel(b, tile)
                od.rectangle(box, outline=color, width=4)
                text(od, (box[0] + 3, box[3] - 20), f"{prefix} {float(s):.2f}", 13, color, True)
    img.save(SHOTS / "qualitative_predictions.png")


def draw_terminal():
    log = (RESULTS / "training_log.txt").read_text().strip().splitlines()
    eval_rows = json.loads((RESULTS / "metrics.json").read_text())["results"]
    lines = log[-28:] + ["", "evaluation:"]
    for r in eval_rows:
        lines.append(
            f"{r['mode']:12s} mAP50={r['mAP50']:.4f} mAP50-95={r['mAP50_95']:.4f} "
            f"latency={r['latency_ms']:.2f}ms fps={r['fps']:.1f}"
        )
    W = 1500
    H = 96 + len(lines) * 31
    img = Image.new("RGB", (W, H), (18, 24, 33))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 54), fill=(31, 40, 52))
    text(d, (25, 13), "YOLOv10-style reproduction | actual terminal transcript", 23, (229, 235, 242), True, mono=True)
    for i, line in enumerate(lines):
        color = (115, 210, 176) if "mAP50=" in line else (222, 229, 236)
        text(d, (25, 76 + i * 31), line, 19, color, mono=True)
    img.save(SHOTS / "terminal_output.png")


if __name__ == "__main__":
    draw_training_curve()
    draw_metrics_dashboard()
    draw_qualitative()
    draw_terminal()
    print("saved figures to", SHOTS)
