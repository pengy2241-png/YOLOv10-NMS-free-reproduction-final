"""Synthetic Shapes-DET dataset used by the simplified YOLOv10 reproduction.

The dataset is intentionally small and controlled. It isolates the core
methodological question (one-to-many + NMS vs. one-to-one NMS-free) without
requiring a GPU or a large benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

CLASS_NAMES = ("circle", "square", "triangle")
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class DetectionTarget:
    boxes: torch.Tensor  # normalized xywh, shape [N, 4]
    labels: torch.Tensor  # shape [N]


def _random_color(rng: np.random.Generator) -> tuple[int, int, int]:
    palette = [
        (235, 72, 72),
        (62, 137, 235),
        (72, 190, 120),
        (244, 164, 58),
        (175, 92, 220),
        (40, 190, 210),
    ]
    base = np.array(palette[int(rng.integers(0, len(palette)))], dtype=np.float32)
    jitter = rng.normal(0, 8, size=3)
    return tuple(np.clip(base + jitter, 0, 255).astype(np.uint8).tolist())


def _overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    # boxes are xyxy in pixels
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _make_background(rng: np.random.Generator, size: int) -> np.ndarray:
    base = rng.uniform(205, 245, size=3)
    yy, xx = np.mgrid[0:size, 0:size]
    direction = rng.uniform(-1.0, 1.0, size=2)
    gradient = (
        direction[0] * (xx / size - 0.5)
        + direction[1] * (yy / size - 0.5)
    ) * rng.uniform(4, 12)
    noise = rng.normal(0, 2.2, size=(size, size, 3))
    image = base.reshape(1, 1, 3) + gradient[..., None] + noise
    return np.clip(image, 0, 255).astype(np.uint8)


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    cls_id: int,
    box: np.ndarray,
    fill: tuple[int, int, int],
    rng: np.random.Generator,
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
    outline = tuple(int(v * 0.55) for v in fill)
    width = max(1, int(round(min(x2 - x1, y2 - y1) * 0.06)))
    if cls_id == 0:  # circle
        draw.ellipse((x1, y1, x2 - 1, y2 - 1), fill=fill, outline=outline, width=width)
    elif cls_id == 1:  # square / rectangle
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=fill, outline=outline, width=width)
    else:  # triangle
        # Jitter the three vertices slightly so the class does not collapse to
        # one exact template.
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half_w, half_h = (x2 - x1) / 2, (y2 - y1) / 2
        top = (cx + rng.normal(0, 1.2), y1)
        left = (x1, cy + rng.normal(0, 1.2))
        right = (x2 - 1, cy + rng.normal(0, 1.2))
        draw.polygon([top, left, right], fill=fill, outline=outline, width=width)


def make_sample(index: int, size: int, seed: int, train: bool) -> tuple[torch.Tensor, DetectionTarget]:
    rng = np.random.default_rng(seed + index * 7919)
    image = _make_background(rng, size)
    pil = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(pil, "RGBA")

    n_objects = int(rng.integers(1, 4)) if train else int(rng.integers(1, 4))
    boxes_px: list[np.ndarray] = []
    labels: list[int] = []
    attempts = 0
    while len(boxes_px) < n_objects and attempts < 80:
        attempts += 1
        cls_id = int(rng.integers(0, len(CLASS_NAMES)))
        side = rng.uniform(17, 38)
        aspect = rng.uniform(0.82, 1.18)
        w = side * aspect
        h = side / aspect
        margin = 5
        if w >= size - 2 * margin or h >= size - 2 * margin:
            continue
        x1 = rng.uniform(margin, size - margin - w)
        y1 = rng.uniform(margin, size - margin - h)
        box = np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)
        if any(_overlap_ratio(box, old) > 0.20 for old in boxes_px):
            continue
        boxes_px.append(box)
        labels.append(cls_id)
        _draw_shape(draw, cls_id, box, _random_color(rng), rng)

    if not boxes_px:
        # Guaranteed fallback keeps the dataset well-formed in the unlikely
        # event that every random candidate was rejected.
        box = np.array([size * 0.35, size * 0.35, size * 0.65, size * 0.65], dtype=np.float32)
        boxes_px.append(box)
        labels.append(0)
        _draw_shape(draw, 0, box, _random_color(rng), rng)

    # A light post-draw noise pass makes the task less template-like.
    arr = np.asarray(pil, dtype=np.float32)
    arr += rng.normal(0, 2.0, size=arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5

    boxes = torch.tensor(
        [[(x1 + x2) / 2 / size, (y1 + y2) / 2 / size, (x2 - x1) / size, (y2 - y1) / size]
         for x1, y1, x2, y2 in boxes_px],
        dtype=torch.float32,
    )
    target = DetectionTarget(boxes=boxes, labels=torch.tensor(labels, dtype=torch.long))
    return tensor, target


class ShapesDetectionDataset(Dataset):
    def __init__(self, length: int, size: int = 128, seed: int = 0, train: bool = True):
        self.length = int(length)
        self.size = int(size)
        self.seed = int(seed)
        self.train = bool(train)
        self.items = [make_sample(i, self.size, self.seed, self.train) for i in range(self.length)]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        image, target = self.items[index]
        return image, target.boxes.clone(), target.labels.clone()


def detection_collate(batch: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    images, boxes, labels = zip(*batch)
    return torch.stack(images, 0), list(boxes), list(labels)


def describe_dataset() -> dict:
    return {
        "name": "Shapes-DET",
        "classes": list(CLASS_NAMES),
        "image_size": 128,
        "train_samples": 768,
        "val_samples": 192,
        "objects_per_image": "1-3",
        "note": "Controlled synthetic benchmark for mechanism-level reproduction, not a COCO replacement.",
    }
