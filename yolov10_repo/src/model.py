"""A compact YOLOv10-style detector with dual prediction heads."""
from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.bn2(self.conv2(y))
        return self.act(x + y)


class TinyBackbone(nn.Module):
    """Stride-8 feature extractor; 128x128 input -> 16x16 feature map."""
    def __init__(self, width: int = 24):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, width, 2),
            ConvBNAct(width, width * 2, 2),
            ResidualBlock(width * 2),
            ConvBNAct(width * 2, width * 4, 2),
            ResidualBlock(width * 4),
            ResidualBlock(width * 4),
            nn.Conv2d(width * 4, width * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class PredictionHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.stem = ConvBNAct(in_ch, in_ch)
        self.out = nn.Conv2d(in_ch, 5 + num_classes, 1)
        self.num_classes = num_classes
        nn.init.constant_(self.out.bias[4], -4.0)  # objectness prior

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        y = self.out(self.stem(x))
        obj_logits, cls_logits, box_raw = torch.split(
            y, [1, self.num_classes, 4], dim=1
        )
        return {
            "obj_logits": obj_logits,
            "cls_logits": cls_logits,
            "box_raw": box_raw,
        }


class DualHeadDetector(nn.Module):
    """Shared backbone plus a one-to-many head and a one-to-one head."""
    def __init__(self, num_classes: int = 3, width: int = 24, image_size: int = 128, stride: int = 8):
        super().__init__()
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.stride = int(stride)
        self.grid_size = self.image_size // self.stride
        self.backbone = TinyBackbone(width)
        feat_ch = width * 4
        self.o2m_head = PredictionHead(feat_ch, num_classes)
        self.o2o_head = PredictionHead(feat_ch, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        feat = self.backbone(x)
        return {"o2m": self.o2m_head(feat), "o2o": self.o2o_head(feat)}

    def flatten_head(self, head: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obj = head["obj_logits"].permute(0, 2, 3, 1).reshape(x_flat := head["obj_logits"].shape[0], -1)
        cls = head["cls_logits"].permute(0, 2, 3, 1).reshape(x_flat, -1, self.num_classes)
        raw = head["box_raw"].permute(0, 2, 3, 1).reshape(x_flat, -1, 4)
        return {"obj_logits": obj, "cls_logits": cls, "box_raw": raw}

    def decode_head(
        self,
        head: Dict[str, torch.Tensor],
        conf_threshold: float = 0.001,
        topk: int = 100,
    ) -> list[dict[str, torch.Tensor]]:
        """Decode a head into per-image boxes, scores and labels.

        Box parameterization: the four raw values are sigmoid-transformed into
        normalized center offsets and widths/heights. It is deliberately simple
        and stable for a CPU-only teaching reproduction.
        """
        flat = self.flatten_head(head)
        obj_prob = torch.sigmoid(flat["obj_logits"])
        cls_prob = F.softmax(flat["cls_logits"], dim=-1)
        scores_all = obj_prob.unsqueeze(-1) * cls_prob

        device = scores_all.device
        ys, xs = torch.meshgrid(
            torch.arange(self.grid_size, device=device),
            torch.arange(self.grid_size, device=device),
            indexing="ij",
        )
        priors = torch.stack(
            [(xs.float() + 0.5) / self.grid_size, (ys.float() + 0.5) / self.grid_size],
            dim=-1,
        ).reshape(-1, 2)

        box_raw = flat["box_raw"]
        cx = (priors[None, :, 0] + torch.tanh(box_raw[..., 0]) / self.grid_size).clamp(0, 1)
        cy = (priors[None, :, 1] + torch.tanh(box_raw[..., 1]) / self.grid_size).clamp(0, 1)
        bw = (0.04 + 0.62 * torch.sigmoid(box_raw[..., 2])).clamp(0.02, 0.9)
        bh = (0.04 + 0.62 * torch.sigmoid(box_raw[..., 3])).clamp(0.02, 0.9)
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1).clamp(0, 1)

        results = []
        for i in range(scores_all.shape[0]):
            scores, labels = scores_all[i].max(dim=-1)
            keep = scores > conf_threshold
            if keep.sum() == 0:
                keep = scores > scores.max()
            boxes_i = boxes[i][keep]
            scores_i = scores[keep]
            labels_i = labels[keep]
            if scores_i.numel() > topk:
                scores_i, idx = scores_i.topk(topk)
                boxes_i = boxes_i[idx]
                labels_i = labels_i[idx]
            results.append({"boxes": boxes_i, "scores": scores_i, "labels": labels_i})
        return results


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(num_classes: int = 3, width: int = 24, image_size: int = 128) -> DualHeadDetector:
    return DualHeadDetector(num_classes=num_classes, width=width, image_size=image_size)
