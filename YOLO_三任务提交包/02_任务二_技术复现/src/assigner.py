"""Consistent dual assignment for one-to-many and one-to-one heads.

Both heads use the same candidate generation and matching cost. The only
difference is the number of cells each ground-truth object may claim:
- one-to-many: top-k=3
- one-to-one: top-k=1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class Assignment:
    cell_idx: int
    gt_idx: int
    cost: float


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """IoU for normalized xyxy boxes, shape [N,M]."""
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (area1[:, None] + area2[None, :] - inter + eps)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2], dim=-1
    )


def _distance_to_prior(pred_boxes_xyxy: torch.Tensor, gt_boxes_xyxy: torch.Tensor, grid: int) -> torch.Tensor:
    pred_xy = (pred_boxes_xyxy[:, :2] + pred_boxes_xyxy[:, 2:]) / 2
    gt_xy = (gt_boxes_xyxy[:, :2] + gt_boxes_xyxy[:, 2:]) / 2
    return torch.cdist(pred_xy * grid, gt_xy * grid)


def _candidate_cells(distance: torch.Tensor, gt_idx: int, k: int, radius: float) -> torch.Tensor:
    allowed = (distance[:, gt_idx] <= radius).nonzero(as_tuple=False).flatten()
    if allowed.numel() >= k:
        return allowed
    # Fallback: keep the nearest cells so every GT always receives supervision.
    return distance[:, gt_idx].topk(min(k, distance.shape[0])).indices


def assign_image(
    pred_boxes_xyxy: torch.Tensor,
    pred_obj_logits: torch.Tensor,
    pred_cls_logits: torch.Tensor,
    gt_boxes_xywh: torch.Tensor,
    gt_labels: torch.Tensor,
    grid: int,
    topk: int,
    radius: float = 2.5,
) -> list[Assignment]:
    """Assign cells to GT boxes with a shared cost and a greedy uniqueness rule."""
    if gt_boxes_xywh.numel() == 0:
        return []
    gt_xyxy = xywh_to_xyxy(gt_boxes_xywh).clamp(0, 1)
    iou = pairwise_iou(pred_boxes_xyxy, gt_xyxy)
    l1 = torch.cdist(pred_boxes_xyxy, gt_xyxy, p=1)
    distance = _distance_to_prior(pred_boxes_xyxy, gt_xyxy, grid)

    # Classification cost is shared by both branches. It uses the target class
    # logit rather than a separately trained assignment network.
    log_cls = torch.log_softmax(pred_cls_logits, dim=-1)
    cls_cost = torch.stack(
        [-log_cls[:, int(cls)] for cls in gt_labels.tolist()], dim=1
    )
    obj_cost = -torch.nn.functional.logsigmoid(pred_obj_logits)[:, None]
    cost = 2.0 * cls_cost + 1.0 * obj_cost + 3.0 * l1 + 2.0 * (1.0 - iou)

    pairs = []
    for gt_idx in range(gt_boxes_xywh.shape[0]):
        candidates = _candidate_cells(distance, gt_idx, topk, radius)
        for cell in candidates.tolist():
            pairs.append((float(cost[cell, gt_idx].detach().cpu()), int(cell), int(gt_idx)))
    pairs.sort(key=lambda x: x[0])

    assignments: list[Assignment] = []
    used_cells: set[int] = set()
    counts = [0] * gt_boxes_xywh.shape[0]
    for c, cell, gt_idx in pairs:
        if counts[gt_idx] >= topk or cell in used_cells:
            continue
        assignments.append(Assignment(cell_idx=cell, gt_idx=gt_idx, cost=c))
        used_cells.add(cell)
        counts[gt_idx] += 1
    return assignments


def assign_batch(
    head: dict[str, torch.Tensor],
    gt_boxes: Iterable[torch.Tensor],
    gt_labels: Iterable[torch.Tensor],
    grid: int,
    topk: int,
) -> list[list[Assignment]]:
    pred_boxes = head["boxes_xyxy"]  # [B,N,4]
    pred_obj = head["obj_logits"]     # [B,N]
    pred_cls = head["cls_logits"]     # [B,N,C]
    out = []
    for i, (boxes_i, labels_i) in enumerate(zip(gt_boxes, gt_labels)):
        out.append(
            assign_image(
                pred_boxes[i], pred_obj[i], pred_cls[i], boxes_i, labels_i, grid=grid, topk=topk
            )
        )
    return out
