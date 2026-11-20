"""Training losses for the dual-head detector."""
from __future__ import annotations

from typing import Iterable

import torch
from torch.nn import functional as F

from .assigner import Assignment, pairwise_iou, xywh_to_xyxy


def raw_box_to_xyxy(raw_boxes: torch.Tensor, grid: int) -> torch.Tensor:
    """[B,N,4] raw box predictions -> normalized xyxy boxes."""
    device = raw_boxes.device
    ys, xs = torch.meshgrid(
        torch.arange(grid, device=device),
        torch.arange(grid, device=device),
        indexing="ij",
    )
    priors = torch.stack(
        [(xs.float() + 0.5) / grid, (ys.float() + 0.5) / grid], dim=-1
    ).reshape(-1, 2)
    cx = (priors[None, :, 0] + torch.tanh(raw_boxes[..., 0]) / grid).clamp(0, 1)
    cy = (priors[None, :, 1] + torch.tanh(raw_boxes[..., 1]) / grid).clamp(0, 1)
    bw = (0.04 + 0.62 * torch.sigmoid(raw_boxes[..., 2])).clamp(0.02, 0.9)
    bh = (0.04 + 0.62 * torch.sigmoid(raw_boxes[..., 3])).clamp(0.02, 0.9)
    return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1).clamp(0, 1)


def flatten_head(head: dict[str, torch.Tensor], num_classes: int, grid: int):
    obj = head["obj_logits"].permute(0, 2, 3, 1).reshape(head["obj_logits"].shape[0], -1)
    cls = head["cls_logits"].permute(0, 2, 3, 1).reshape(
        head["obj_logits"].shape[0], -1, num_classes
    )
    raw = head["box_raw"].permute(0, 2, 3, 1).reshape(head["obj_logits"].shape[0], -1, 4)
    boxes = raw_box_to_xyxy(raw, grid)
    return {
        "obj_logits": obj,
        "cls_logits": cls,
        "box_raw": raw,
        "boxes_xyxy": boxes,
    }


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t).pow(gamma) * ce).mean()


def giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    iou = pairwise_iou(pred_boxes, target_boxes, eps=eps).diag()
    lt = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
    rb = torch.min(pred_boxes[:, 2:], target_boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * (
        pred_boxes[:, 3] - pred_boxes[:, 1]
    ).clamp(min=0)
    area_t = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0) * (
        target_boxes[:, 3] - target_boxes[:, 1]
    ).clamp(min=0)
    union = area_p + area_t - inter
    iou = inter / (union + eps)
    enclose_lt = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
    enclose_rb = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose = enclose_wh[:, 0] * enclose_wh[:, 1]
    giou = iou - (enclose - union) / (enclose + eps)
    return 1 - giou


def head_loss(
    flat_head: dict[str, torch.Tensor],
    assignments: Iterable[list[Assignment]],
    gt_boxes: Iterable[torch.Tensor],
    gt_labels: Iterable[torch.Tensor],
    num_classes: int,
    hard_negative_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    device = flat_head["obj_logits"].device
    batch, num_cells = flat_head["obj_logits"].shape
    obj_target = torch.zeros(batch, num_cells, device=device)
    cls_target = torch.zeros(batch, num_cells, num_classes, device=device)
    box_target = torch.zeros(batch, num_cells, 4, device=device)
    pos_mask = torch.zeros(batch, num_cells, dtype=torch.bool, device=device)

    for b, image_assignments in enumerate(assignments):
        boxes_b = gt_boxes[b].to(device)
        labels_b = gt_labels[b].to(device)
        for a in image_assignments:
            obj_target[b, a.cell_idx] = 1.0
            cls_target[b, a.cell_idx, int(labels_b[a.gt_idx])] = 1.0
            box_target[b, a.cell_idx] = boxes_b[a.gt_idx]
            pos_mask[b, a.cell_idx] = True

    obj_loss = sigmoid_focal_loss(flat_head["obj_logits"], obj_target)
    if pos_mask.any():
        pos_cls = flat_head["cls_logits"][pos_mask]
        pos_cls_target = cls_target[pos_mask]
        cls_loss = F.cross_entropy(pos_cls, pos_cls_target.argmax(dim=-1))
        pos_boxes = flat_head["boxes_xyxy"][pos_mask]
        pos_box_target = xywh_to_xyxy(box_target[pos_mask]).clamp(0, 1)
        box_l1 = F.l1_loss(pos_boxes, pos_box_target)
        box_giou = giou_loss(pos_boxes, pos_box_target).mean()
    else:
        zero = flat_head["obj_logits"].sum() * 0.0
        cls_loss, box_l1, box_giou = zero, zero, zero

    hard_neg_loss = flat_head["obj_logits"].sum() * 0.0
    if hard_negative_weight > 0:
        grid = int(round(num_cells ** 0.5))
        neighbor_mask = torch.zeros_like(pos_mask)
        for b, image_assignments in enumerate(assignments):
            for a in image_assignments:
                row, col = divmod(a.cell_idx, grid)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = row + dr, col + dc
                        if 0 <= rr < grid and 0 <= cc < grid:
                            neighbor_mask[b, rr * grid + cc] = True
        neighbor_mask = neighbor_mask & (~pos_mask)
        if neighbor_mask.any():
            hard_neg_loss = F.binary_cross_entropy_with_logits(
                flat_head["obj_logits"][neighbor_mask],
                torch.zeros(int(neighbor_mask.sum()), device=device),
            )

    total = obj_loss + 2.0 * cls_loss + 5.0 * (box_l1 + box_giou) + float(hard_negative_weight) * hard_neg_loss
    return {
        "loss": total,
        "obj_loss": obj_loss.detach(),
        "hard_neg_loss": hard_neg_loss.detach(),
        "cls_loss": cls_loss.detach(),
        "box_l1": box_l1.detach(),
        "box_giou": box_giou.detach(),
        "num_pos": torch.tensor(float(pos_mask.sum()), device=device),
    }
