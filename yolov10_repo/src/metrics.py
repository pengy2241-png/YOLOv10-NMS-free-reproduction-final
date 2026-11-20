"""Detection metrics and NMS utilities."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2], dim=-1
    )


def box_iou_one(box: torch.Tensor, boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros(0, device=boxes.device)
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return inter / (area1 + area2 - inter + eps)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou_one(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def class_aware_nms(pred: dict[str, torch.Tensor], iou_threshold: float = 0.5) -> dict[str, torch.Tensor]:
    boxes, scores, labels = pred["boxes"], pred["scores"], pred["labels"]
    kept_indices = []
    for cls in labels.unique().tolist():
        cls_idx = (labels == int(cls)).nonzero(as_tuple=False).flatten()
        keep_local = nms(boxes[cls_idx], scores[cls_idx], iou_threshold)
        kept_indices.append(cls_idx[keep_local])
    if not kept_indices:
        return {"boxes": boxes[:0], "scores": scores[:0], "labels": labels[:0]}
    keep = torch.cat(kept_indices)
    keep = keep[scores[keep].argsort(descending=True)]
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}


def _match_image(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Return per-class TP/FP flags and scores for one image."""
    out = {}
    for cls in range(int(max(pred_labels.max().item() if pred_labels.numel() else -1, gt_labels.max().item() if gt_labels.numel() else -1)) + 1):
        p_mask = pred_labels == cls
        g_mask = gt_labels == cls
        p_boxes = pred_boxes[p_mask]
        p_scores = pred_scores[p_mask]
        g_boxes = gt_boxes[g_mask]
        order = p_scores.argsort(descending=True)
        p_boxes = p_boxes[order]
        p_scores = p_scores[order]
        matched = torch.zeros(g_boxes.shape[0], dtype=torch.bool)
        tp = torch.zeros(p_boxes.shape[0], dtype=torch.bool)
        for i in range(p_boxes.shape[0]):
            if g_boxes.shape[0] == 0:
                continue
            ious = box_iou_one(p_boxes[i], g_boxes)
            ious[matched] = -1
            best_iou, best_idx = ious.max(dim=0)
            if float(best_iou) >= iou_threshold:
                tp[i] = True
                matched[int(best_idx)] = True
        out[cls] = (tp, p_scores)
    return out


def average_precision(tp: torch.Tensor, scores: torch.Tensor, num_gt: int) -> float:
    if num_gt == 0:
        return float("nan")
    if scores.numel() == 0:
        return 0.0
    order = scores.argsort(descending=True)
    tp = tp[order].float()
    fp = 1 - tp
    tp_cum = tp.cumsum(0)
    fp_cum = fp.cumsum(0)
    recall = tp_cum / max(num_gt, 1)
    precision = tp_cum / (tp_cum + fp_cum).clamp(min=1e-7)
    # All-point interpolated AP.
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    for i in range(mpre.numel() - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])
    idx = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=False).flatten()
    return float(((mrec[1:][idx] - mrec[:-1][idx]) * mpre[1:][idx]).sum())


def evaluate_detections(
    predictions: list[dict[str, torch.Tensor]],
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    num_classes: int = 3,
    iou_thresholds: Iterable[float] = (0.5,),
) -> dict[str, float]:
    aps = {}
    for iou_thr in iou_thresholds:
        per_class = []
        all_tp = defaultdict(list)
        all_scores = defaultdict(list)
        num_gt = defaultdict(int)
        for pred, boxes, labels in zip(predictions, gt_boxes, gt_labels):
            boxes_xyxy = xywh_to_xyxy(boxes) if boxes.numel() else boxes.reshape(0, 4)
            for cls in labels.tolist():
                num_gt[int(cls)] += 1
            image_match = _match_image(
                pred["boxes"], pred["scores"], pred["labels"], boxes_xyxy, labels, float(iou_thr)
            )
            for cls, (tp, scores) in image_match.items():
                all_tp[cls].append(tp)
                all_scores[cls].append(scores)
        for cls in range(num_classes):
            tp = torch.cat(all_tp[cls]) if all_tp[cls] else torch.zeros(0)
            scores = torch.cat(all_scores[cls]) if all_scores[cls] else torch.zeros(0)
            per_class.append(average_precision(tp, scores, num_gt[cls]))
        aps[float(iou_thr)] = sum(x for x in per_class if x == x) / max(
            1, sum(1 for x in per_class if x == x)
        )
    return {f"mAP@{k:.2f}": v for k, v in aps.items()}


def summary_metrics(
    predictions: list[dict[str, torch.Tensor]],
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    num_classes: int = 3,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    ap50 = evaluate_detections(predictions, gt_boxes, gt_labels, num_classes, (0.5,))["mAP@0.50"]
    map5095 = evaluate_detections(
        predictions, gt_boxes, gt_labels, num_classes, [0.5 + 0.05 * i for i in range(10)]
    )
    map5095_value = sum(map5095.values()) / len(map5095)
    tp = fp = fn = 0
    for pred, boxes, labels in zip(predictions, gt_boxes, gt_labels):
        boxes_xyxy = xywh_to_xyxy(boxes) if boxes.numel() else boxes.reshape(0, 4)
        keep = pred["scores"] >= conf_threshold
        p_boxes, p_labels = pred["boxes"][keep], pred["labels"][keep]
        matched = torch.zeros(boxes_xyxy.shape[0], dtype=torch.bool)
        for i in range(p_boxes.shape[0]):
            candidates = (labels == int(p_labels[i])).nonzero(as_tuple=False).flatten()
            if candidates.numel() == 0:
                fp += 1
                continue
            ious = box_iou_one(p_boxes[i], boxes_xyxy[candidates])
            ious[matched[candidates]] = -1
            best_iou, best_local = ious.max(dim=0)
            if float(best_iou) >= iou_threshold:
                tp += 1
                matched[int(candidates[int(best_local)])] = True
            else:
                fp += 1
        fn += int((~matched).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-7)
    return {
        "mAP50": float(ap50),
        "mAP50_95": float(map5095_value),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
