"""Train the simplified YOLOv10-style dual-head detector."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.assigner import assign_batch
from src.data import CLASS_NAMES, ShapesDetectionDataset, detection_collate
from src.loss import flatten_head, head_loss
from src.model import build_model, count_parameters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train-size", type=int, default=768)
    parser.add_argument("--val-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--o2o-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    return parser.parse_args()


def make_loader(dataset, batch_size, shuffle, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate,
        drop_last=False,
    )


def compute_losses(model, images, gt_boxes, gt_labels, device, o2o_weight=1.0, hard_negative_weight=0.0):
    outputs = model(images.to(device))
    flat_o2m = flatten_head(outputs["o2m"], model.num_classes, model.grid_size)
    flat_o2o = flatten_head(outputs["o2o"], model.num_classes, model.grid_size)
    a_o2m = assign_batch(flat_o2m, gt_boxes, gt_labels, model.grid_size, topk=3)
    a_o2o = assign_batch(flat_o2o, gt_boxes, gt_labels, model.grid_size, topk=1)
    l_o2m = head_loss(flat_o2m, a_o2m, gt_boxes, gt_labels, model.num_classes)
    l_o2o = head_loss(flat_o2o, a_o2o, gt_boxes, gt_labels, model.num_classes, hard_negative_weight=hard_negative_weight)
    # The one-to-many branch provides dense supervision; the one-to-one branch
    # is the deployment head and receives the same matching metric.
    total = 2.0 * l_o2m["loss"] + float(o2o_weight) * l_o2o["loss"]
    return total, l_o2m, l_o2o


def mean_stats(stats: list[dict[str, float]], prefix: str) -> dict[str, float]:
    if not stats:
        return {}
    keys = [k for k in stats[0] if isinstance(stats[0][k], (int, float))]
    return {f"{prefix}{k}": sum(float(s[k]) for s in stats) / len(stats) for k in keys}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_size = 32 if args.smoke_test else args.train_size
    val_size = 16 if args.smoke_test else args.val_size
    epochs = 1 if args.smoke_test else args.epochs
    train_ds = ShapesDetectionDataset(train_size, size=args.image_size, seed=args.seed, train=True)
    val_ds = ShapesDetectionDataset(val_size, size=args.image_size, seed=args.seed + 10000, train=False)
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)

    log_lines = []
    model = build_model(num_classes=len(CLASS_NAMES), width=args.width, image_size=args.image_size).to(device)
    if args.resume:
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"resumed from {args.resume}", flush=True)
        log_lines.append(f"resumed from {args.resume}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    best_val = float("inf")
    history = []
    started = time.time()

    header = (
        f"device={device} | params={count_parameters(model):,} | "
        f"train={len(train_ds)} | val={len(val_ds)} | epochs={epochs} | batch={args.batch_size}"
    )
    print(header, flush=True)
    log_lines.append(header)

    for epoch in range(1, epochs + 1):
        model.train()
        train_stats = []
        for images, gt_boxes, gt_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            total, l_m, l_o = compute_losses(model, images, gt_boxes, gt_labels, device, args.o2o_weight, args.hard_negative_weight)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_stats.append(
                {
                    "total": float(total.detach().cpu()),
                    "o2m": float(l_m["loss"].detach().cpu()),
                    "o2o": float(l_o["loss"].detach().cpu()),
                    "o2m_pos": float(l_m["num_pos"].detach().cpu()),
                    "o2o_pos": float(l_o["num_pos"].detach().cpu()),
                }
            )

        model.eval()
        val_stats = []
        with torch.no_grad():
            for images, gt_boxes, gt_labels in val_loader:
                total, l_m, l_o = compute_losses(model, images, gt_boxes, gt_labels, device, args.o2o_weight, args.hard_negative_weight)
                val_stats.append(
                    {
                        "total": float(total.detach().cpu()),
                        "o2m": float(l_m["loss"].detach().cpu()),
                        "o2o": float(l_o["loss"].detach().cpu()),
                        "o2m_pos": float(l_m["num_pos"].detach().cpu()),
                        "o2o_pos": float(l_o["num_pos"].detach().cpu()),
                    }
                )

        scheduler.step()
        tr = mean_stats(train_stats, "train_")
        va = mean_stats(val_stats, "val_")
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **tr,
            **va,
        }
        history.append(row)
        val_loss = row.get("val_total", float("inf"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "class_names": list(CLASS_NAMES),
                    "epoch": epoch,
                    "best_val_loss": best_val,
                },
                out_dir / "best.pt",
            )
        line = (
            f"epoch {epoch:03d}/{epochs:03d} | "
            f"train {row.get('train_total', 0):.4f} | val {val_loss:.4f} | "
            f"o2m {row.get('val_o2m', 0):.4f} | o2o {row.get('val_o2o', 0):.4f} | "
            f"lr {row['lr']:.2e}"
        )
        print(line, flush=True)
        log_lines.append(line)

    elapsed = time.time() - started
    summary = {
        "args": vars(args),
        "class_names": list(CLASS_NAMES),
        "num_parameters": count_parameters(model),
        "elapsed_seconds": elapsed,
        "best_val_loss": best_val,
        "history": history,
    }
    (out_dir / "train_log.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "training_log.txt").write_text("\n".join(log_lines) + "\n")
    (out_dir / "environment.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "python": __import__("platform").python_version(),
                "device": str(device),
                "cpu_threads": torch.get_num_threads(),
            },
            indent=2,
        )
    )
    print(f"saved: {out_dir / 'best.pt'} | elapsed {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
