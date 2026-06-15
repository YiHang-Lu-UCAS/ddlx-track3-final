from __future__ import annotations

import argparse
import csv
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from data import DDLIFaceClassificationDataset, count_labels
from model import build_convnext_base
from utils import append_csv_row, is_main_process, save_json, seed_everything, strip_module_prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvNeXt-B face-level classifier for DDL-I.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained-path", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def setup_distributed() -> Tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def build_transforms(input_size: int):
    train_tfms = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, val_tfms


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return value
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / world_size


def gather_numpy(values: np.ndarray, world_size: int) -> np.ndarray:
    if world_size == 1:
        return values
    tensor = torch.as_tensor(values, device="cuda")
    sizes = [torch.zeros(1, device="cuda", dtype=torch.long) for _ in range(world_size)]
    local_size = torch.tensor([tensor.shape[0]], device="cuda", dtype=torch.long)
    dist.all_gather(sizes, local_size)
    max_size = max(int(size.item()) for size in sizes)
    if tensor.shape[0] < max_size:
        pad_shape = (max_size - tensor.shape[0],) + tensor.shape[1:]
        tensor = torch.cat([tensor, torch.zeros(pad_shape, device="cuda", dtype=tensor.dtype)], dim=0)
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    arrays = []
    for gathered_tensor, size in zip(gathered, sizes):
        arrays.append(gathered_tensor[: int(size.item())].cpu().numpy())
    return np.concatenate(arrays, axis=0)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    return {
        "acc": float(accuracy_score(labels, preds)),
        "auc": float(roc_auc_score(labels, probs)),
        "ap": float(average_precision_score(labels, probs)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, scaler, epoch: int, best: Dict[str, float], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": strip_module_prefix(model.state_dict()),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best": best,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    seed_everything(args.seed + rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tfms, val_tfms = build_transforms(args.input_size)
    train_ds = DDLIFaceClassificationDataset(args.train_manifest, args.dataset_root, transform=train_tfms)
    val_ds = DDLIFaceClassificationDataset(args.val_manifest, args.dataset_root, transform=val_tfms)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = build_convnext_base(args.pretrained_path or None).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    counts = count_labels(args.train_manifest)
    pos_weight = torch.tensor([counts["real_face"] / max(1, counts["fake_face"])], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    best = {"auc": -math.inf, "acc": -math.inf}
    raw_model = model.module if isinstance(model, DDP) else model
    if args.resume:
        ckpt = torch.load(Path(args.resume).expanduser(), map_location="cpu")
        raw_model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best = dict(ckpt.get("best", best))

    if is_main_process():
        save_json(
            output_dir / "run_config.json",
            {
                "args": vars(args),
                "world_size": world_size,
                "train_size": len(train_ds),
                "val_size": len(val_ds),
                "train_label_counts": counts,
                "effective_global_batch_size": args.batch_size * world_size,
            },
        )

    metrics_fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "acc",
        "auc",
        "ap",
        "precision",
        "recall",
        "f1",
        "lr",
        "epoch_seconds",
    ]

    train_started_at = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        seen = 0
        for step, (images, labels, _, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.cuda.amp.autocast(enabled=args.amp) if device.type == "cuda" else nullcontext()
            with autocast_ctx:
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            running_loss += loss.detach().item() * batch_size
            seen += batch_size
            if is_main_process() and step % args.log_interval == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} train_loss={running_loss / max(1, seen):.5f}", flush=True)

        train_loss = torch.tensor(running_loss / max(1, seen), device=device)
        train_loss = reduce_mean(train_loss, world_size).item()

        model.eval()
        val_running_loss = 0.0
        val_seen = 0
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True).unsqueeze(1)
                autocast_ctx = torch.cuda.amp.autocast(enabled=args.amp) if device.type == "cuda" else nullcontext()
                with autocast_ctx:
                    logits = model(images)
                    loss = criterion(logits, labels)
                probs = torch.sigmoid(logits).squeeze(1)
                val_running_loss += loss.detach().item() * images.size(0)
                val_seen += images.size(0)
                all_probs.append(probs.detach().cpu().numpy())
                all_labels.append(labels.squeeze(1).detach().cpu().numpy())

        val_loss = torch.tensor(val_running_loss / max(1, val_seen), device=device)
        val_loss = reduce_mean(val_loss, world_size).item()
        probs_np = gather_numpy(np.concatenate(all_probs), world_size)
        labels_np = gather_numpy(np.concatenate(all_labels), world_size).astype(np.int64)

        if is_main_process():
            metrics = compute_metrics(labels_np, probs_np, args.threshold)
            epoch_seconds = time.time() - epoch_start
            avg_epoch_seconds = (time.time() - train_started_at) / max(1, epoch - start_epoch + 1)
            remaining_seconds = avg_epoch_seconds * max(0, args.epochs - epoch - 1)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **metrics,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": epoch_seconds,
            }
            append_csv_row(output_dir / "metrics.csv", row, metrics_fields)
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                f"acc={metrics['acc']:.4f} auc={metrics['auc']:.4f} "
                f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
                f"f1={metrics['f1']:.4f} epoch_sec={epoch_seconds:.1f} "
                f"eta_hours={remaining_seconds / 3600:.2f}",
                flush=True,
            )
            improved_auc = metrics["auc"] > best["auc"]
            improved_acc = metrics["acc"] > best["acc"]
            if improved_auc:
                best["auc"] = metrics["auc"]
            if improved_acc:
                best["acc"] = metrics["acc"]
            save_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, scaler, epoch, best, args)
            if improved_auc:
                save_checkpoint(output_dir / "checkpoints" / "best_auc.pt", model, optimizer, scheduler, scaler, epoch, best, args)
            if improved_acc:
                save_checkpoint(output_dir / "checkpoints" / "best_acc.pt", model, optimizer, scheduler, scaler, epoch, best, args)

        scheduler.step()
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
