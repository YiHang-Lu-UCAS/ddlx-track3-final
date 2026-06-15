from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from data import DDLIFaceClassificationDataset
from model import build_convnext_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a ConvNeXt-B DDL-I classifier checkpoint.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tfms = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = DDLIFaceClassificationDataset(args.manifest, args.dataset_root, transform=tfms)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_convnext_base(None)
    ckpt = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for images, labels, _, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.numpy()
            autocast_ctx = torch.cuda.amp.autocast(enabled=args.amp) if device.type == "cuda" else nullcontext()
            with autocast_ctx:
                probs = torch.sigmoid(model(images)).squeeze(1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= args.threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    print(
        {
            "acc": float(accuracy_score(labels, preds)),
            "auc": float(roc_auc_score(labels, probs)),
            "ap": float(average_precision_score(labels, probs)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    )


if __name__ == "__main__":
    main()

