from __future__ import annotations

import argparse
import ast
import csv
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import build_convnext_base


LABEL_TO_INDEX = {"real_face": 0, "fake_face": 1}


class DevFaceClassificationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.transform = transform
        self.rows: List[Tuple[str, str, Tuple[int, int, int, int], int, int]] = []

        with self.manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"image_id", "image_path", "crop_bbox", "face_label", "face_id"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

            for row in reader:
                face_label = row["face_label"]
                if face_label not in LABEL_TO_INDEX:
                    raise ValueError(f"Unexpected face_label={face_label!r} in {self.manifest_path}")
                self.rows.append(
                    (
                        row["image_id"],
                        row["image_path"],
                        self._parse_bbox(row["crop_bbox"]),
                        LABEL_TO_INDEX[face_label],
                        int(row["face_id"]),
                    )
                )

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _parse_bbox(raw: str) -> Tuple[int, int, int, int]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = ast.literal_eval(raw)
        if not isinstance(payload, (list, tuple)) or len(payload) != 4:
            raise ValueError(f"Invalid crop_bbox={raw!r}")
        x1, y1, x2, y2 = [int(round(float(v))) for v in payload]
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid non-positive crop_bbox={raw!r}")
        return x1, y1, x2, y2

    def __getitem__(self, index: int):
        image_id, image_path, crop_bbox, label, face_id = self.rows[index]
        with Image.open(image_path) as img:
            crop = img.convert("RGB").crop(crop_bbox)

        if self.transform is not None:
            crop = self.transform(crop)

        return crop, label, image_id, face_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a ConvNeXt-B classifier on DDL-I dev face crops.")
    parser.add_argument("--manifest", required=True, help="Path to dev_faces.csv.")
    parser.add_argument("--checkpoint", required=True, help="Path to classifier checkpoint.")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--predictions-out", default="", help="Optional CSV path for per-face predictions.")
    return parser.parse_args()


def compute_face_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    metrics: Dict[str, float] = {
        "face_acc": float(accuracy_score(labels, preds)),
        "face_precision": float(precision),
        "face_recall": float(recall),
        "face_f1": float(f1),
    }

    if len(np.unique(labels)) == 2:
        metrics["face_auc"] = float(roc_auc_score(labels, probs))
        metrics["face_ap"] = float(average_precision_score(labels, probs))
    else:
        metrics["face_auc"] = float("nan")
        metrics["face_ap"] = float("nan")

    return metrics


def write_predictions(path: str | Path, image_ids: List[str], face_ids: List[int], labels: np.ndarray, probs: np.ndarray, threshold: float) -> None:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds = (probs >= threshold).astype(np.int64)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "face_id",
                "label",
                "pred",
                "fake_prob",
                "correct",
            ],
        )
        writer.writeheader()
        for image_id, face_id, label, pred, prob in zip(image_ids, face_ids, labels, preds, probs):
            writer.writerow(
                {
                    "image_id": image_id,
                    "face_id": int(face_id),
                    "label": int(label),
                    "pred": int(pred),
                    "fake_prob": float(prob),
                    "correct": int(label) == int(pred),
                }
            )


def main() -> None:
    args = parse_args()
    tfms = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds = DevFaceClassificationDataset(args.manifest, transform=tfms)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_convnext_base(None)
    ckpt = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_image_ids: List[str] = []
    all_face_ids: List[int] = []

    with torch.no_grad():
        for images, labels, image_ids, face_ids in loader:
            images = images.to(device, non_blocking=True)
            autocast_ctx = torch.cuda.amp.autocast(enabled=args.amp) if device.type == "cuda" else nullcontext()
            with autocast_ctx:
                probs = torch.sigmoid(model(images)).squeeze(1).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_image_ids.extend(list(image_ids))
            all_face_ids.extend([int(v) for v in face_ids.numpy().tolist()])

    probs_np = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels).astype(np.int64)
    metrics = compute_face_metrics(labels_np, probs_np, args.threshold)
    metrics["num_faces"] = int(labels_np.shape[0])
    metrics["num_real_faces"] = int((labels_np == 0).sum())
    metrics["num_fake_faces"] = int((labels_np == 1).sum())
    metrics["threshold"] = float(args.threshold)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.predictions_out:
        write_predictions(args.predictions_out, all_image_ids, all_face_ids, labels_np, probs_np, args.threshold)


if __name__ == "__main__":
    main()
