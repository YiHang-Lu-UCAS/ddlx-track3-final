from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

class TestFaceDataset:
    def __init__(self, manifest: Path, transform: Callable | None = None) -> None:
        self.rows: list[dict[str, str]] = []
        self.transform = transform
        with manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"image_id", "image_path", "face_id", "crop_bbox"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing columns in face manifest: {sorted(missing)}")
            self.rows = list(reader)

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def parse_box(raw: str) -> tuple[int, int, int, int]:
        box = json.loads(raw)
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        return x1, y1, x2, y2

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            crop = image.convert("RGB").crop(self.parse_box(row["crop_bbox"]))
        if self.transform:
            crop = self.transform(crop)
        return crop, row["image_id"], str(row["face_id"]), row["crop_bbox"]


def load_test_images(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ConvNeXt-B classifier on test face crops.")
    parser.add_argument("--face-manifest", required=True, type=Path)
    parser.add_argument("--test-images", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--predictions-out", required=True, type=Path)
    parser.add_argument("--image-scores-out", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    try:
        from .model import build_convnext_base
    except ImportError:  # pragma: no cover - keeps direct script execution working
        from model import build_convnext_base

    tfms = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    pred_rows: list[dict[str, object]] = []
    probs_by_image: dict[str, list[float]] = defaultdict(list)
    ds = TestFaceDataset(args.face_manifest.expanduser().resolve(), transform=tfms)
    if len(ds) > 0:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_convnext_base(None)
        ckpt = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device).eval()

        with torch.no_grad():
            for images, image_ids, face_ids, crop_bboxes in loader:
                images = images.to(device, non_blocking=True)
                autocast_ctx = torch.cuda.amp.autocast(enabled=args.amp) if device.type == "cuda" else nullcontext()
                with autocast_ctx:
                    probs = torch.sigmoid(model(images)).squeeze(1).cpu().numpy()
                for image_id, face_id, crop_bbox, prob in zip(image_ids, face_ids, crop_bboxes, probs):
                    prob_f = float(prob)
                    probs_by_image[str(image_id)].append(prob_f)
                    pred_rows.append(
                        {
                            "image_id": str(image_id),
                            "face_id": str(face_id),
                            "fake_prob": prob_f,
                            "crop_bbox": str(crop_bbox),
                        }
                    )

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "face_id", "fake_prob", "crop_bbox"])
        writer.writeheader()
        writer.writerows(pred_rows)

    image_rows = load_test_images(args.test_images.expanduser().resolve())
    args.image_scores_out.parent.mkdir(parents=True, exist_ok=True)
    with args.image_scores_out.open("w", newline="", encoding="utf-8") as handle:
        fields = ["image_id", "image_path", "width", "height", "fake_prob", "classification", "num_faces"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in image_rows:
            probs = np.asarray(probs_by_image.get(row["image_id"], []), dtype=np.float64)
            score = float(probs.max()) if probs.size else 0.0
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "width": row["width"],
                    "height": row["height"],
                    "fake_prob": score,
                    "classification": "fake" if score >= args.threshold else "real",
                    "num_faces": int(probs.size),
                }
            )

    summary = {
        "faces": len(pred_rows),
        "images": len(image_rows),
        "images_with_faces": len(probs_by_image),
        "threshold": args.threshold,
        "predictions_out": str(args.predictions_out),
        "image_scores_out": str(args.image_scores_out),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
