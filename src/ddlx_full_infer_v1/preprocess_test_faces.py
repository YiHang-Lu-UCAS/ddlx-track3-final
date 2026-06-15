from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


FACE_FIELDS = [
    "image_id",
    "image_path",
    "width",
    "height",
    "face_id",
    "face_bbox",
    "crop_bbox",
    "det_confidence",
    "landmarks",
    "status",
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_mtcnn(device: str):
    try:
        from facenet_pytorch import MTCNN
    except ImportError as exc:
        raise RuntimeError("facenet_pytorch is required for test face preprocessing.") from exc
    return MTCNN(keep_all=True, device=device, thresholds=[0.6, 0.7, 0.7], min_face_size=20, post_process=False)


def expand_bbox(box: list[float], scale: float, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, x2 - x1) * scale
    bh = max(1.0, y2 - y1) * scale
    out = [
        max(0, int(round(cx - bw / 2.0))),
        max(0, int(round(cy - bh / 2.0))),
        min(width, int(round(cx + bw / 2.0))),
        min(height, int(round(cy + bh / 2.0))),
    ]
    if out[2] <= out[0]:
        out[2] = min(width, out[0] + 1)
    if out[3] <= out[1]:
        out[3] = min(height, out[1] + 1)
    return out


def detect_faces(mtcnn, image: Image.Image) -> tuple[list[list[float]], list[float], list[Any]]:
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
    if boxes is None:
        return [], [], []
    return boxes.tolist(), probs.tolist(), landmarks.tolist() if landmarks is not None else [None] * len(boxes)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect faces and landmarks for image-only DDL-X test inference.")
    parser.add_argument("--test-images", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--crop-scale", type=float, default=1.4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rows = load_rows(args.test_images.expanduser().resolve())
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_face_outputs" / "shard_00"
    raw_dir.mkdir(parents=True, exist_ok=True)
    faces_jsonl = raw_dir / "faces.jsonl"
    images_jsonl = raw_dir / "images.jsonl"

    if args.device == "auto":
        try:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu"
    else:
        device = args.device
    mtcnn = build_mtcnn(device)

    face_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    with faces_jsonl.open("w", encoding="utf-8") as face_handle, images_jsonl.open("w", encoding="utf-8") as image_handle:
        for idx, row in enumerate(rows, start=1):
            image_id = row["image_id"]
            image_path = row["image_path"]
            width = int(float(row["width"]))
            height = int(float(row["height"]))
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                boxes, probs, landmarks = detect_faces(mtcnn, rgb)

            accepted = 0
            for face_id, (box, prob, lm) in enumerate(zip(boxes, probs, landmarks)):
                crop_bbox = expand_bbox([float(v) for v in box], args.crop_scale, width, height)
                payload = {
                    "image_id": image_id,
                    "image_path": image_path,
                    "width": width,
                    "height": height,
                    "face_id": face_id,
                    "face_bbox": json.dumps([float(v) for v in box]),
                    "crop_bbox": json.dumps(crop_bbox),
                    "det_confidence": float(prob) if prob is not None else 0.0,
                    "landmarks": json.dumps(lm),
                    "status": "accepted",
                }
                face_rows.append(payload)
                face_handle.write(
                    json.dumps(
                        {
                            **payload,
                            "face_bbox": [float(v) for v in box],
                            "crop_bbox": crop_bbox,
                            "landmarks": lm,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                accepted += 1

            image_payload = {
                "image_id": image_id,
                "image_path": image_path,
                "width": width,
                "height": height,
                "faces_detected": len(boxes),
                "accepted_faces": accepted,
            }
            image_rows.append(image_payload)
            image_handle.write(json.dumps(image_payload, ensure_ascii=False) + "\n")
            if idx % 1000 == 0:
                print(f"[preprocess] {idx}/{len(rows)} images", flush=True)

    write_csv(out_dir / "face_manifest_accepted.csv", face_rows)
    summary = {
        "test_images": str(args.test_images),
        "out_dir": str(out_dir),
        "images": len(rows),
        "faces": len(face_rows),
        "images_without_faces": sum(1 for row in image_rows if int(row["accepted_faces"]) == 0),
        "device": device,
        "crop_scale": args.crop_scale,
    }
    (out_dir / "preprocess_test_faces_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
