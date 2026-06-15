from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def parse_box(raw: str) -> list[int]:
    return [int(round(float(v))) for v in json.loads(raw)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="0")
    ap.add_argument("--predict-conf", type=float, default=0.001)
    ap.add_argument("--predict-iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=20)
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    model = YOLO(args.model)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    detection_count = 0
    error_count = 0

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "face_id", "conf", "bbox"])
        writer.writeheader()
        for start in range(0, len(rows), args.batch):
            batch_rows = rows[start : start + args.batch]
            crops = []
            usable_rows = []
            for row in batch_rows:
                try:
                    crop_box = parse_box(row["crop_bbox"])
                    with Image.open(row["image_path"]) as image:
                        crop = image.convert("RGB").crop(tuple(crop_box)).copy()
                    crops.append(crop)
                    usable_rows.append((row, crop_box))
                except Exception as exc:
                    error_count += 1
                    print(f"[warning] skip {row.get('image_id', '?')}: {exc}", flush=True)
            if not crops:
                continue
            results = model.predict(
                source=crops,
                imgsz=args.imgsz,
                conf=args.predict_conf,
                iou=args.predict_iou,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
            )
            for (row, crop_box), pred in zip(usable_rows, results):
                if pred.boxes is None:
                    continue
                ox, oy = crop_box[0], crop_box[1]
                boxes = pred.boxes.xyxy.cpu().numpy().tolist()
                scores = pred.boxes.conf.cpu().numpy().tolist()
                for box, score in zip(boxes, scores):
                    full_box = [
                        int(round(box[0])) + ox,
                        int(round(box[1])) + oy,
                        int(round(box[2])) + ox,
                        int(round(box[3])) + oy,
                    ]
                    writer.writerow(
                        {
                            "image_id": row["image_id"],
                            "face_id": row.get("face_id", ""),
                            "conf": float(score),
                            "bbox": json.dumps(full_box),
                        }
                    )
                    detection_count += 1
            if start and start % 5000 == 0:
                print(f"[predict] {start}/{len(rows)} detections={detection_count}", flush=True)

    summary = {
        "faces": len(rows),
        "detections": detection_count,
        "errors": error_count,
        "seconds": time.time() - start_time,
        "out_csv": str(out_path),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
