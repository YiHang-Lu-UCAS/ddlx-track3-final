from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any


def to1000(box: list[float], width: int, height: int) -> list[int] | None:
    out = [
        round(float(box[0]) / width * 999) + 1,
        round(float(box[1]) / height * 999) + 1,
        round(float(box[2]) / width * 999) + 1,
        round(float(box[3]) / height * 999) + 1,
    ]
    out = [int(max(1, min(1000, v))) for v in out]
    if out[2] <= out[0]:
        out[2] = min(1000, out[0] + 1)
    if out[3] <= out[1]:
        out[3] = min(1000, out[1] + 1)
    return out if out[2] > out[0] and out[3] > out[1] else None


def mk_box(cx: float, cy: float, width: float, height: float, iw: int, ih: int) -> list[int] | None:
    box = [
        max(0, int(round(cx - width / 2))),
        max(0, int(round(cy - height / 2))),
        min(iw, int(round(cx + width / 2))),
        min(ih, int(round(cy + height / 2))),
    ]
    return box if box[2] > box[0] and box[3] > box[1] else None


def landmark_boxes(landmarks: Any, iw: int, ih: int) -> list[list[int]]:
    if not isinstance(landmarks, list) or len(landmarks) < 5:
        return []
    pts = [[float(p[0]), float(p[1])] for p in landmarks[:5]]
    d = max(2.0, math.dist(pts[0], pts[1]))
    mouth = ((pts[3][0] + pts[4][0]) / 2.0, (pts[3][1] + pts[4][1]) / 2.0)
    specs = [
        (pts[2][0], pts[2][1] + 0.03 * d, 1.02 * d, 0.88 * d),
        (pts[0][0], pts[0][1] - 0.06 * d, 1.02 * d, 0.68 * d),
        (pts[1][0], pts[1][1] - 0.06 * d, 1.02 * d, 0.68 * d),
        (mouth[0], mouth[1], 1.16 * d, 0.74 * d),
    ]
    out: list[list[int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for cx, cy, bw, bh in specs:
        box = mk_box(cx, cy, bw, bh, iw, ih)
        if box and tuple(box) not in seen:
            seen.add(tuple(box))
            out.append(box)
    return out[:4]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_best_landmarks(raw_face_dir: Path, face_predictions: Path) -> dict[str, Any]:
    best_face: dict[str, tuple[float, str]] = {}
    for row in read_csv(face_predictions):
        prob = float(row.get("fake_prob") or 0.0)
        current = best_face.get(row["image_id"])
        if current is None or prob > current[0]:
            best_face[row["image_id"]] = (prob, str(row.get("face_id", "0")))
    wanted = {(image_id, face_id) for image_id, (_prob, face_id) in best_face.items()}
    out: dict[str, Any] = {}
    for path in sorted(raw_face_dir.glob("shard_*/faces.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (str(row["image_id"]), str(row.get("face_id", "0")))
                if key in wanted:
                    out[key[0]] = row.get("landmarks")
    return out


def write_zip(json_dir: Path, zip_path: Path) -> str:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(json_dir.glob("*.json")):
            archive.write(path, Path("json") / path.name)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build base DDL-X JSON files from model labels and WBF boxes.")
    parser.add_argument("--test-images", required=True, type=Path)
    parser.add_argument("--image-scores", required=True, type=Path)
    parser.add_argument("--detector-boxes", required=True, type=Path)
    parser.add_argument("--face-predictions", required=True, type=Path)
    parser.add_argument("--raw-face-dir", required=True, type=Path)
    parser.add_argument("--output-json-dir", required=True, type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--zip-path", type=Path)
    args = parser.parse_args()

    images = {row["image_id"]: row for row in read_csv(args.test_images)}
    scores = {row["image_id"]: row for row in read_csv(args.image_scores)}
    detector = json.loads(args.detector_boxes.read_text(encoding="utf-8")) if args.detector_boxes.exists() else {}
    landmarks = load_best_landmarks(args.raw_face_dir, args.face_predictions)
    args.output_json_dir.mkdir(parents=True, exist_ok=True)

    counts = {"total": 0, "fake": 0, "real": 0, "wbf_box_images": 0, "fallback_box_images": 0, "no_box_fake": 0}
    for image_id in sorted(images):
        image = images[image_id]
        score = scores.get(image_id)
        if score is None:
            raise RuntimeError(f"Missing image score for {image_id}")
        width = int(float(image["width"]))
        height = int(float(image["height"]))
        label = score["classification"]
        boxes_1000: list[list[int]] = []
        if label == "fake":
            for box in detector.get(image_id, {}).get("Bounding boxes", []):
                converted = to1000(box, width, height)
                if converted:
                    boxes_1000.append(converted)
            if boxes_1000:
                counts["wbf_box_images"] += 1
            else:
                fallback = [to1000(box, width, height) for box in landmark_boxes(landmarks.get(image_id), width, height)]
                boxes_1000 = [box for box in fallback if box]
                if boxes_1000:
                    counts["fallback_box_images"] += 1
                else:
                    counts["no_box_fake"] += 1
        payload = {
            "Bounding boxes": boxes_1000 if label == "fake" and boxes_1000 else None,
            "Visible forgery traces": (
                "This image has been tampered with." if label == "fake" else "This image has not been tampered with."
            ),
            "Classification result": label,
        }
        (args.output_json_dir / f"{image_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        counts["total"] += 1
        counts[label] += 1

    summary: dict[str, object] = {**counts, "output_json_dir": str(args.output_json_dir)}
    if args.zip_path:
        summary["zip_path"] = str(args.zip_path)
        summary["zip_sha256"] = write_zip(args.output_json_dir, args.zip_path)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

