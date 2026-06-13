from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path


def to1000(box: list[float], width: int, height: int) -> list[int] | None:
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    out = [
        round(x1 / width * 999) + 1,
        round(y1 / height * 999) + 1,
        round(x2 / width * 999) + 1,
        round(y2 / height * 999) + 1,
    ]
    out = [int(max(1, min(1000, v))) for v in out]
    if out[2] <= out[0]:
        out[2] = min(1000, out[0] + 1)
    if out[3] <= out[1]:
        out[3] = min(1000, out[1] + 1)
    return out if out[2] > out[0] and out[3] > out[1] else None


def parse_box(raw: str) -> list[int] | None:
    try:
        box = json.loads(raw)
    except Exception:
        return None
    if not isinstance(box, list) or len(box) != 4:
        return None
    out = [int(round(float(v))) for v in box]
    return out if out[2] > out[0] and out[3] > out[1] else None


def mk_box(cx: float, cy: float, width: float, height: float, iw: int, ih: int) -> list[int] | None:
    x1 = max(0.0, cx - width / 2)
    y1 = max(0.0, cy - height / 2)
    x2 = min(float(iw), cx + width / 2)
    y2 = min(float(ih), cy + height / 2)
    out = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    return out if out[2] > out[0] and out[3] > out[1] else None


def landmark_boxes(landmarks: object, iw: int, ih: int, mode: str) -> list[list[int]]:
    if not isinstance(landmarks, list) or len(landmarks) < 5:
        return []
    pts = [[float(p[0]), float(p[1])] for p in landmarks[:5]]
    d = max(2.0, math.dist(pts[0], pts[1]))
    mouth = ((pts[3][0] + pts[4][0]) / 2.0, (pts[3][1] + pts[4][1]) / 2.0)
    specs = []
    if "eyes" in mode:
        specs.extend(
            [
                (pts[0][0], pts[0][1] - 0.06 * d, 1.02 * d, 0.68 * d),
                (pts[1][0], pts[1][1] - 0.06 * d, 1.02 * d, 0.68 * d),
            ]
        )
    if "nose" in mode:
        specs.append((pts[2][0], pts[2][1] + 0.03 * d, 1.02 * d, 0.88 * d))
    if "mouth" in mode:
        specs.append((mouth[0], mouth[1], 1.16 * d, 0.74 * d))

    boxes = []
    seen = set()
    for cx, cy, bw, bh in specs:
        box = mk_box(cx, cy, bw, bh, iw, ih)
        if box and tuple(box) not in seen:
            seen.add(tuple(box))
            boxes.append(box)
    cap = 4 if mode == "nose_eyes_mouth" else 3
    return boxes[:cap]


def load_text_zip(path: Path) -> dict[str, dict]:
    payloads = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            payloads[Path(name).stem] = json.loads(zf.read(name).decode("utf-8"))
    return payloads


def write_zip(json_dir: Path, zip_path: Path) -> str:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(json_dir.glob("*.json")):
            zf.write(path, Path("json") / path.name)
    digest = hashlib.sha256()
    with zip_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal-text-zip", required=True, type=Path)
    ap.add_argument("--eyes-text-zip", required=True, type=Path)
    ap.add_argument("--eyes-mouth-text-zip", required=True, type=Path)
    ap.add_argument("--nose-eyes-mouth-text-zip", required=True, type=Path)
    ap.add_argument("--image-scores", required=True, type=Path)
    ap.add_argument("--test-images", required=True, type=Path)
    ap.add_argument("--detector-boxes", required=True, type=Path)
    ap.add_argument("--cls-preds-dir", required=True, type=Path)
    ap.add_argument("--raw-face-dir", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    text_zips = {
        "normal": args.normal_text_zip,
        "fake_nobox_eyes": args.eyes_text_zip,
        "fake_nobox_eyes_mouth": args.eyes_mouth_text_zip,
        "fake_nobox_nose_eyes_mouth": args.nose_eyes_mouth_text_zip,
    }
    texts = {name: load_text_zip(path) for name, path in text_zips.items()}
    pred = json.loads(args.detector_boxes.read_text(encoding="utf-8"))

    dims: dict[str, tuple[int, int]] = {}
    classifications: dict[str, str] = {}
    with args.image_scores.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dims[row["image_id"]] = (int(float(row["width"])), int(float(row["height"])))
            classifications[row["image_id"]] = row["classification"]

    expected = []
    with args.test_images.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            expected.append(row["image_id"])

    best_face: dict[str, tuple[float, str, list[int]]] = {}
    for path in sorted(args.cls_preds_dir.glob("shard_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                box = parse_box(row.get("crop_bbox", ""))
                if box is None:
                    continue
                prob = float(row.get("fake_prob") or 0.0)
                current = best_face.get(row["image_id"])
                if current is None or prob > current[0]:
                    best_face[row["image_id"]] = (prob, str(row.get("face_id", "0")), box)

    needed = {(iid, face_id) for iid, (_, face_id, _) in best_face.items()}
    landmarks: dict[tuple[str, str], object] = {}
    for path in sorted(args.raw_face_dir.glob("shard_*/faces.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["image_id"], str(row.get("face_id", 0)))
                if key in needed:
                    landmarks[key] = row.get("landmarks")

    variants = {
        "normal": "none",
        "fake_nobox_eyes": "eyes",
        "fake_nobox_eyes_mouth": "eyes_mouth",
        "fake_nobox_nose_eyes_mouth": "nose_eyes_mouth",
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    summaries = {}

    for name, fallback in variants.items():
        text = texts[name]
        json_dir = args.out_root / name / "json"
        if json_dir.exists():
            shutil.rmtree(json_dir)
        json_dir.mkdir(parents=True)
        counts = {
            "variant": name,
            "text_zip": str(text_zips[name]),
            "images": 0,
            "fake": 0,
            "real": 0,
            "fake_with_detector_boxes": 0,
            "fake_without_detector_boxes": 0,
            "fallback_added_images": 0,
            "fallback_added_boxes": 0,
            "missing_fallback": 0,
            "text_changed": 0,
            "total_boxes": 0,
        }
        for iid in expected:
            payload = dict(text[iid])
            label = payload["Classification result"]
            if label != classifications.get(iid):
                raise RuntimeError(f"{name}: classification mismatch for {iid}: zip={label} scores={classifications.get(iid)}")
            boxes_1000: list[list[int]] = []
            if label == "fake":
                width, height = dims[iid]
                seen = set()
                for box in pred.get(iid, {}).get("Bounding boxes", []):
                    converted = to1000(box, width, height)
                    if converted and tuple(converted) not in seen:
                        seen.add(tuple(converted))
                        boxes_1000.append(converted)
                if boxes_1000:
                    counts["fake_with_detector_boxes"] += 1
                else:
                    counts["fake_without_detector_boxes"] += 1
                    if fallback != "none" and iid in best_face:
                        _, face_id, _ = best_face[iid]
                        raw_boxes = landmark_boxes(landmarks.get((iid, face_id)), width, height, fallback)
                        boxes_1000 = [box for box in (to1000(b, width, height) for b in raw_boxes) if box]
                    if boxes_1000:
                        counts["fallback_added_images"] += 1
                        counts["fallback_added_boxes"] += len(boxes_1000)
                    elif fallback != "none":
                        counts["missing_fallback"] += 1
                payload["Bounding boxes"] = boxes_1000 or None
            else:
                payload["Bounding boxes"] = None

            if payload["Visible forgery traces"] != text[iid]["Visible forgery traces"]:
                counts["text_changed"] += 1
            if payload["Bounding boxes"]:
                counts["total_boxes"] += len(payload["Bounding boxes"])
            counts["images"] += 1
            counts[label] += 1
            (json_dir / f"{iid}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        if counts["images"] != 100000:
            raise RuntimeError(f"{name}: expected 100000 images, got {counts['images']}")
        if counts["text_changed"]:
            raise RuntimeError(f"{name}: text changed in {counts['text_changed']} files")
        zip_path = args.out_root / name / f"submission_ddl_x_test_{args.tag}_{name}.zip"
        sha = write_zip(json_dir, zip_path)
        counts["zip_path"] = str(zip_path)
        counts["zip_bytes"] = zip_path.stat().st_size
        counts["sha256"] = sha
        (args.out_root / name / "summary.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
        summaries[name] = counts

    (args.out_root / "four_variant_summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
