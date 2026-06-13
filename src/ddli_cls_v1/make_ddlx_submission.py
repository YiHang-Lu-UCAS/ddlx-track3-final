#!/usr/bin/env python3
"""Convert TruFor .npz outputs into DDL-X Track 3 submission JSON files.

Expected TruFor output keys include:
  - map: anomaly/localization heatmap
  - score: image-level forgery score
  - imgsize: optional original size metadata

The script is deliberately conservative: it never saves mask images and writes
only one JSON per input image.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def list_npz(npz_dir: Path) -> list[Path]:
    return sorted(p for p in npz_dir.rglob("*.npz") if p.is_file())


def build_npz_index(npz_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in list_npz(npz_dir):
        index[path.stem] = path
        nested_stem = Path(path.stem).stem
        if nested_stem:
            index[nested_stem] = path
    return index


def as_scalar(value: np.ndarray | float | int) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[0])


def normalize_map(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim == 3:
        heatmap = np.squeeze(heatmap)
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2-D heatmap, got shape {heatmap.shape}")
    finite = np.isfinite(heatmap)
    if not finite.any():
        return np.zeros_like(heatmap, dtype=np.float32)
    heatmap = np.where(finite, heatmap, 0.0)
    lo = float(heatmap.min())
    hi = float(heatmap.max())
    if lo < 0.0 or hi > 1.0:
        if math.isclose(hi, lo):
            return np.zeros_like(heatmap, dtype=np.float32)
        heatmap = (heatmap - lo) / (hi - lo)
    return np.clip(heatmap, 0.0, 1.0)


def largest_component_bbox(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    binary = np.asarray(binary).astype(bool)
    if not binary.any():
        return None

    try:
        import cv2  # type: ignore

        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
        if num <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = int(np.argmax(areas)) + 1
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        return x, y, x + w - 1, y + h - 1
    except Exception:
        pass

    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    best: tuple[int, int, int, int, int] | None = None
    for sy, sx in np.argwhere(binary):
        if seen[sy, sx]:
            continue
        q: deque[tuple[int, int]] = deque([(int(sy), int(sx))])
        seen[sy, sx] = True
        count = 0
        x1 = x2 = int(sx)
        y1 = y2 = int(sy)
        while q:
            y, x = q.popleft()
            count += 1
            x1, x2 = min(x1, x), max(x2, x)
            y1, y2 = min(y1, y), max(y2, y)
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        if best is None or count > best[0]:
            best = (count, x1, y1, x2, y2)
    if best is None:
        return None
    _, x1, y1, x2, y2 = best
    return x1, y1, x2, y2


def bbox_from_heatmap(
    heatmap: np.ndarray,
    threshold: float,
    min_area_ratio: float,
    fallback_top_percent: float,
) -> tuple[int, int, int, int] | None:
    h, w = heatmap.shape
    min_area = max(1, int(h * w * min_area_ratio))
    binary = heatmap >= threshold
    bbox = largest_component_bbox(binary)
    if bbox is not None and component_area(binary, bbox) >= min_area:
        return bbox

    cutoff = np.percentile(heatmap, max(0.0, min(100.0, 100.0 - fallback_top_percent)))
    binary = heatmap >= cutoff
    bbox = largest_component_bbox(binary)
    if bbox is not None and component_area(binary, bbox) >= min_area:
        return bbox
    return None


def component_area(binary: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return int(binary[y1 : y2 + 1, x1 : x2 + 1].sum())


def scale_bbox_to_1000(bbox: tuple[int, int, int, int], map_shape: tuple[int, int]) -> list[int]:
    h, w = map_shape
    x1, y1, x2, y2 = bbox
    scaled = [
        round(x1 / max(1, w - 1) * 999) + 1,
        round(y1 / max(1, h - 1) * 999) + 1,
        round(x2 / max(1, w - 1) * 999) + 1,
        round(y2 / max(1, h - 1) * 999) + 1,
    ]
    return [int(max(1, min(1000, v))) for v in scaled]


def fake_text(bbox: list[int]) -> str:
    return (
        f"The image is fake. Visible forgery traces are located around {bbox}, "
        "including inconsistent facial texture, blending artifacts, and abnormal boundary transitions."
    )


def real_text() -> str:
    return "The image appears real with no visible forgery traces."


def write_json(path: Path, label: str, bbox: list[int] | None) -> None:
    payload = {
        "Bounding boxes": bbox,
        "Visible forgery traces": real_text() if label == "real" else fake_text(bbox or [1, 1, 1000, 1000]),
        "Classification result": label,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def zip_json_dir(json_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(json_dir.rglob("*.json")):
            zf.write(path, Path("json") / path.name)


def convert(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir)
    npz_dir = Path(args.npz_dir)
    json_dir = Path(args.json_dir)
    images = list_images(image_dir)
    npz_index = build_npz_index(npz_dir)
    missing: list[str] = []
    counts = {"real": 0, "fake": 0}

    for image_path in images:
        npz_path = npz_index.get(image_path.stem)
        if npz_path is None:
            missing.append(image_path.name)
            if args.skip_missing:
                continue
            raise FileNotFoundError(f"Missing npz for {image_path.name}")

        data = np.load(npz_path)
        if "score" not in data or "map" not in data:
            raise KeyError(f"{npz_path} must contain 'score' and 'map'. Keys: {data.files}")
        score = as_scalar(data["score"])
        label = "fake" if score >= args.cls_threshold else "real"
        bbox_1000: list[int] | None = None
        if label == "fake":
            heatmap = normalize_map(data["map"])
            bbox = bbox_from_heatmap(
                heatmap,
                threshold=args.map_threshold,
                min_area_ratio=args.min_area_ratio,
                fallback_top_percent=args.fallback_top_percent,
            )
            bbox_1000 = [1, 1, 1000, 1000] if bbox is None else scale_bbox_to_1000(bbox, heatmap.shape)
        counts[label] += 1
        write_json(json_dir / f"{image_path.stem}.json", label, bbox_1000)

    if args.zip_path:
        zip_json_dir(json_dir, Path(args.zip_path))

    print(f"images={len(images)} real={counts['real']} fake={counts['fake']} json_dir={json_dir}")
    if missing:
        print(f"missing_npz={len(missing)} first_missing={missing[:5]}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="DDL_X image directory.")
    parser.add_argument("--npz-dir", required=True, help="Directory containing TruFor .npz outputs.")
    parser.add_argument("--json-dir", required=True, help="Output json directory.")
    parser.add_argument("--zip-path", default="", help="Optional output submission zip path.")
    parser.add_argument("--cls-threshold", type=float, default=0.5)
    parser.add_argument("--map-threshold", type=float, default=0.5)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--fallback-top-percent", type=float, default=5.0)
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    convert(parse_args())
