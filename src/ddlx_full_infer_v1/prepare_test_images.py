from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FIELDS = ["image_id", "image_path", "width", "height"]


def iter_images(image_dir: Path) -> Iterable[Path]:
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def shard_for(image_id: str, num_shards: int) -> int:
    digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, num_shards)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a DDL-X test image directory and create inference manifests.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    image_dir = args.image_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not image_dir.is_dir():
        raise SystemExit(f"Missing image directory: {image_dir}")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")

    from PIL import Image

    rows: list[dict[str, object]] = []
    for path in iter_images(image_dir):
        with Image.open(path) as image:
            width, height = image.size
        rows.append(
            {
                "image_id": path.stem,
                "image_path": str(path),
                "width": int(width),
                "height": int(height),
            }
        )

    if not rows:
        raise SystemExit(f"No images found under {image_dir}")
    if len({str(row["image_id"]) for row in rows}) != len(rows):
        raise SystemExit("Duplicate image_id values found. Use unique file stems for DDL-X test images.")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "test_images.csv", rows)
    shard_rows: dict[int, list[dict[str, object]]] = {idx: [] for idx in range(args.num_shards)}
    for row in rows:
        shard_rows[shard_for(str(row["image_id"]), args.num_shards)].append(row)
    for idx, shard in shard_rows.items():
        write_csv(out_dir / "shards" / f"shard_{idx:02d}_images.csv", shard)

    summary = {
        "image_dir": str(image_dir),
        "out_dir": str(out_dir),
        "images": len(rows),
        "num_shards": args.num_shards,
        "shard_counts": {f"{idx:02d}": len(shard_rows[idx]) for idx in range(args.num_shards)},
    }
    (out_dir / "prepare_test_images_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
