from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from ultralytics.data.utils import get_hash, img2label_paths, save_dataset_cache_file


def relative_train_path(path: str) -> str:
    marker = "/images/train/"
    if marker not in path:
        raise ValueError(f"Unexpected training image path: {path}")
    return path.split(marker, 1)[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-cache", required=True, type=Path)
    ap.add_argument("--train-index", required=True, type=Path)
    ap.add_argument("--target-cache", required=True, type=Path)
    ap.add_argument("--backup-existing", required=True, type=Path)
    ap.add_argument("--summary-out", required=True, type=Path)
    args = ap.parse_args()

    new_files = args.train_index.read_text(encoding="utf-8").splitlines()
    source = np.load(str(args.source_cache), allow_pickle=True).item()
    source_labels = source["labels"]

    if args.target_cache.exists():
        args.backup_existing.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.target_cache, args.backup_existing)

    selected = []
    skipped = []
    new_index = 0
    for label in source_labels:
        old_relative = relative_train_path(label["im_file"])
        if new_index < len(new_files) and old_relative == relative_train_path(new_files[new_index]):
            label["im_file"] = new_files[new_index]
            selected.append(label)
            new_index += 1
        else:
            skipped.append(old_relative)

    if new_index != len(new_files):
        missing = relative_train_path(new_files[new_index])
        raise RuntimeError(f"Cache order does not resolve the new index at row {new_index}: {missing}")

    label_files = img2label_paths(new_files)
    backgrounds = sum(1 for label in selected if len(label["cls"]) == 0)
    converted = {
        "labels": selected,
        "hash": get_hash(label_files + new_files),
        "results": (len(selected), 0, backgrounds, 0, len(selected)),
        "msgs": [],
    }
    args.target_cache.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_cache_file("", args.target_cache, converted, source["version"])

    summary = {
        "source_cache": str(args.source_cache),
        "source_records": len(source_labels),
        "target_cache": str(args.target_cache),
        "target_records": len(selected),
        "skipped_records": len(skipped),
        "skipped_relative_paths": skipped,
        "background_records": backgrounds,
        "train_index": str(args.train_index),
        "backup_existing": str(args.backup_existing),
        "method": "reused verified labels in source order; replaced image paths; recomputed Ultralytics path-size hash",
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
