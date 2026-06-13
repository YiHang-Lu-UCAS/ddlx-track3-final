from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def prompt_for(label: str, boxes: list[list[int]] | None) -> str:
    if label == "fake":
        region_text = json.dumps(boxes, ensure_ascii=True) if boxes else "no reliable detector box was retained"
        return (
            "You are writing the Visible forgery traces field for a forensic image submission. "
            "The upstream classifier labels this image as fake. "
            f"Candidate manipulated bounding boxes from the detector are: {region_text}. "
            "Describe only visually supportable manipulation traces in English, focusing on the candidate regions "
            "when present. Include a concise final statement that the image has been tampered with. "
            "Do not output JSON or bounding box coordinates."
        )
    return (
        "You are writing the Visible forgery traces field for a forensic image submission. "
        "The upstream classifier labels this image as real and no manipulation box is retained. "
        "Describe visually supportable evidence of consistency in English and end with a concise statement that "
        "the image has not been tampered with. Do not output JSON or bounding box coordinates."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-json-dir", required=True, type=Path)
    parser.add_argument("--test-images", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=6)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=False)
    image_paths: dict[str, str] = {}
    with args.test_images.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_paths[row["image_id"]] = row["image_path"]

    handles = [(args.out_dir / f"shard_{idx:02d}.jsonl").open("w", encoding="utf-8") for idx in range(args.num_shards)]
    manifest_handles = [
        (args.out_dir / f"shard_{idx:02d}_manifest.jsonl").open("w", encoding="utf-8")
        for idx in range(args.num_shards)
    ]
    counts: Counter[str] = Counter()
    shard_counts: Counter[int] = Counter()
    missing_images: list[str] = []
    try:
        for index, json_path in enumerate(sorted(args.submission_json_dir.glob("*.json"))):
            image_id = json_path.stem
            image_path = image_paths.get(image_id)
            if image_path is None:
                missing_images.append(image_id)
                continue
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            label = str(payload["Classification result"]).lower()
            boxes = payload.get("Bounding boxes")
            record = {
                "messages": [
                    {"role": "user", "content": prompt_for(label, boxes)},
                    {"role": "assistant", "content": str(payload["Visible forgery traces"])},
                ],
                "images": [image_path],
            }
            shard = index % args.num_shards
            handles[shard].write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest_handles[shard].write(
                json.dumps(
                    {"image_id": image_id, "json_path": str(json_path), "label": label, "boxes": boxes},
                    ensure_ascii=False,
                )
                + "\n"
            )
            counts[label] += 1
            shard_counts[shard] += 1
    finally:
        for handle in handles + manifest_handles:
            handle.close()

    if missing_images:
        raise RuntimeError(f"missing image paths for {len(missing_images)} IDs, e.g. {missing_images[:3]}")
    if sum(counts.values()) != 100000:
        raise RuntimeError(f"expected 100000 JSON files, got {sum(counts.values())}")
    summary = {
        "total": sum(counts.values()),
        "labels": dict(counts),
        "num_shards": args.num_shards,
        "shard_counts": {f"{idx:02d}": shard_counts[idx] for idx in range(args.num_shards)},
        "submission_json_dir": str(args.submission_json_dir),
        "test_images": str(args.test_images),
    }
    (args.out_dir / "input_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
