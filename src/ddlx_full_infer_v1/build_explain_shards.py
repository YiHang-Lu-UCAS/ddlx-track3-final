from __future__ import annotations

import argparse
import csv
import json
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
    parser = argparse.ArgumentParser(description="Build Qwen-VL inference JSONL shards from base submission JSON.")
    parser.add_argument("--submission-json-dir", required=True, type=Path)
    parser.add_argument("--test-images", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_paths: dict[str, str] = {}
    with args.test_images.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_paths[row["image_id"]] = row["image_path"]

    handles = [(args.out_dir / f"shard_{idx:02d}.jsonl").open("w", encoding="utf-8") for idx in range(args.num_shards)]
    manifest_handles = [
        (args.out_dir / f"shard_{idx:02d}_manifest.jsonl").open("w", encoding="utf-8")
        for idx in range(args.num_shards)
    ]
    total = 0
    try:
        for index, json_path in enumerate(sorted(args.submission_json_dir.glob("*.json"))):
            image_id = json_path.stem
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            label = str(payload["Classification result"]).lower()
            boxes = payload.get("Bounding boxes")
            record = {
                "messages": [
                    {"role": "user", "content": prompt_for(label, boxes)},
                    {"role": "assistant", "content": str(payload["Visible forgery traces"])},
                ],
                "images": [image_paths[image_id]],
            }
            shard = index % args.num_shards
            handles[shard].write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest_handles[shard].write(
                json.dumps({"image_id": image_id, "label": label, "boxes": boxes}, ensure_ascii=False) + "\n"
            )
            total += 1
    finally:
        for handle in handles + manifest_handles:
            handle.close()
    summary = {"total": total, "num_shards": args.num_shards, "out_dir": str(args.out_dir)}
    (args.out_dir / "input_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

