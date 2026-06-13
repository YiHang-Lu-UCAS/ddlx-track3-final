from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_for(label: str, boxes: list[list[int]]) -> str:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-faces", required=True, type=Path)
    ap.add_argument("--json-dir", required=True, type=Path)
    ap.add_argument("--indices-dir", required=True, type=Path)
    ap.add_argument("--detector-boxes", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    image_meta: dict[str, tuple[str, str]] = {}
    with args.dev_faces.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_meta.setdefault(row["image_id"], (row["image_path"], row["json_path"]))
    detector = json.loads(args.detector_boxes.read_text(encoding="utf-8"))

    reports: dict[str, dict[str, object]] = {}
    records_by_split: dict[str, list[dict[str, object]]] = {}
    for split in ["train", "tune", "calib", "holdout"]:
        records: list[dict[str, object]] = []
        counts: Counter[str] = Counter()
        fake_with_box = 0
        fake_without_box = 0
        suppressed_real_detector_boxes = 0
        lengths: list[int] = []
        for image_id in read_ids(args.indices_dir / f"{split}_image_ids.txt"):
            image_path, json_path = image_meta[image_id]
            annotation = json.loads(Path(json_path).read_text(encoding="utf-8"))
            label = str(annotation["Classification result"]).lower()
            boxes = detector.get(image_id, {}).get("Bounding boxes", []) if label == "fake" else []
            if label == "fake":
                fake_with_box += bool(boxes)
                fake_without_box += not bool(boxes)
            elif image_id in detector:
                suppressed_real_detector_boxes += 1
            target = str(annotation["Visible forgery traces"])
            records.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt_for(label, boxes)},
                        {"role": "assistant", "content": target},
                    ],
                    "images": [image_path],
                }
            )
            counts[label] += 1
            lengths.append(len(target))
        records_by_split[split] = records
        out = args.out_dir / f"{split}_swift.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        reports[split] = {
            "rows": len(records),
            "labels": dict(counts),
            "fake_with_detector_box": fake_with_box,
            "fake_without_detector_box": fake_without_box,
            "suppressed_real_detector_boxes": suppressed_real_detector_boxes,
            "target_char_max": max(lengths) if lengths else 0,
            "jsonl": str(out),
        }

    def balanced_long_sample(split: str, per_label: int) -> list[dict[str, object]]:
        rows = records_by_split[split]
        grouped: dict[str, list[dict[str, object]]] = {"fake": [], "real": []}
        for row in rows:
            target = str(row["messages"][1]["content"]).lower()
            label = "fake" if "has been tampered with" in target and "not been tampered" not in target else "real"
            grouped[label].append(row)
        selected: list[dict[str, object]] = []
        for label in ["fake", "real"]:
            grouped[label].sort(key=lambda row: len(str(row["messages"][1]["content"])), reverse=True)
            selected.extend(grouped[label][:per_label])
        return selected

    subsets = {
        "smoke_train_swift.jsonl": balanced_long_sample("train", 128),
        "smoke_val_swift.jsonl": balanced_long_sample("tune", 32),
        "tune_eval_1024_swift.jsonl": balanced_long_sample("tune", 512),
    }
    for name, records in subsets.items():
        with (args.out_dir / name).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        reports[name] = {"rows": len(records), "jsonl": str(args.out_dir / name)}

    (args.out_dir / "jsonl_summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
