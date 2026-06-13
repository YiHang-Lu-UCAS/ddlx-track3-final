from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_unique_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["image_id"] for row in csv.DictReader(handle)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-faces", required=True, type=Path)
    ap.add_argument("--json-dir", required=True, type=Path)
    ap.add_argument("--tune-csv", required=True, type=Path)
    ap.add_argument("--calib-csv", required=True, type=Path)
    ap.add_argument("--holdout-csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--num-shards", type=int, default=8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "indices").mkdir()
    (args.out_dir / "detector_face_shards").mkdir()

    tune = load_unique_ids(args.tune_csv)
    calib = load_unique_ids(args.calib_csv)
    holdout = load_unique_ids(args.holdout_csv)
    overlap = {
        "tune_calib": len(tune & calib),
        "tune_holdout": len(tune & holdout),
        "calib_holdout": len(calib & holdout),
    }

    rows: list[dict[str, str]] = []
    with args.dev_faces.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_image: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_image.setdefault(row["image_id"], []).append(row)
    usable_ids = set(by_image)
    train = usable_ids - tune - calib - holdout
    splits = {"train": train, "tune": tune & usable_ids, "calib": calib & usable_ids, "holdout": holdout & usable_ids}

    json_paths = {p.stem: p for p in args.json_dir.glob("*.json")}
    missing_json: list[str] = []
    json_not_in_faces: list[str] = []
    labels: dict[str, Counter[str]] = {name: Counter() for name in splits}
    text_lengths: dict[str, list[int]] = {name: [] for name in splits}
    valid_ids: dict[str, list[str]] = {name: [] for name in splits}

    for split, ids in splits.items():
        for image_id in sorted(ids):
            stem = Path(image_id).stem
            path = json_paths.get(stem)
            if path is None:
                missing_json.append(image_id)
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            labels[split][str(data.get("Classification result", "")).lower()] += 1
            text_lengths[split].append(len(str(data.get("Visible forgery traces", ""))))
            valid_ids[split].append(image_id)
    usable_stems = {Path(x).stem for x in usable_ids}
    json_not_in_faces = sorted(f"{stem}.json" for stem in set(json_paths) - usable_stems)

    for split, ids in valid_ids.items():
        (args.out_dir / "indices" / f"{split}_image_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    all_rows = [row for image_id in sorted(usable_ids) for row in by_image[image_id]]
    headers = list(all_rows[0]) if all_rows else []
    shard_counts: list[int] = []
    for index in range(args.num_shards):
        shard_rows = all_rows[index:: args.num_shards]
        shard_counts.append(len(shard_rows))
        path = args.out_dir / "detector_face_shards" / f"shard_{index:02d}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(shard_rows)

    def text_stats(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {}
        values = sorted(values)
        return {
            "min": values[0],
            "median": values[len(values) // 2],
            "p90": values[int(len(values) * 0.9)],
            "max": values[-1],
            "mean": round(sum(values) / len(values), 2),
        }

    report = {
        "dev_faces_rows": len(rows),
        "dev_faces_unique_images": len(usable_ids),
        "json_files": len(json_paths),
        "json_not_in_dev_faces_count": len(json_not_in_faces),
        "json_not_in_dev_faces": json_not_in_faces[:20],
        "missing_json_count": len(missing_json),
        "missing_json": missing_json[:20],
        "reserved_split_overlap": overlap,
        "split_images": {name: len(ids) for name, ids in valid_ids.items()},
        "split_labels": {name: dict(count) for name, count in labels.items()},
        "text_length_chars": {name: text_stats(values) for name, values in text_lengths.items()},
        "detector_shard_face_rows": shard_counts,
    }
    (args.out_dir / "data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
