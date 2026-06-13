from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_json_dir(json_dir: Path, zip_path: Path) -> str:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(json_dir.glob("*.json")):
            archive.write(path, Path("json") / path.name)
    return sha256(zip_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--num-shards", type=int, default=6)
    args = parser.parse_args()

    replacements: dict[str, dict[str, str]] = {variant: {} for variant in args.variants}
    empty = 0
    missing_statement: Counter[str] = Counter()
    coord_like = 0
    chars: list[int] = []
    for shard in range(args.num_shards):
        manifest = load_jsonl(args.input_dir / f"shard_{shard:02d}_manifest.jsonl")
        predictions = load_jsonl(args.prediction_dir / f"shard_{shard:02d}_predictions.jsonl")
        if len(manifest) != len(predictions):
            raise RuntimeError(f"shard {shard:02d} mismatch: manifest={len(manifest)} prediction={len(predictions)}")
        for info, pred in zip(manifest, predictions):
            variant = str(info["variant"])
            image_id = str(info["image_id"])
            response = str(pred.get("response", ""))
            if not response.strip():
                empty += 1
            if "This image has been tampered with." not in response:
                missing_statement[variant] += 1
                response = response.rstrip() + "\n\nSummary: This image has been tampered with."
            if "[" in response and any(char.isdigit() for char in response[response.find("[") : response.find("[") + 48]):
                coord_like += 1
            replacements[variant][image_id] = response
            chars.append(len(response))

    if empty:
        raise RuntimeError(f"empty responses: {empty}")

    args.out_root.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for variant in args.variants:
        src_json_dir = args.source_root / variant / "json"
        out_variant = args.out_root / variant
        out_json_dir = out_variant / "json"
        out_variant.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_json_dir, out_json_dir)

        counts = {
            "variant": variant,
            "images": 0,
            "fake": 0,
            "real": 0,
            "text_replacements": 0,
            "empty_responses": 0,
            "missing_expected_final_statement_in_replacements": int(missing_statement.get(variant, 0)),
            "total_boxes": 0,
        }
        for image_id, response in replacements[variant].items():
            path = out_json_dir / f"{image_id}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("Classification result") != "fake":
                raise RuntimeError(f"{variant}/{image_id}: replacement target is not fake")
            payload["Visible forgery traces"] = response
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            counts["text_replacements"] += 1

        for path in out_json_dir.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            label = str(payload["Classification result"])
            counts["images"] += 1
            counts[label] += 1
            text = str(payload.get("Visible forgery traces", ""))
            if not text.strip():
                counts["empty_responses"] += 1
            boxes = payload.get("Bounding boxes")
            if boxes:
                counts["total_boxes"] += len(boxes)

        if counts["images"] != 100000:
            raise RuntimeError(f"{variant}: expected 100000 images, got {counts['images']}")
        if counts["empty_responses"]:
            raise RuntimeError(f"{variant}: empty responses remain")
        zip_path = out_variant / f"submission_ddl_x_test_{args.tag}_{variant}_fake_nobox_textrerun.zip"
        digest = zip_json_dir(out_json_dir, zip_path)
        counts["zip_path"] = str(zip_path)
        counts["zip_bytes"] = zip_path.stat().st_size
        counts["sha256"] = digest
        (out_variant / "summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
        summaries[variant] = counts

    global_summary = {
        "source_root": str(args.source_root),
        "out_root": str(args.out_root),
        "records_replaced_total": sum(len(v) for v in replacements.values()),
        "records_replaced_by_variant": {variant: len(values) for variant, values in replacements.items()},
        "coordinate_like_replacement_responses": coord_like,
        "replacement_chars_min": min(chars) if chars else 0,
        "replacement_chars_mean": (sum(chars) / len(chars)) if chars else 0,
        "replacement_chars_max": max(chars) if chars else 0,
        "variants": summaries,
    }
    (args.out_root / "fake_nobox_textrerun_summary.json").write_text(json.dumps(global_summary, indent=2), encoding="utf-8")
    print(json.dumps(global_summary, indent=2))


if __name__ == "__main__":
    main()
