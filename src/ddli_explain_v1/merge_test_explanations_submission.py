from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path


def predictions_for_shard(prediction_dir: Path, shard: int) -> list[Path]:
    if shard == 1:
        return [
            prediction_dir / "shard_01_predictions.jsonl",
            prediction_dir / "shard_01_resume_after12000_predictions.jsonl",
        ]
    return [prediction_dir / f"shard_{shard:02d}_predictions.jsonl"]


def read_jsonl(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json-dir", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--output-json-dir", required=True, type=Path)
    parser.add_argument("--zip-path", required=True, type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--num-shards", default=6, type=int)
    args = parser.parse_args()

    args.output_json_dir.mkdir(parents=True, exist_ok=False)
    counts: Counter[str] = Counter()
    missing_statement: Counter[str] = Counter()
    empty_responses = 0
    coordinate_leaks = 0
    chars: list[int] = []
    total = 0

    for shard in range(args.num_shards):
        manifest = read_jsonl([args.manifest_dir / f"shard_{shard:02d}_manifest.jsonl"])
        predictions = read_jsonl(predictions_for_shard(args.prediction_dir, shard))
        if len(manifest) != len(predictions):
            raise RuntimeError(f"shard {shard:02d} rows differ: manifest={len(manifest)} prediction={len(predictions)}")
        for info, pred in zip(manifest, predictions):
            image_id = str(info["image_id"])
            label = str(info["label"])
            source_path = args.source_json_dir / f"{image_id}.json"
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            if payload["Classification result"] != label:
                raise RuntimeError(f"label mismatch: {image_id}")
            response = str(pred.get("response", ""))
            if not response.strip():
                empty_responses += 1
            expected = (
                "This image has been tampered with."
                if label == "fake"
                else "This image has not been tampered with."
            )
            if expected not in response:
                missing_statement[label] += 1
            if "[" in response and any(char.isdigit() for char in response[response.find("[") : response.find("[") + 40]):
                coordinate_leaks += 1
            payload["Visible forgery traces"] = response
            (args.output_json_dir / f"{image_id}.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            counts[label] += 1
            chars.append(len(response))
            total += 1

    if total != 100000:
        raise RuntimeError(f"expected 100000 output records, got {total}")
    if empty_responses:
        raise RuntimeError(f"found {empty_responses} empty model responses")

    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output_json_dir.glob("*.json")):
            archive.write(path, Path("json") / path.name)

    digest = hashlib.sha256(args.zip_path.read_bytes()).hexdigest()
    summary = {
        "total": total,
        "labels": dict(counts),
        "empty_responses": empty_responses,
        "missing_expected_final_statement": dict(missing_statement),
        "coordinate_like_responses": coordinate_leaks,
        "response_chars_min": min(chars),
        "response_chars_mean": sum(chars) / len(chars),
        "response_chars_max": max(chars),
        "zip_path": str(args.zip_path),
        "zip_sha256": digest,
    }
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
