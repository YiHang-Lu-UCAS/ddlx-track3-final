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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-json-dir", required=True, type=Path)
    parser.add_argument("--rerun-input-dir", required=True, type=Path)
    parser.add_argument("--rerun-prediction-dir", required=True, type=Path)
    parser.add_argument("--output-json-dir", required=True, type=Path)
    parser.add_argument("--zip-path", required=True, type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--num-shards", default=6, type=int)
    parser.add_argument("--expected-replacements", required=True, type=int)
    args = parser.parse_args()

    args.output_json_dir.parent.mkdir(parents=True, exist_ok=True)
    if args.output_json_dir.exists():
        raise FileExistsError(args.output_json_dir)
    shutil.copytree(args.raw_json_dir, args.output_json_dir)

    replacements = 0
    rerun_remaining: Counter[str] = Counter()
    for shard in range(args.num_shards):
        manifest = load_jsonl(args.rerun_input_dir / f"shard_{shard:02d}_manifest.jsonl")
        predictions = load_jsonl(args.rerun_prediction_dir / f"shard_{shard:02d}_predictions.jsonl")
        if len(manifest) != len(predictions):
            raise RuntimeError(f"rerun shard {shard:02d} mismatch: {len(manifest)} != {len(predictions)}")
        for info, prediction in zip(manifest, predictions):
            image_id = str(info["image_id"])
            label = str(info["label"])
            response = str(prediction.get("response", ""))
            path = args.output_json_dir / f"{image_id}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["Visible forgery traces"] = response
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            expected = (
                "This image has been tampered with."
                if label == "fake"
                else "This image has not been tampered with."
            )
            if expected not in response:
                rerun_remaining[label] += 1
            replacements += 1

    if replacements != args.expected_replacements:
        raise RuntimeError(f"expected {args.expected_replacements} replacements, got {replacements}")

    all_missing: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    empty = 0
    for path in args.output_json_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = str(payload["Classification result"])
        response = str(payload.get("Visible forgery traces", ""))
        labels[label] += 1
        if not response.strip():
            empty += 1
        expected = (
            "This image has been tampered with."
            if label == "fake"
            else "This image has not been tampered with."
        )
        if expected not in response:
            all_missing[label] += 1

    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output_json_dir.glob("*.json")):
            archive.write(path, Path("json") / path.name)

    digest = hashlib.sha256(args.zip_path.read_bytes()).hexdigest()
    summary = {
        "replacements": replacements,
        "labels": dict(labels),
        "empty_responses": empty,
        "rerun_still_missing_expected_final_statement": dict(rerun_remaining),
        "all_missing_expected_final_statement": dict(all_missing),
        "zip_path": str(args.zip_path),
        "zip_sha256": digest,
    }
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
