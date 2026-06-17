from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def response_from(row: dict[str, object]) -> str:
    for key in ("response", "answer", "generated_text"):
        if key in row:
            return str(row[key])
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
    return ""


def expected_statement(label: str) -> str:
    return (
        "This image has been tampered with."
        if label == "fake"
        else "This image has not been tampered with."
    )


def ensure_complete_explanation(response: str, label: str) -> tuple[str, bool]:
    """Ensure the final JSON explanation ends with the required conclusion."""
    response = response.strip()
    expected = expected_statement(label)
    if not response:
        return expected, True
    if expected in response:
        return response, False
    separator = "" if response.endswith((".", "!", "?")) else "."
    return f"{response}{separator} Summary: {expected}", True


def write_zip(json_dir: Path, zip_path: Path) -> str:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(json_dir.glob("*.json")):
            archive.write(path, Path("json") / path.name)
    return hashlib.sha256(zip_path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Qwen explanations into DDL-X JSON files.")
    parser.add_argument("--source-json-dir", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--output-json-dir", required=True, type=Path)
    parser.add_argument("--zip-path", required=True, type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    args.output_json_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    final_statement_appended: Counter[str] = Counter()
    empty = 0
    chars: list[int] = []
    total = 0
    for shard in range(args.num_shards):
        manifest = read_jsonl(args.manifest_dir / f"shard_{shard:02d}_manifest.jsonl")
        predictions = read_jsonl(args.prediction_dir / f"shard_{shard:02d}_predictions.jsonl")
        if len(manifest) != len(predictions):
            raise RuntimeError(f"Shard {shard:02d} row mismatch: manifest={len(manifest)} predictions={len(predictions)}")
        for info, pred in zip(manifest, predictions):
            image_id = str(info["image_id"])
            payload = json.loads((args.source_json_dir / f"{image_id}.json").read_text(encoding="utf-8-sig"))
            response = response_from(pred).strip()
            if not response:
                empty += 1
                response = str(payload.get("Visible forgery traces", ""))
            label = str(payload["Classification result"])
            response, appended = ensure_complete_explanation(response, label)
            if appended:
                final_statement_appended[label] += 1
            payload["Visible forgery traces"] = response
            (args.output_json_dir / f"{image_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            counts[label] += 1
            chars.append(len(response))
            total += 1
    digest = write_zip(args.output_json_dir, args.zip_path)
    summary = {
        "total": total,
        "labels": dict(counts),
        "empty_responses_replaced_by_base_text": empty,
        "final_statement_appended": dict(final_statement_appended),
        "response_chars_min": min(chars) if chars else 0,
        "response_chars_mean": (sum(chars) / len(chars)) if chars else 0,
        "response_chars_max": max(chars) if chars else 0,
        "zip_path": str(args.zip_path),
        "zip_sha256": digest,
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
