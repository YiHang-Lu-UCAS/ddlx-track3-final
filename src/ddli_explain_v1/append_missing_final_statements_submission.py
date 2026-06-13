from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path


def expected_statement(label: str) -> str:
    if label == "fake":
        return "This image has been tampered with."
    if label == "real":
        return "This image has not been tampered with."
    raise ValueError(f"unexpected classification: {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json-dir", required=True, type=Path)
    parser.add_argument("--output-json-dir", required=True, type=Path)
    parser.add_argument("--zip-path", required=True, type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    args = parser.parse_args()

    if args.output_json_dir.exists():
        raise FileExistsError(args.output_json_dir)
    args.output_json_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.input_json_dir, args.output_json_dir)

    appended: Counter[str] = Counter()
    remaining: Counter[str] = Counter()
    empty = 0
    total = 0
    for path in args.output_json_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = str(payload["Classification result"])
        text = str(payload.get("Visible forgery traces", ""))
        expected = expected_statement(label)
        if expected not in text:
            text = text.rstrip() + "\n\nSummary: " + expected
            payload["Visible forgery traces"] = text
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            appended[label] += 1
        if not text.strip():
            empty += 1
        if expected not in text:
            remaining[label] += 1
        total += 1

    if total != 100000:
        raise RuntimeError(f"expected 100000 JSON files, got {total}")
    if empty:
        raise RuntimeError(f"empty responses remain: {empty}")

    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output_json_dir.glob("*.json")):
            archive.write(path, Path("json") / path.name)
    digest = hashlib.sha256(args.zip_path.read_bytes()).hexdigest()
    summary = {
        "total": total,
        "appended_statements": dict(appended),
        "remaining_missing_expected_final_statement": dict(remaining),
        "empty_responses": empty,
        "zip_path": str(args.zip_path),
        "zip_sha256": digest,
    }
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
