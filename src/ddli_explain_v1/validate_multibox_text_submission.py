from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json-dir", required=True, type=Path)
    parser.add_argument("--output-json-dir", required=True, type=Path)
    parser.add_argument("--zip-path", required=True, type=Path)
    args = parser.parse_args()

    source_paths = sorted(args.source_json_dir.glob("*.json"))
    output_paths = sorted(args.output_json_dir.glob("*.json"))
    if len(source_paths) != 100000 or len(output_paths) != 100000:
        raise RuntimeError(f"json count invalid: source={len(source_paths)} output={len(output_paths)}")

    errors: list[str] = []
    counts: Counter[str] = Counter()
    for source_path, output_path in zip(source_paths, output_paths):
        if source_path.name != output_path.name:
            errors.append(f"filename mismatch: {source_path.name} != {output_path.name}")
            continue
        source = json.loads(source_path.read_text(encoding="utf-8"))
        output = json.loads(output_path.read_text(encoding="utf-8"))
        if output.get("Classification result") != source.get("Classification result"):
            errors.append(f"{source_path.stem}: classification changed")
        if output.get("Bounding boxes") != source.get("Bounding boxes"):
            errors.append(f"{source_path.stem}: boxes changed")
        if not isinstance(output.get("Visible forgery traces"), str) or not output["Visible forgery traces"].strip():
            errors.append(f"{source_path.stem}: empty text")
        counts[str(output.get("Classification result"))] += 1
    if errors:
        raise RuntimeError("; ".join(errors[:10]))

    with zipfile.ZipFile(args.zip_path) as archive:
        names = archive.namelist()
        if len(names) != 100000 or archive.testzip() is not None:
            raise RuntimeError("zip validation failed")
        if any(not name.startswith("json/") or not name.endswith(".json") for name in names):
            raise RuntimeError("zip entry path invalid")
    print(json.dumps({"json": len(output_paths), "labels": dict(counts), "zip_entries": len(names), "validation": "OK"}, indent=2))


if __name__ == "__main__":
    main()
