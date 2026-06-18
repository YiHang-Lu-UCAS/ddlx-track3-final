from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ms-swift multimodal JSONL training data.")
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()

    rows = 0
    missing_images: list[str] = []
    with args.jsonl.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            images = row.get("images")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{args.jsonl}:{line_number}: expected at least two messages")
            if not isinstance(images, list) or not images:
                raise ValueError(f"{args.jsonl}:{line_number}: expected a non-empty images list")
            if args.check_images:
                missing_images.extend(str(path) for path in images if not Path(str(path)).is_file())
            rows += 1

    if rows == 0:
        raise ValueError(f"No records found in {args.jsonl}")
    if missing_images:
        preview = "\n".join(missing_images[:20])
        raise FileNotFoundError(f"Missing {len(missing_images)} referenced images; first entries:\n{preview}")
    print(json.dumps({"jsonl": str(args.jsonl), "rows": rows, "missing_images": 0}, indent=2))


if __name__ == "__main__":
    main()
