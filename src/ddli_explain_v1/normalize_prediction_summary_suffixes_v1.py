from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FAKE_SUFFIX = "Summary: This image has been tampered with."
REAL_SUFFIX = "Summary: This image has not been tampered with."
SUMMARY_RE = re.compile(
    r"(?:summary\s*:\s*)?this image has(?: not)? been tampered with\.?",
    flags=re.IGNORECASE,
)


def expected_suffix(row: dict[str, object]) -> str:
    messages = row["messages"]
    prompt = str(messages[0]["content"])
    return REAL_SUFFIX if "labels this image as real" in prompt else FAKE_SUFFIX


def normalize_response(response: str, suffix: str) -> str:
    text = SUMMARY_RE.sub("", response)
    text = re.sub(r"\s+", " ", text).strip(" \n\t-")
    return f"{text}\n\n{suffix}" if text else suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = {"rows": len(rows), "changed": 0, "fake": 0, "real": 0}
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            suffix = expected_suffix(row)
            key = "real" if suffix == REAL_SUFFIX else "fake"
            counts[key] += 1
            normalized = normalize_response(str(row["response"]), suffix)
            counts["changed"] += normalized != row["response"]
            row["response"] = normalized
            row["summary_suffix_normalized"] = True
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
