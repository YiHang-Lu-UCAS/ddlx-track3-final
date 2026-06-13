from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def condition_for(record: dict[str, object]) -> str:
    prompt = str(record["messages"][0]["content"])
    if "labels this image as real" in prompt:
        return "real"
    if "no reliable detector box was retained" in prompt:
        return "fake_nobox"
    return "fake_box"


def template_for(condition: str) -> str:
    if condition == "real":
        return (
            "No visible forgery traces are detected. Facial texture, boundaries, and lighting "
            "appear visually consistent. Summary: This image has not been tampered with."
        )
    if condition == "fake_nobox":
        return (
            "Visible inconsistencies suggest image manipulation, although no reliable localized "
            "region is retained. Summary: This image has been tampered with."
        )
    return (
        "Localized texture and boundary inconsistencies are visible in the detected facial region, "
        "suggesting manipulation. Summary: This image has been tampered with."
    )


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sample-size", default=512, type=int)
    parser.add_argument("--seed", default=20260531, type=int)
    parser.add_argument("--num-shards", default=8, type=int)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(args.seed)
    grouped: dict[str, list[dict[str, object]]] = {"fake": [], "real": []}
    for record in records:
        condition = condition_for(record)
        grouped["real" if condition == "real" else "fake"].append(record)
    for group in grouped.values():
        rng.shuffle(group)

    fake_count = args.sample_size // 2
    real_count = args.sample_size - fake_count
    selected = grouped["fake"][:fake_count] + grouped["real"][:real_count]
    rng.shuffle(selected)

    write_jsonl(args.out_dir / "tune_fixed_512.jsonl", selected)
    for shard_id in range(args.num_shards):
        write_jsonl(args.out_dir / f"tune_fixed_512_shard_{shard_id:02d}.jsonl", selected[shard_id:: args.num_shards])

    templates = []
    conditions = Counter()
    for record in selected:
        condition = condition_for(record)
        conditions[condition] += 1
        templates.append(
            {
                "response": template_for(condition),
                "labels": record["messages"][1]["content"],
                "messages": record["messages"],
                "images": record["images"],
                "condition": condition,
            }
        )
    write_jsonl(args.out_dir / "template_predictions.jsonl", templates)

    benchmark = selected[:8]
    write_jsonl(args.out_dir / "benchmark_8.jsonl", benchmark)
    summary = {
        "source": str(args.source),
        "sample_size": len(selected),
        "seed": args.seed,
        "num_shards": args.num_shards,
        "labels": {"fake": fake_count, "real": real_count},
        "conditions": dict(conditions),
        "files": {
            "fixed_dataset": str(args.out_dir / "tune_fixed_512.jsonl"),
            "template_predictions": str(args.out_dir / "template_predictions.jsonl"),
            "benchmark_dataset": str(args.out_dir / "benchmark_8.jsonl"),
        },
    }
    (args.out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
