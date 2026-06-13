from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-index", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--global-batch", default=96, type=int)
    ap.add_argument("--smoke-batches", default=10, type=int)
    args = ap.parse_args()

    lines = args.source_index.read_text(encoding="utf-8").splitlines()
    kept_count = len(lines) - (len(lines) % args.global_batch)
    smoke_count = args.global_batch * args.smoke_batches
    if smoke_count > kept_count:
        raise ValueError("Smoke subset exceeds full divisible index.")

    output_root = args.output_root
    indices = output_root / "indices"
    indices.mkdir(parents=True, exist_ok=False)

    full_path = indices / f"train_divisible_{kept_count}.txt"
    smoke_path = indices / f"train_smoke_{smoke_count}.txt"
    full_path.write_text("\n".join(lines[:kept_count]) + "\n", encoding="utf-8")
    smoke_path.write_text("\n".join(lines[:smoke_count]) + "\n", encoding="utf-8")

    val_source = args.source_index.parent / "val_all.txt"
    if not val_source.is_file():
        raise FileNotFoundError(val_source)

    def write_yaml(path: Path, train_name: str) -> None:
        path.write_text(
            "\n".join(
                [
                    f"path: {output_root}",
                    f"train: indices/{train_name}",
                    f"val: {val_source}",
                    "names:",
                    "  0: tamper",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    write_yaml(output_root / "full_divisible96.yaml", full_path.name)
    write_yaml(output_root / "smoke_divisible96.yaml", smoke_path.name)

    summary = {
        "source_index": str(args.source_index),
        "source_rows": len(lines),
        "global_batch": args.global_batch,
        "full_rows": kept_count,
        "dropped_from_source_tail": len(lines) - kept_count,
        "full_batches": kept_count // args.global_batch,
        "smoke_rows": smoke_count,
        "smoke_batches": args.smoke_batches,
        "selection": "ordered prefix; source and original dataset are unchanged",
    }
    (output_root / "index_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
