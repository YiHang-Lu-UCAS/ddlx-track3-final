from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from bert_score import BERTScorer


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-type", default="roberta-large")
    parser.add_argument("--num-layers", default=17, type=int)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = [normalize(str(row["response"])) for row in rows]
    references = [normalize(str(row["labels"])) for row in rows]

    scorer = BERTScorer(
        model_type=args.model_type,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        device=args.device,
        idf=False,
        rescale_with_baseline=False,
        lang="en",
    )
    precision, recall, f1 = scorer.score(candidates, references)
    details = []
    for index, row in enumerate(rows):
        details.append(
            {
                "index": index,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "candidate_chars": len(candidates[index]),
                "reference_chars": len(references[index]),
                "condition": row.get("condition"),
            }
        )
    summary = {
        "predictions": str(args.predictions),
        "rows": len(rows),
        "model_type": args.model_type,
        "num_layers": args.num_layers,
        "idf": False,
        "rescale_with_baseline": False,
        "device": args.device,
        "hash": scorer.hash,
        "mean_precision": float(torch.mean(precision)),
        "mean_recall": float(torch.mean(recall)),
        "mean_f1": float(torch.mean(f1)),
        "mean_candidate_chars": sum(len(text) for text in candidates) / len(candidates),
        "mean_reference_chars": sum(len(text) for text in references) / len(references),
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
