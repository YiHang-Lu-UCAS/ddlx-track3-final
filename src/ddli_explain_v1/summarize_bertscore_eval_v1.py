from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_score(root: Path, name: str) -> dict[str, object]:
    score = json.loads((root / name).read_text(encoding="utf-8"))
    return {
        "rows": score["rows"],
        "mean_precision": score["mean_precision"],
        "mean_recall": score["mean_recall"],
        "mean_f1": score["mean_f1"],
        "mean_candidate_chars": score["mean_candidate_chars"],
        "mean_reference_chars": score["mean_reference_chars"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()

    summary = {
        "metric": {
            "implementation": "bert-score",
            "model": "/home/pengsiran/projects_data/luyihang/models/roberta-large",
            "num_layers": 17,
            "idf": False,
            "rescale_with_baseline": False,
            "selection_metric": "mean_f1",
        },
        "frozen_decoder": {
            "checkpoint": (
                "/home/pengsiran/projects_data/luyihang/experiments/ddli_explain_qwen25vl3b_lora_v1/"
                "formal_run_v1/v0-20260526-222731/checkpoint-1500"
            ),
            "max_new_tokens": 512,
            "max_batch_size_per_gpu": 8,
            "gpus": "0,1,2,3,4,5",
            "summary_suffix_normalization": True,
        },
        "tune512": {
            "template": load_score(args.root, "template_predictions_bertscore.json"),
            "checkpoint1500_max256": load_score(
                args.root, "checkpoint1500_tune512_max256_predictions_suffixfixed_bertscore.json"
            ),
            "checkpoint1500_max384": load_score(
                args.root, "checkpoint1500_tune512_max384_predictions_suffixfixed_bertscore.json"
            ),
            "checkpoint1500_max512": load_score(
                args.root, "checkpoint1500_tune512_max512_predictions_suffixfixed_bertscore.json"
            ),
            "checkpoint1500_max640": load_score(
                args.root, "checkpoint1500_tune512_predictions_suffixfixed_bertscore.json"
            ),
        },
        "calib512": {
            "checkpoint1500_max384": load_score(
                args.root, "checkpoint1500_calib512_max384_predictions_suffixfixed_bertscore.json"
            ),
            "checkpoint1500_max512": load_score(
                args.root, "checkpoint1500_calib512_max512_predictions_suffixfixed_bertscore.json"
            ),
            "checkpoint1500_max640": load_score(
                args.root, "checkpoint1500_calib512_max640_predictions_suffixfixed_bertscore.json"
            ),
        },
        "holdout512": {
            "template": load_score(args.root, "holdout_template_predictions_bertscore.json"),
            "checkpoint1500_max512": load_score(
                args.root, "checkpoint1500_holdout512_max512_predictions_suffixfixed_bertscore.json"
            ),
        },
        "decision": {
            "retrain_qwen_v2": False,
            "reason": (
                "checkpoint-1500 clearly beats the fixed template baseline; max512 is the best calib setting "
                "among max384/max512/max640 while remaining faster than max640."
            ),
            "full_test_generated": False,
        },
    }
    out = args.root / "bertscore_eval_summary_v1.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
