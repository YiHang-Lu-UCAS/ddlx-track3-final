from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image-level DDL-I dev metrics from per-face predictions.")
    parser.add_argument("--faces-manifest", required=True, help="Path to dev_faces.csv with image_label column.")
    parser.add_argument("--predictions", required=True, help="CSV produced by eval_dev_faces.py --predictions-out.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--aggregation",
        choices=["max", "mean"],
        default="max",
        help="How to aggregate face fake probabilities into one image fake probability.",
    )
    parser.add_argument("--sweep-out", default="", help="Optional CSV path for image-level threshold sweep.")
    return parser.parse_args()


def load_image_labels(faces_manifest: str | Path) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    with Path(faces_manifest).expanduser().resolve().open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "image_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Faces manifest missing required columns: {sorted(missing)}")
        for row in reader:
            image_label = row["image_label"].strip().lower()
            if image_label not in {"real", "fake"}:
                raise ValueError(f"Unexpected image_label={image_label!r}")
            label = 1 if image_label == "fake" else 0
            previous = labels.get(row["image_id"])
            if previous is not None and previous != label:
                raise ValueError(f"Conflicting image labels for {row['image_id']}")
            labels[row["image_id"]] = label
    return labels


def load_face_probs(predictions: str | Path) -> Dict[str, List[float]]:
    probs_by_image: Dict[str, List[float]] = defaultdict(list)
    with Path(predictions).expanduser().resolve().open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "fake_prob"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Predictions CSV missing required columns: {sorted(missing)}")
        for row in reader:
            probs_by_image[row["image_id"]].append(float(row["fake_prob"]))
    return dict(probs_by_image)


def aggregate_probs(probs_by_image: Dict[str, List[float]], aggregation: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for image_id, probs in probs_by_image.items():
        arr = np.asarray(probs, dtype=np.float64)
        if aggregation == "max":
            out[image_id] = float(arr.max())
        elif aggregation == "mean":
            out[image_id] = float(arr.mean())
        else:
            raise ValueError(f"Unexpected aggregation={aggregation!r}")
    return out


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "image_acc": float(accuracy_score(labels, preds)),
        "image_precision": float(precision),
        "image_recall": float(recall),
        "image_f1": float(f1),
        "pred_fake_rate": float(preds.mean()),
    }
    if len(np.unique(labels)) == 2:
        metrics["image_auc"] = float(roc_auc_score(labels, probs))
        metrics["image_ap"] = float(average_precision_score(labels, probs))
    else:
        metrics["image_auc"] = float("nan")
        metrics["image_ap"] = float("nan")
    return metrics


def write_sweep(path: str | Path, labels: np.ndarray, probs: np.ndarray) -> Dict[str, Dict[str, float]]:
    thresholds = np.arange(0.05, 0.95 + 0.025, 0.05)
    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        row = {"threshold": float(round(threshold, 10)), **compute_metrics(labels, probs, float(threshold))}
        rows.append(row)

    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "best_acc": max(rows, key=lambda r: r["image_acc"]),
        "best_f1": max(rows, key=lambda r: r["image_f1"]),
    }


def main() -> None:
    args = parse_args()
    image_labels = load_image_labels(args.faces_manifest)
    probs_by_image = load_face_probs(args.predictions)
    image_probs = aggregate_probs(probs_by_image, args.aggregation)

    shared_ids = sorted(set(image_labels) & set(image_probs))
    labels = np.asarray([image_labels[image_id] for image_id in shared_ids], dtype=np.int64)
    probs = np.asarray([image_probs[image_id] for image_id in shared_ids], dtype=np.float64)

    metrics = compute_metrics(labels, probs, args.threshold)
    metrics.update(
        {
            "threshold": float(args.threshold),
            "aggregation": args.aggregation,
            "num_images_scored": int(labels.shape[0]),
            "num_real_images": int((labels == 0).sum()),
            "num_fake_images": int((labels == 1).sum()),
            "num_manifest_images_with_faces": int(len(image_labels)),
            "num_prediction_images": int(len(image_probs)),
        }
    )

    if args.sweep_out:
        metrics["sweep"] = write_sweep(args.sweep_out, labels, probs)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
