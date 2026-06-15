from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep classification thresholds from dev_face_predictions.csv.")
    parser.add_argument("--predictions", required=True, help="CSV produced by eval_dev_faces.py --predictions-out.")
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--end", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--out", default="", help="Optional CSV path for the threshold sweep table.")
    return parser.parse_args()


def load_predictions(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    labels: List[int] = []
    probs: List[float] = []
    with Path(path).expanduser().resolve().open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"label", "fake_prob"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Predictions CSV missing required columns: {sorted(missing)}")
        for row in reader:
            labels.append(int(row["label"]))
            probs.append(float(row["fake_prob"]))
    return np.asarray(labels, dtype=np.int64), np.asarray(probs, dtype=np.float64)


def metrics_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    return {
        "threshold": float(threshold),
        "face_acc": float(accuracy_score(labels, preds)),
        "face_precision": float(precision),
        "face_recall": float(recall),
        "face_f1": float(f1),
        "pred_fake_rate": float(preds.mean()),
    }


def write_table(path: str | Path, rows: List[Dict[str, float]]) -> None:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    labels, probs = load_predictions(args.predictions)

    thresholds = np.arange(args.start, args.end + args.step / 2.0, args.step)
    rows = [metrics_at_threshold(labels, probs, float(round(t, 10))) for t in thresholds]

    best_acc = max(rows, key=lambda r: r["face_acc"])
    best_f1 = max(rows, key=lambda r: r["face_f1"])
    best_precision_with_recall_50 = max(
        (r for r in rows if r["face_recall"] >= 0.50),
        key=lambda r: r["face_precision"],
        default=None,
    )

    print("threshold sweep")
    print("=" * 92)
    print(f"{'thr':>6} {'acc':>9} {'precision':>10} {'recall':>9} {'f1':>9} {'pred_fake':>10}")
    for r in rows:
        print(
            f"{r['threshold']:6.2f} "
            f"{r['face_acc']:9.4f} "
            f"{r['face_precision']:10.4f} "
            f"{r['face_recall']:9.4f} "
            f"{r['face_f1']:9.4f} "
            f"{r['pred_fake_rate']:10.4f}"
        )
    print("=" * 92)
    summary = {
        "num_faces": int(labels.shape[0]),
        "num_real_faces": int((labels == 0).sum()),
        "num_fake_faces": int((labels == 1).sum()),
        "best_acc": best_acc,
        "best_f1": best_f1,
        "best_precision_with_recall_at_least_0.50": best_precision_with_recall_50,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out:
        write_table(args.out, rows)


if __name__ == "__main__":
    main()
