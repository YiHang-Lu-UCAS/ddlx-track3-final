from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate predicted tamper bboxes against DDL-I JSON Bounding boxes.")
    p.add_argument("--manifest", required=True, help="dev_faces.csv or image manifest containing image_id/json_path/image_label.")
    p.add_argument("--pred-json", required=True, help="Prediction JSON from predict_bboxes.py.")
    p.add_argument("--out-json", required=True, help="Output summary JSON.")
    p.add_argument("--out-csv", default="", help="Optional per-image result CSV.")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument(
        "--gt-coordinate-size",
        type=int,
        default=1024,
        help="Scale JSON GT boxes from this square coordinate space to actual image size. Use 0 to disable.",
    )
    p.add_argument("--missing-pred-as-empty", action="store_true", help="Evaluate manifest images not in pred-json as empty predictions.")
    return p.parse_args()


def parse_box_list(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except Exception:
            try:
                value = ast.literal_eval(text)
            except Exception:
                return []
    if not isinstance(value, list):
        return []
    boxes: list[list[float]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 4:
            try:
                x1, y1, x2, y2 = [float(v) for v in item]
            except Exception:
                continue
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
    return boxes


def scale_boxes_from_reference(boxes: list[list[float]], image_path: str, reference_size: int) -> list[list[float]]:
    if reference_size <= 0 or not image_path:
        return boxes
    try:
        with Image.open(image_path) as im:
            w, h = im.size
    except Exception:
        return boxes
    sx = float(w) / float(reference_size)
    sy = float(h) / float(reference_size)
    scaled = []
    for x1, y1, x2, y2 in boxes:
        scaled.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
    return scaled


def load_json_boxes(path: str | Path, image_path: str = "", reference_size: int = 0) -> list[list[float]]:
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    boxes = parse_box_list(data.get("Bounding boxes"))
    return scale_boxes_from_reference(boxes, image_path, reference_size)


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def greedy_match(preds: list[list[float]], gts: list[list[float]], iou_threshold: float) -> tuple[int, int, int, list[float]]:
    pairs: list[tuple[float, int, int]] = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            iou = box_iou(p, g)
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matched_ious: list[float] = []
    for iou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matched_ious.append(iou)
    tp = len(matched_ious)
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn, matched_ious


def best_iou_stats(preds: list[list[float]], gts: list[list[float]]) -> tuple[float, float]:
    if not preds or not gts:
        return 0.0, 0.0
    best_gt = []
    for g in gts:
        best_gt.append(max(box_iou(p, g) for p in preds))
    best_pred = []
    for p in preds:
        best_pred.append(max(box_iou(p, g) for g in gts))
    return float(np.mean(best_gt)), float(np.mean(best_pred))


def load_manifest_unique(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("image_id") or Path(row.get("image_path", "")).name
            if not image_id or image_id in out:
                continue
            out[image_id] = {
                "image_id": image_id,
                "json_path": row.get("json_path", ""),
                "image_path": row.get("image_path", ""),
                "image_label": row.get("image_label", ""),
            }
    return out


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def main() -> None:
    args = parse_args()
    manifest = load_manifest_unique(Path(args.manifest).expanduser().resolve())
    with Path(args.pred_json).expanduser().open("r", encoding="utf-8") as f:
        pred_payload = json.load(f)

    image_ids = sorted(manifest) if args.missing_pred_as_empty else sorted(set(manifest) & set(pred_payload))
    rows: list[dict[str, Any]] = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    matched_ious_all: list[float] = []
    best_gt_all: list[float] = []
    best_pred_all: list[float] = []

    image_cls = {
        "true_fake_pred_fake": 0,
        "true_fake_pred_real": 0,
        "true_real_pred_fake": 0,
        "true_real_pred_real": 0,
    }

    for image_id in image_ids:
        info = manifest.get(image_id, {"json_path": "", "image_label": ""})
        gts = load_json_boxes(
            info.get("json_path", ""),
            image_path=info.get("image_path", ""),
            reference_size=args.gt_coordinate_size,
        )
        pred_item = pred_payload.get(image_id, {})
        preds = parse_box_list(pred_item.get("Bounding boxes") if isinstance(pred_item, dict) else pred_item)
        tp, fp, fn, matched_ious = greedy_match(preds, gts, args.iou_threshold)
        bg, bp = best_iou_stats(preds, gts)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        matched_ious_all.extend(matched_ious)
        if gts:
            best_gt_all.append(bg)
        if preds:
            best_pred_all.append(bp)

        gt_fake = bool(gts)
        pred_fake = bool(preds)
        if gt_fake and pred_fake:
            image_cls["true_fake_pred_fake"] += 1
        elif gt_fake and not pred_fake:
            image_cls["true_fake_pred_real"] += 1
        elif not gt_fake and pred_fake:
            image_cls["true_real_pred_fake"] += 1
        else:
            image_cls["true_real_pred_real"] += 1

        rows.append(
            {
                "image_id": image_id,
                "image_label": info.get("image_label", ""),
                "num_gt": len(gts),
                "num_pred": len(preds),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "matched_iou_mean": float(np.mean(matched_ious)) if matched_ious else 0.0,
                "best_gt_iou_mean": bg,
                "best_pred_iou_mean": bp,
                "gt_boxes": json.dumps(gts, ensure_ascii=False),
                "pred_boxes": json.dumps(preds, ensure_ascii=False),
            }
        )

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    image_acc = safe_div(image_cls["true_fake_pred_fake"] + image_cls["true_real_pred_real"], len(image_ids))
    image_precision = safe_div(image_cls["true_fake_pred_fake"], image_cls["true_fake_pred_fake"] + image_cls["true_real_pred_fake"])
    image_recall = safe_div(image_cls["true_fake_pred_fake"], image_cls["true_fake_pred_fake"] + image_cls["true_fake_pred_real"])
    image_f1 = safe_div(2 * image_precision * image_recall, image_precision + image_recall)

    summary = {
        "iou_threshold": args.iou_threshold,
        "gt_coordinate_size": args.gt_coordinate_size,
        "num_images": len(image_ids),
        "num_manifest_images": len(manifest),
        "num_prediction_images": len(pred_payload),
        "bbox_tp": tp,
        "bbox_fp": fp,
        "bbox_fn": fn,
        "bbox_precision": precision,
        "bbox_recall": recall,
        "bbox_f1": f1,
        "matched_iou_mean": float(np.mean(matched_ious_all)) if matched_ious_all else 0.0,
        "best_gt_iou_mean": float(np.mean(best_gt_all)) if best_gt_all else 0.0,
        "best_pred_iou_mean": float(np.mean(best_pred_all)) if best_pred_all else 0.0,
        "image_acc_has_box": image_acc,
        "image_precision_has_box": image_precision,
        "image_recall_has_box": image_recall,
        "image_f1_has_box": image_f1,
        "image_confusion_has_box": image_cls,
    }

    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.out_csv:
        out_csv = Path(args.out_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image_id"])
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
