from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path


WEIGHTS = {"old": 0.7, "repeat2": 1.35, "yolov8m": 1.0}


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union else 0.0


def load_csvs(pattern: str, model_name: str, pre_conf: float):
    out: dict[str, list[tuple[float, list[float], str]]] = defaultdict(list)
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files match {pattern}")
    rows = kept = 0
    for path in files:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                score = float(row["conf"])
                if score < pre_conf:
                    continue
                out[row["image_id"]].append((score, [float(v) for v in json.loads(row["bbox"])], model_name))
                kept += 1
    return out, {"files": len(files), "rows": rows, "kept_pre_conf": kept}


def cluster_box(items: list[tuple[float, list[float], str]]) -> list[float]:
    denom = sum(max(1e-6, score) * WEIGHTS[name] for score, _box, name in items)
    box = []
    for i in range(4):
        box.append(sum(b[i] * max(1e-6, score) * WEIGHTS[name] for score, b, name in items) / denom)
    return box


def cluster_score(items: list[tuple[float, list[float], str]]) -> float:
    weighted = sum(score * WEIGHTS[name] for score, _box, name in items)
    denom = sum(WEIGHTS[name] for _score, _box, name in items)
    model_bonus = min(1.15, 1.0 + 0.05 * (len({name for _score, _box, name in items}) - 1))
    return (weighted / denom) * model_bonus if denom else 0.0


def weighted_fusion(candidates: list[tuple[float, list[float], str]], match_iou: float):
    clusters: list[dict[str, object]] = []
    for item in sorted(candidates, key=lambda x: x[0] * WEIGHTS[x[2]], reverse=True):
        score, box, name = item
        best_idx = -1
        best_iou = 0.0
        for idx, cluster in enumerate(clusters):
            v = iou(box, cluster["box"])  # type: ignore[arg-type]
            if v > best_iou:
                best_iou = v
                best_idx = idx
        if best_idx >= 0 and best_iou >= match_iou:
            clusters[best_idx]["items"].append(item)  # type: ignore[index,union-attr]
            items = clusters[best_idx]["items"]  # type: ignore[assignment]
            clusters[best_idx]["box"] = cluster_box(items)  # type: ignore[arg-type]
            clusters[best_idx]["score"] = cluster_score(items)  # type: ignore[arg-type]
            clusters[best_idx]["models"] = {x[2] for x in items}  # type: ignore[union-attr]
        else:
            clusters.append({"items": [item], "box": box[:], "score": score * WEIGHTS[name], "models": {name}})
    return clusters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-glob", required=True)
    ap.add_argument("--repeat2-glob", required=True)
    ap.add_argument("--yolov8m-glob", required=True)
    ap.add_argument("--pre-conf", type=float, default=0.125)
    ap.add_argument("--wbf-iou", type=float, default=0.35)
    ap.add_argument("--post-conf", type=float, default=0.175)
    ap.add_argument("--max-boxes", type=int, default=3)
    ap.add_argument("--require-models", type=int, default=2)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary", required=True)
    args = ap.parse_args()

    loaded = {}
    summaries = {}
    for name, pattern in [
        ("old", args.old_glob),
        ("repeat2", args.repeat2_glob),
        ("yolov8m", args.yolov8m_glob),
    ]:
        loaded[name], summaries[name] = load_csvs(pattern, name, args.pre_conf)

    image_ids = set()
    for det in loaded.values():
        image_ids.update(det)

    output: dict[str, dict[str, list[list[int]]]] = {}
    selected_boxes = 0
    clusters_seen = clusters_kept = require_filtered = score_filtered = 0
    for iid in sorted(image_ids):
        candidates: list[tuple[float, list[float], str]] = []
        for name in ("old", "repeat2", "yolov8m"):
            candidates.extend(loaded[name].get(iid, []))
        clusters = weighted_fusion(candidates, args.wbf_iou)
        clusters_seen += len(clusters)
        keep = []
        for c in clusters:
            models = c["models"]  # type: ignore[assignment]
            if len(models) < args.require_models:  # type: ignore[arg-type]
                require_filtered += 1
                continue
            if float(c["score"]) < args.post_conf:
                score_filtered += 1
                continue
            box = [int(round(v)) for v in c["box"]]  # type: ignore[union-attr]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            keep.append((float(c["score"]), len(models), box))  # type: ignore[arg-type]
        keep.sort(key=lambda x: (x[0], x[1]), reverse=True)
        boxes = [box for _score, _models, box in keep[: args.max_boxes]]
        if boxes:
            output[iid] = {"Bounding boxes": boxes}
            selected_boxes += len(boxes)
            clusters_kept += len(boxes)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    summary = {
        "config": {
            "models": "old+repeat2+yolov8m",
            "weights": WEIGHTS,
            "pre_conf": args.pre_conf,
            "wbf_iou": args.wbf_iou,
            "post_conf": args.post_conf,
            "max_boxes": args.max_boxes,
            "require_models": args.require_models,
        },
        "input": summaries,
        "images_with_candidates": len(image_ids),
        "clusters_seen": clusters_seen,
        "clusters_kept_as_boxes": clusters_kept,
        "require_models_filtered": require_filtered,
        "post_conf_filtered": score_filtered,
        "images_with_boxes": len(output),
        "selected_boxes": selected_boxes,
        "out_json": str(out_json),
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
