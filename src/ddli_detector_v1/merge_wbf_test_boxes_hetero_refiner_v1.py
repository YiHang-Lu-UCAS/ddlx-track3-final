from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/home/pengsiran/projects_data/luyihang/ddli_segmentation_v1")
from eval_bboxes_against_json import load_json_boxes, load_manifest_unique


ROOT = Path("/home/pengsiran/projects_data/luyihang")
META = Path("/media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-I/dev/metadata_v1")
EXP = ROOT / "experiments"
BASES = ["old", "repeat2", "yolov8m"]
WEIGHTS = {"old": 0.7, "repeat2": 1.35, "yolov8m": 1.0}
TUNE_DETS = {
    "old": EXP / "ddli_bbox_detector_fullmask_continue96_bestdet_stageb3_6gpu_v3" / "tune_detections.csv",
    "repeat2": EXP / "ddli_bbox_detector_fullmask_continue96_repeat2_conservative_lr1e4_v1" / "tune_detections.csv",
    "yolov8m": EXP / "ddli_bbox_detector_hetero_yolov8m512_stageab_v1" / "tune_detections.csv",
}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union else 0.0


def load_det_csv(path: Path, pre_conf: float = 0.0):
    out: dict[str, list[tuple[float, list[float]]]] = defaultdict(list)
    rows = kept = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            score = float(row["conf"])
            if score < pre_conf:
                continue
            out[row["image_id"]].append((score, [float(v) for v in json.loads(row["bbox"])]))
            kept += 1
    return out, {"files": 1, "rows": rows, "kept_pre_conf": kept}


def load_csvs(pattern: str, pre_conf: float):
    out: dict[str, list[tuple[float, list[float]]]] = defaultdict(list)
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
                out[row["image_id"]].append((score, [float(v) for v in json.loads(row["bbox"])]))
                kept += 1
    return out, {"files": len(files), "rows": rows, "kept_pre_conf": kept}


def cluster_box(items):
    denom = sum(max(1e-6, score) * WEIGHTS[name] for score, _box, name in items)
    return [
        max(0, min(1024, sum(box[j] * max(1e-6, score) * WEIGHTS[name] for score, box, name in items) / denom))
        for j in range(4)
    ]


def cluster_score(items):
    denom = sum(WEIGHTS[name] for _score, _box, name in items)
    val = sum(score * WEIGHTS[name] for score, _box, name in items) / denom
    return val * min(1.15, 1.0 + 0.05 * (len({name for _score, _box, name in items}) - 1))


def weighted_fusion(candidates, match_iou: float):
    clusters = []
    for score, box, name in sorted(candidates, key=lambda x: x[0] * WEIGHTS[x[2]], reverse=True):
        best_idx = -1
        best_iou = 0.0
        for idx, cluster in enumerate(clusters):
            v = iou(box, cluster["box"])
            if v > best_iou:
                best_idx = idx
                best_iou = v
        item = (score, box, name)
        if best_idx >= 0 and best_iou >= match_iou:
            clusters[best_idx]["items"].append(item)
            clusters[best_idx]["box"] = cluster_box(clusters[best_idx]["items"])
            clusters[best_idx]["score"] = cluster_score(clusters[best_idx]["items"])
            clusters[best_idx]["models"] = {x[2] for x in clusters[best_idx]["items"]}
        else:
            clusters.append({"items": [item], "box": box[:], "score": score * WEIGHTS[name], "models": {name}})
    return clusters


def build_clusters(dets_by_model, image_ids, pre_conf, wbf_iou, post_conf, max_boxes, require_models):
    out = {}
    stats = defaultdict(int)
    for iid in sorted(image_ids):
        candidates = []
        for name in BASES:
            for score, box in dets_by_model[name].get(iid, []):
                if score >= pre_conf:
                    candidates.append((score, box, name))
        if not candidates:
            continue
        clusters = weighted_fusion(candidates, wbf_iou)
        stats["clusters_seen"] += len(clusters)
        keep = []
        for c in clusters:
            if len(c["models"]) < require_models:
                stats["require_models_filtered"] += 1
                continue
            if c["score"] < post_conf:
                stats["post_conf_filtered"] += 1
                continue
            keep.append(c)
        keep.sort(key=lambda c: (c["score"], len(c["models"])), reverse=True)
        keep = keep[:max_boxes]
        if keep:
            out[iid] = keep
            stats["images_with_boxes"] += 1
            stats["selected_boxes"] += len(keep)
    return out, dict(stats)


def feature(cluster):
    x1, y1, x2, y2 = cluster["box"]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    conf = {m: 0.0 for m in BASES}
    xs, ys, ws, hs = [], [], [], []
    for score, box, name in cluster["items"]:
        conf[name] = max(conf[name], score)
        bw = max(1.0, box[2] - box[0])
        bh = max(1.0, box[3] - box[1])
        xs.append((box[0] + box[2]) / 2.0)
        ys.append((box[1] + box[3]) / 2.0)
        ws.append(bw)
        hs.append(bh)
    return [
        x1 / 1024.0, y1 / 1024.0, x2 / 1024.0, y2 / 1024.0,
        w / 1024.0, h / 1024.0, (w * h) / (1024.0 * 1024.0),
        math.log(max(w / h, 1e-6)), float(cluster["score"]), float(len(cluster["models"])), float(len(cluster["items"])),
        conf["old"], conf["repeat2"], conf["yolov8m"],
        float(np.std(xs) / w), float(np.std(ys) / h), float(np.std(ws) / w), float(np.std(hs) / h),
    ]


def target(pred_box, gt_box):
    px1, py1, px2, py2 = pred_box
    gx1, gy1, gx2, gy2 = gt_box
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0
    gw = max(1.0, gx2 - gx1)
    gh = max(1.0, gy2 - gy1)
    gcx = (gx1 + gx2) / 2.0
    gcy = (gy1 + gy2) / 2.0
    return [
        np.clip((gcx - pcx) / pw, -1, 1),
        np.clip((gcy - pcy) / ph, -1, 1),
        np.clip(math.log(gw / pw), -1.5, 1.5),
        np.clip(math.log(gh / ph), -1.5, 1.5),
    ]


def train_examples(tune_clusters, min_iou: float):
    raw = load_manifest_unique(META / "dev_faces_localization_tune12k_seed20260524.csv")
    xs, ys = [], []
    fake_images = matched = 0
    for iid, row in raw.items():
        if row["image_label"] != "fake":
            continue
        fake_images += 1
        gt_boxes = load_json_boxes(row["json_path"], image_path=row["image_path"], reference_size=1024)
        pairs = []
        for pi, c in enumerate(tune_clusters.get(iid, [])):
            for gi, gt in enumerate(gt_boxes):
                pairs.append((iou(c["box"], gt), pi, gi))
        used_p, used_g = set(), set()
        for v, pi, gi in sorted(pairs, reverse=True):
            if v < min_iou or pi in used_p or gi in used_g:
                continue
            xs.append(feature(tune_clusters[iid][pi]))
            ys.append(target(tune_clusters[iid][pi]["box"], gt_boxes[gi]))
            used_p.add(pi)
            used_g.add(gi)
            matched += 1
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), {"fake_images": fake_images, "matched_examples": matched}


def apply_delta(box, delta, alpha: float, scale: float):
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx, dy, dw, dh = [float(v) for v in delta]
    cx += alpha * np.clip(dx, -0.5, 0.5) * w
    cy += alpha * np.clip(dy, -0.5, 0.5) * h
    nw = w * math.exp(alpha * np.clip(dw, -0.75, 0.75)) * scale
    nh = h * math.exp(alpha * np.clip(dh, -0.75, 0.75)) * scale
    out = [cx - nw / 2.0, cy - nh / 2.0, cx + nw / 2.0, cy + nh / 2.0]
    out = [max(0, min(1024, v)) for v in out]
    if out[2] <= out[0]:
        out[2] = min(1024, out[0] + 1)
    if out[3] <= out[1]:
        out[3] = min(1024, out[1] + 1)
    return out


def clusters_to_json(clusters_by_image, model, alpha: float, scale: float):
    output = {}
    selected_boxes = 0
    for iid, clusters in sorted(clusters_by_image.items()):
        feats = np.asarray([feature(c) for c in clusters], dtype=np.float32)
        deltas = model.predict(feats)
        boxes = []
        for c, delta in zip(clusters, deltas):
            box = [int(round(v)) for v in apply_delta(c["box"], delta, alpha=alpha, scale=scale)]
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
        if boxes:
            output[iid] = {"Bounding boxes": boxes}
            selected_boxes += len(boxes)
    return output, selected_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-glob", required=True)
    ap.add_argument("--repeat2-glob", required=True)
    ap.add_argument("--yolov8m-glob", required=True)
    ap.add_argument("--pre-conf", type=float, default=0.125)
    ap.add_argument("--wbf-iou", type=float, default=0.35)
    ap.add_argument("--post-conf", type=float, default=0.175)
    ap.add_argument("--max-boxes", type=int, default=3)
    ap.add_argument("--require-models", type=int, default=2)
    ap.add_argument("--min-train-iou", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.65)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary", required=True)
    args = ap.parse_args()

    tune_dets = {}
    tune_input = {}
    for name in BASES:
        tune_dets[name], tune_input[name] = load_det_csv(TUNE_DETS[name], pre_conf=0.0)
    tune_ids = set()
    for det in tune_dets.values():
        tune_ids.update(det)
    tune_clusters, tune_cluster_stats = build_clusters(
        tune_dets, tune_ids, args.pre_conf, args.wbf_iou, args.post_conf, args.max_boxes, args.require_models
    )
    x, y, train_stats = train_examples(tune_clusters, min_iou=args.min_train_iou)
    if len(x) < 100:
        raise SystemExit(f"Too few refiner examples: {len(x)}")
    model = MultiOutputRegressor(make_pipeline(StandardScaler(), Ridge(alpha=10.0)))
    model.fit(x, y)

    test_dets = {}
    input_stats = {}
    for name, pattern in [("old", args.old_glob), ("repeat2", args.repeat2_glob), ("yolov8m", args.yolov8m_glob)]:
        test_dets[name], input_stats[name] = load_csvs(pattern, pre_conf=0.0)
    test_ids = set()
    for det in test_dets.values():
        test_ids.update(det)
    test_clusters, test_cluster_stats = build_clusters(
        test_dets, test_ids, args.pre_conf, args.wbf_iou, args.post_conf, args.max_boxes, args.require_models
    )
    output, selected_boxes = clusters_to_json(test_clusters, model, args.alpha, args.scale)

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
            "refiner": "ridge",
            "min_train_iou": args.min_train_iou,
            "alpha": args.alpha,
            "scale": args.scale,
        },
        "tune_input": tune_input,
        "tune_cluster_stats": tune_cluster_stats,
        "train_stats": {**train_stats, "x_shape": list(x.shape), "y_shape": list(y.shape)},
        "test_input": input_stats,
        "test_cluster_stats": test_cluster_stats,
        "images_with_boxes": len(output),
        "selected_boxes": selected_boxes,
        "out_json": str(out_json),
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
