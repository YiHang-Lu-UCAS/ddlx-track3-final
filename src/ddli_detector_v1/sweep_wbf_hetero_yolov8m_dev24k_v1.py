from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, "/home/pengsiran/projects_data/luyihang/ddli_classification_v1")
sys.path.insert(0, "/home/pengsiran/projects_data/luyihang/ddli_segmentation_v1")

from eval_bboxes_against_json import load_json_boxes, load_manifest_unique
from eval_region_iou_multibox import union_area
from sweep_cls_gate_cleanup_e2e import face_scores


ROOT = Path("/home/pengsiran/projects_data/luyihang")
EXP_ROOT = ROOT / "experiments"
META = Path("/media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-I/dev/metadata_v1")
CLS_EXP = EXP_ROOT / "ddli_cls_dev_adapt_e2e_cleanup_v1"
HETERO_EXP = EXP_ROOT / "ddli_bbox_detector_hetero_yolov8m512_stageab_v1"
OUT = HETERO_EXP / "wbf_hetero_yolov8m_dev24k_v1"

BASELINE = {
    "fake_iou_gt_07": 12046,
    "score_est_gt07_bert07": 0.7653132836862977,
    "best_config": {"cls_gate": 0.20, "conf": 0.15, "nms": 0.25, "max_boxes": 3},
}

MODELS = {
    "old": EXP_ROOT / "ddli_bbox_detector_fullmask_continue96_bestdet_stageb3_6gpu_v3",
    "repeat1": EXP_ROOT / "ddli_bbox_detector_fullmask_continue96_currentbest_stageab_repeat_v1",
    "repeat2": EXP_ROOT / "ddli_bbox_detector_fullmask_continue96_repeat2_conservative_lr1e4_v1",
    "yolov8m": HETERO_EXP,
}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union else 0.0


def load_det(path: Path):
    det = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            det[row["image_id"]].append((float(row["conf"]), json.loads(row["bbox"])))
    return det


def load_context(manifest_path: Path, face_pred_path: Path):
    raw_info = load_manifest_unique(manifest_path)
    info = {}
    for iid, row in raw_info.items():
        gt = load_json_boxes(row["json_path"], image_path=row["image_path"], reference_size=1024)
        info[iid] = {
            "image_label": row["image_label"],
            "gt_boxes": gt,
            "gt_area": union_area(gt),
        }
    return info, face_scores(face_pred_path)


def cluster_box(items):
    denom = sum(max(1e-6, score) * weight for score, _box, weight, _name in items)
    out = []
    for j in range(4):
        out.append(sum(box[j] * max(1e-6, score) * weight for score, box, weight, _name in items) / denom)
    out[0] = max(0, min(1024, out[0]))
    out[1] = max(0, min(1024, out[1]))
    out[2] = max(0, min(1024, out[2]))
    out[3] = max(0, min(1024, out[3]))
    if out[2] < out[0]:
        out[0], out[2] = out[2], out[0]
    if out[3] < out[1]:
        out[1], out[3] = out[3], out[1]
    return out


def cluster_score(items):
    weighted = sum(score * weight for score, _box, weight, _name in items)
    denom = sum(weight for _score, _box, weight, _name in items)
    model_bonus = min(1.15, 1.0 + 0.05 * (len({name for _score, _box, _weight, name in items}) - 1))
    return (weighted / denom) * model_bonus if denom else 0.0


def weighted_fusion(candidates, match_iou):
    clusters = []
    for score, box, weight, model_name in sorted(candidates, key=lambda x: x[0] * x[2], reverse=True):
        best_idx = -1
        best_iou = 0.0
        for idx, cluster in enumerate(clusters):
            v = iou(box, cluster["box"])
            if v > best_iou:
                best_iou = v
                best_idx = idx
        item = (score, box, weight, model_name)
        if best_idx >= 0 and best_iou >= match_iou:
            clusters[best_idx]["items"].append(item)
            clusters[best_idx]["box"] = cluster_box(clusters[best_idx]["items"])
            clusters[best_idx]["score"] = cluster_score(clusters[best_idx]["items"])
            clusters[best_idx]["models"] = {x[3] for x in clusters[best_idx]["items"]}
        else:
            clusters.append({"items": [item], "box": box[:], "score": score * weight, "models": {model_name}})
    return clusters


def build_wbf(dets_by_model, model_names, weights, gate, pre_conf, wbf_iou, post_conf, max_boxes, require_models):
    out = {}
    for iid in gate:
        candidates = []
        for name in model_names:
            for score, box in dets_by_model[name].get(iid, []):
                if score >= pre_conf:
                    candidates.append((score, box, weights[name], name))
        if not candidates:
            continue
        clusters = weighted_fusion(candidates, wbf_iou)
        keep = []
        for c in clusters:
            if c["score"] < post_conf:
                continue
            if require_models and len(c["models"]) < require_models:
                continue
            keep.append((c["score"], [int(round(v)) for v in c["box"]], len(c["models"])))
        keep.sort(key=lambda x: (x[0], x[2]), reverse=True)
        boxes = [box for _score, box, _nm in keep[:max_boxes]]
        if boxes:
            out[iid] = boxes
    return out


def per_image_iou(pred_boxes, row):
    gt = row["gt_boxes"]
    ga = row["gt_area"]
    pa = union_area(pred_boxes)
    ua = union_area(gt + pred_boxes)
    inter = ga + pa - ua
    return inter / ua if ua else 1.0


def evaluate(pred, gate, info, split, config):
    n = len(info)
    fake_n = real_n = 0
    tp = tn = fp = fn = 0
    fake_iou_sum = all_iou_sum = base_sum = 0.0
    fake_zero = real_false_box = num_pred_boxes = 0
    fake_ge = fake_gt = all_ge = all_gt = 0
    bins = defaultdict(int)
    for iid, row in info.items():
        truth = row["image_label"] == "fake"
        pred_fake = iid in gate
        pred_boxes = pred.get(iid, [])
        v = per_image_iou(pred_boxes, row)
        acc = 1.0 if pred_fake == truth else 0.0
        base_sum += 0.2 * acc + 0.5 * v
        all_iou_sum += v
        num_pred_boxes += len(pred_boxes)
        all_ge += int(v >= 0.7)
        all_gt += int(v > 0.7)
        if truth:
            fake_n += 1
            fake_iou_sum += v
            fake_zero += int(v == 0.0)
            fake_ge += int(v >= 0.7)
            fake_gt += int(v > 0.7)
            if v < 0.3:
                bins["fake_iou_lt_03"] += 1
            elif v < 0.5:
                bins["fake_iou_03_05"] += 1
            elif v < 0.6:
                bins["fake_iou_05_06"] += 1
            elif v < 0.7:
                bins["fake_iou_06_07"] += 1
            elif v < 0.8:
                bins["fake_iou_07_08"] += 1
            elif v < 0.9:
                bins["fake_iou_08_09"] += 1
            else:
                bins["fake_iou_ge_09"] += 1
        else:
            real_n += 1
            real_false_box += int(bool(pred_boxes))
        if truth and pred_fake:
            tp += 1
        elif truth:
            fn += 1
        elif pred_fake:
            fp += 1
        else:
            tn += 1
    out = {
        "split": split,
        **config,
        "num_images": n,
        "fake_images": fake_n,
        "real_images": real_n,
        "classification_acc": (tp + tn) / n,
        "fake_image_region_iou_mean": fake_iou_sum / fake_n,
        "all_image_iou_mean": all_iou_sum / n,
        "fake_zero_iou_rate": fake_zero / fake_n,
        "real_false_box_rate": real_false_box / real_n,
        "num_pred_boxes": num_pred_boxes,
        "pred_fake_images": len(gate),
        "fake_iou_ge_07": fake_ge,
        "fake_iou_gt_07": fake_gt,
        "fake_iou_ge_07_rate": fake_ge / fake_n,
        "fake_iou_gt_07_rate": fake_gt / fake_n,
        "all_iou_ge_07": all_ge,
        "all_iou_gt_07": all_gt,
        "score_no_text_per_image": base_sum / n,
    }
    for key in ["fake_iou_lt_03", "fake_iou_03_05", "fake_iou_05_06", "fake_iou_06_07", "fake_iou_07_08", "fake_iou_08_09", "fake_iou_ge_09"]:
        out[key] = bins[key]
    out["score_est_gt07_bert07"] = out["score_no_text_per_image"] + 0.3 * 0.7 * (fake_gt + (real_n - real_false_box)) / n
    out["score_est_ge07_bert07"] = out["score_no_text_per_image"] + 0.3 * 0.7 * (fake_ge + (real_n - real_false_box)) / n
    return out


def merge_rows(a, b):
    out = dict(a)
    out["split"] = "dev24k"
    sum_keys = [
        "num_images", "fake_images", "real_images", "num_pred_boxes", "pred_fake_images",
        "fake_iou_ge_07", "fake_iou_gt_07", "all_iou_ge_07", "all_iou_gt_07",
        "fake_iou_lt_03", "fake_iou_03_05", "fake_iou_05_06", "fake_iou_06_07",
        "fake_iou_07_08", "fake_iou_08_09", "fake_iou_ge_09",
    ]
    for k in sum_keys:
        out[k] = a[k] + b[k]
    n = out["num_images"]
    fake_n = out["fake_images"]
    real_n = out["real_images"]
    for k in ["classification_acc", "all_image_iou_mean", "score_no_text_per_image"]:
        out[k] = (a[k] * a["num_images"] + b[k] * b["num_images"]) / n
    out["fake_image_region_iou_mean"] = (a["fake_image_region_iou_mean"] * a["fake_images"] + b["fake_image_region_iou_mean"] * b["fake_images"]) / fake_n
    out["fake_zero_iou_rate"] = (a["fake_zero_iou_rate"] * a["fake_images"] + b["fake_zero_iou_rate"] * b["fake_images"]) / fake_n
    out["real_false_box_rate"] = (a["real_false_box_rate"] * a["real_images"] + b["real_false_box_rate"] * b["real_images"]) / real_n
    out["fake_iou_ge_07_rate"] = out["fake_iou_ge_07"] / fake_n
    out["fake_iou_gt_07_rate"] = out["fake_iou_gt_07"] / fake_n
    out["score_est_gt07_bert07"] = (a["score_est_gt07_bert07"] * a["num_images"] + b["score_est_gt07_bert07"] * b["num_images"]) / n
    out["score_est_ge07_bert07"] = (a["score_est_ge07_bert07"] * a["num_images"] + b["score_est_ge07_bert07"] * b["num_images"]) / n
    return out


def write_csv(path: Path, rows):
    fields = [
        "rank", "split", "models", "weight_name", "pre_conf", "wbf_iou", "post_conf",
        "max_boxes", "require_models", "cls_gate", "score_est_gt07_bert07",
        "score_no_text_per_image", "fake_iou_gt_07", "fake_iou_gt_07_rate",
        "fake_iou_ge_07", "fake_image_region_iou_mean", "all_image_iou_mean",
        "real_false_box_rate", "fake_zero_iou_rate", "num_pred_boxes",
        "pred_fake_images", "fake_iou_06_07", "fake_iou_07_08",
        "fake_iou_08_09", "fake_iou_ge_09",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            item = dict(row)
            item["rank"] = rank
            writer.writerow({k: item.get(k, "") for k in fields})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    calib_info, calib_scores = load_context(
        META / "dev_faces_localization_calib12k_seed20260524.csv",
        CLS_EXP / "calib" / "face_predictions.csv",
    )
    holdout_info, holdout_scores = load_context(
        META / "dev_faces_localization_holdout12k_seed20260524.csv",
        CLS_EXP / "holdout" / "face_predictions.csv",
    )
    dets = {
        name: {
            "calib": load_det(exp / "calib_detections.csv"),
            "holdout": load_det(exp / "holdout_detections.csv"),
        }
        for name, exp in MODELS.items()
    }
    model_combos = [
        ("repeat2+yolov8m", ["repeat2", "yolov8m"]),
        ("old+repeat2+yolov8m", ["old", "repeat2", "yolov8m"]),
        ("old+repeat1+repeat2+yolov8m", ["old", "repeat1", "repeat2", "yolov8m"]),
    ]
    weight_schemes = {
        "equal": {"old": 1.0, "repeat1": 1.0, "repeat2": 1.0, "yolov8m": 1.0},
        "favor_repeat2": {"old": 0.7, "repeat1": 0.9, "repeat2": 1.35, "yolov8m": 1.0},
        "favor_yolov8m": {"old": 0.7, "repeat1": 0.9, "repeat2": 1.0, "yolov8m": 1.35},
        "favor_repeat2_yolov8m": {"old": 0.65, "repeat1": 0.85, "repeat2": 1.25, "yolov8m": 1.25},
    }
    cls_gate = 0.20
    calib_gate = {iid for iid, score in calib_scores.items() if score >= cls_gate}
    holdout_gate = {iid for iid, score in holdout_scores.items() if score >= cls_gate}
    pre_confs = [0.05, 0.075, 0.10, 0.125, 0.15]
    wbf_ious = [0.35, 0.45, 0.55, 0.65]
    post_confs = [0.08, 0.10, 0.125, 0.15, 0.175]
    max_boxes_values = [3, 4, 5]
    require_models_values = [0, 2]

    rows = []
    total = len(model_combos) * len(weight_schemes) * len(pre_confs) * len(wbf_ious) * len(post_confs) * len(max_boxes_values) * len(require_models_values)
    done = 0
    for combo_name, names in model_combos:
        for weight_name, weights in weight_schemes.items():
            for pre_conf in pre_confs:
                for wbf_iou in wbf_ious:
                    for post_conf in post_confs:
                        for max_boxes in max_boxes_values:
                            for require_models in require_models_values:
                                config = {
                                    "models": combo_name,
                                    "weight_name": weight_name,
                                    "pre_conf": pre_conf,
                                    "wbf_iou": wbf_iou,
                                    "post_conf": post_conf,
                                    "max_boxes": max_boxes,
                                    "require_models": require_models,
                                    "cls_gate": cls_gate,
                                }
                                cp = build_wbf({n: dets[n]["calib"] for n in names}, names, weights, calib_gate, pre_conf, wbf_iou, post_conf, max_boxes, require_models)
                                hp = build_wbf({n: dets[n]["holdout"] for n in names}, names, weights, holdout_gate, pre_conf, wbf_iou, post_conf, max_boxes, require_models)
                                rows.append(merge_rows(
                                    evaluate(cp, calib_gate, calib_info, "calib12k", config),
                                    evaluate(hp, holdout_gate, holdout_info, "holdout12k", config),
                                ))
                                done += 1
                                if done % 100 == 0:
                                    print(f"[wbf-yolov8m-sweep] {done}/{total}", flush=True)

    by_score = sorted(rows, key=lambda r: (r["score_est_gt07_bert07"], r["fake_iou_gt_07"], r["fake_image_region_iou_mean"]), reverse=True)
    by_count = sorted(rows, key=lambda r: (r["fake_iou_gt_07"], r["score_est_gt07_bert07"], r["fake_image_region_iou_mean"]), reverse=True)
    write_csv(OUT / "dev24k_ranked_by_score.csv", by_score)
    write_csv(OUT / "dev24k_ranked_by_fake_iou_gt07.csv", by_count)
    best = by_score[0]
    summary = {
        "baseline_repeat2": BASELINE,
        "grid": {
            "cls_gate": cls_gate,
            "pre_conf": pre_confs,
            "wbf_iou": wbf_ious,
            "post_conf": post_confs,
            "max_boxes": max_boxes_values,
            "require_models": require_models_values,
            "weights": list(weight_schemes),
            "model_combos": [name for name, _ in model_combos],
        },
        "grid_size": len(rows),
        "best_by_score": best,
        "best_by_fake_iou_gt07": by_count[0],
        "top20_by_score": by_score[:20],
        "top20_by_fake_iou_gt07": by_count[:20],
        "accept_for_test": (
            best["fake_iou_gt_07"] > BASELINE["fake_iou_gt_07"]
            and best["score_est_gt07_bert07"] > BASELINE["score_est_gt07_bert07"]
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    final = {
        "baseline_repeat2": BASELINE,
        "wbf_sweep_summary": str(OUT / "summary.json"),
        "best_wbf": best,
        "delta_best_minus_repeat2": {
            "fake_iou_gt_07": best["fake_iou_gt_07"] - BASELINE["fake_iou_gt_07"],
            "score_est_gt07_bert07": best["score_est_gt07_bert07"] - BASELINE["score_est_gt07_bert07"],
        },
        "accept_for_test": summary["accept_for_test"],
    }
    (HETERO_EXP / "final_wbf_comparison_vs_repeat2_baseline.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
