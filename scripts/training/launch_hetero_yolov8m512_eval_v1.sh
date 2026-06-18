#!/usr/bin/env bash
set -euo pipefail

BASE=/home/pengsiran/projects_data/luyihang
PY=/home/pengsiran/anaconda3/envs/vmamba/bin/python
YPY=$BASE/envs/ultralytics_pilot/bin/python
D=/media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-I
EXP=$BASE/experiments/ddli_bbox_detector_hetero_yolov8m512_stageab_v1
MODEL=$EXP/stageb_dev_adapt3_b96/weights/best.pt
LOG=$EXP/eval_pipeline.log

test -s "$MODEL"
test -s "$BASE/ddli_detector_v1/predict_yolo_crop_manifest_all.py"
test -s "$BASE/ddli_detector_v1/sweep_dev_score_threshold_gate.py"
test -s "$BASE/ddli_detector_v1/sweep_wbf_hetero_yolov8m_dev24k_v1.py"
test -s "$D/dev_bbox_face_crops_v1/calib12k/crop_manifest.csv"
test -s "$D/dev_bbox_face_crops_v1/holdout12k/crop_manifest.csv"

exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] Hetero YOLOv8m eval start"
cd "$BASE"

echo "[$(date '+%F %T')] Predict calib detections"
export CUDA_VISIBLE_DEVICES=0
"$YPY" "$BASE/ddli_detector_v1/predict_yolo_crop_manifest_all.py" \
  --manifest "$D/dev_bbox_face_crops_v1/calib12k/crop_manifest.csv" \
  --model "$MODEL" \
  --out-csv "$EXP/calib_detections.csv" \
  --imgsz 512 --batch 128 --device 0 --predict-conf 0.001 --predict-iou 0.7 --max-det 20

echo "[$(date '+%F %T')] Predict holdout detections"
"$YPY" "$BASE/ddli_detector_v1/predict_yolo_crop_manifest_all.py" \
  --manifest "$D/dev_bbox_face_crops_v1/holdout12k/crop_manifest.csv" \
  --model "$MODEL" \
  --out-csv "$EXP/holdout_detections.csv" \
  --imgsz 512 --batch 128 --device 0 --predict-conf 0.001 --predict-iou 0.7 --max-det 20

echo "[$(date '+%F %T')] Run single-model score sweep"
cp "$BASE/ddli_detector_v1/sweep_dev_score_threshold_gate.py" "$BASE/ddli_detector_v1/sweep_dev_score_threshold_gate_yolov8m512_v1.py"
"$PY" - <<'PY'
from pathlib import Path
p = Path("/home/pengsiran/projects_data/luyihang/ddli_detector_v1/sweep_dev_score_threshold_gate_yolov8m512_v1.py")
s = p.read_text(encoding="utf-8")
s = s.replace(
    'EXP = Path("/home/pengsiran/projects_data/luyihang/experiments/ddli_bbox_detector_fullmask_continue96_bestdet_stageb3_6gpu_v3")',
    'EXP = Path("/home/pengsiran/projects_data/luyihang/experiments/ddli_bbox_detector_hetero_yolov8m512_stageab_v1")',
)
s = s.replace(
    'OUT = EXP / "score_gate_iou07_sweep_dev24k_v1"',
    'OUT = EXP / "score_gate_iou07_sweep_dev24k_yolov8m512_v1"',
)
p.write_text(s, encoding="utf-8")
PY
"$PY" "$BASE/ddli_detector_v1/sweep_dev_score_threshold_gate_yolov8m512_v1.py"

echo "[$(date '+%F %T')] Write single-model comparison"
"$PY" - <<'PY'
import json
from pathlib import Path

exp = Path("/home/pengsiran/projects_data/luyihang/experiments/ddli_bbox_detector_hetero_yolov8m512_stageab_v1")
baseline = {
    "fake_iou_gt_07": 12046,
    "score_est_gt07_bert07": 0.7653132836862977,
    "best_config": {"cls_gate": 0.20, "conf": 0.15, "nms": 0.25, "max_boxes": 3},
}
summary = json.loads((exp / "score_gate_iou07_sweep_dev24k_yolov8m512_v1" / "summary.json").read_text(encoding="utf-8"))
best = summary["best_dev24k_by_score_if_bert07_and_iou_gt_07"]
out = {
    "baseline_repeat2": baseline,
    "yolov8m_single_best": best,
    "delta_best_minus_repeat2": {
        "fake_iou_gt_07": best["fake_iou_gt_07"] - baseline["fake_iou_gt_07"],
        "score_est_gt07_bert07": best["score_est_gt07_bert07"] - baseline["score_est_gt07_bert07"],
    },
    "accept_single_for_test": (
        best["fake_iou_gt_07"] > baseline["fake_iou_gt_07"]
        and best["score_est_gt07_bert07"] > baseline["score_est_gt07_bert07"]
    ),
}
(exp / "final_comparison_vs_repeat2_baseline.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2), flush=True)
PY

echo "[$(date '+%F %T')] Run WBF sweep with YOLOv8m"
"$PY" "$BASE/ddli_detector_v1/sweep_wbf_hetero_yolov8m_dev24k_v1.py"

echo "[$(date '+%F %T')] [done] Hetero YOLOv8m eval completed"
