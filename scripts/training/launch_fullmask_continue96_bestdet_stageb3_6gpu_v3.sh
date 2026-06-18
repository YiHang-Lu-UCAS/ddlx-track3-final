#!/usr/bin/env bash
set -euo pipefail

BASE=/home/pengsiran/projects_data/luyihang
PY=/home/pengsiran/anaconda3/envs/vmamba/bin/python
YPY=$BASE/envs/ultralytics_pilot/bin/python
D=/media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-I
DATA=$D/bbox_detector_fullmask_divisible96_continue_v1/full_divisible96_fastval_separate.yaml
MODEL=$BASE/experiments/ddli_bbox_detector_fullmask_divisible128_stageb_compare_v1/dev_adapt/weights/best.pt
REFERENCE=$BASE/experiments/ddli_bbox_detector_fullmask_divisible128_stageb_compare_v1/final_comparison_vs_pilot.json
EXP=$BASE/experiments/ddli_bbox_detector_fullmask_continue96_bestdet_stageb3_6gpu_v3
TRAINER=$BASE/ddli_detector_v1/train_yolo_with_midstep_checkpoints_6gpu.py
META=$D/dev/metadata_v1
CE=$BASE/experiments/ddli_cls_dev_adapt_e2e_cleanup_v1

test -s "$MODEL"
test -s "$REFERENCE"
test -s "$DATA"
test -s "$TRAINER"
test ! -e "$EXP"
mkdir -p "$EXP/source_snapshot"
cp -p "$MODEL" "$EXP/source_snapshot/best_detector_before_fullmask_continue.pt"
cp -p "$REFERENCE" "$EXP/source_snapshot/reference_comparison_before_continue.json"
cd "$BASE"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

exec > >(tee -a "$EXP/pipeline.log") 2>&1
echo "[$(date '+%F %T')] Fullmask continuation: best detector -> 2069472 rows = 21557 global batches"
echo "[$(date '+%F %T')] DDP: GPUs 0-5, global batch=96, workers=6/rank, recovery points each 4000 steps"
"$YPY" -m torch.distributed.run --standalone --nproc_per_node 6 "$TRAINER" \
  --model "$MODEL" \
  --data "$DATA" \
  --project "$EXP" \
  --name fullmask_continue \
  --checkpoint-steps 4000,8000,12000,16000,20000 \
  --device 0,1,2,3,4,5 \
  --workers 6 \
  --batch 96 \
  --imgsz 512 \
  --epochs 1 \
  --optimizer AdamW \
  --lr0 0.0002 \
  --seed 20260528

echo "[$(date '+%F %T')] Stage B start: dev adaptation, 3 epochs, AdamW lr0=0.0002"
"$YPY" - <<PY
from ultralytics import YOLO
m = YOLO('$EXP/fullmask_continue/weights/last.pt')
m.train(data='$D/bbox_detector_pilot_yolov8s_devweighted_v1/finetune.yaml',
        imgsz=512, epochs=3, batch=96, device='0,1,2,3,4,5', workers=6,
        project='$EXP', name='stageb_dev_adapt3', exist_ok=False, cache=False,
        seed=20260528, patience=0, optimizer='AdamW', lr0=0.0002, amp=False)
PY

echo "[$(date '+%F %T')] Holdout comparison with classification gate 0.20"
export CUDA_VISIBLE_DEVICES=0
"$YPY" "$BASE/ddli_detector_v1/predict_yolo_faces_gated.py" --manifest "$D/dev_bbox_face_crops_v1/calib12k/crop_manifest.csv" --face-predictions "$CE/calib/face_predictions.csv" --model "$EXP/stageb_dev_adapt3/weights/best.pt" --out-csv "$EXP/calib_detections.csv" --imgsz 512 --batch 128 --device 0
"$YPY" "$BASE/ddli_detector_v1/predict_yolo_faces_gated.py" --manifest "$D/dev_bbox_face_crops_v1/holdout12k/crop_manifest.csv" --face-predictions "$CE/holdout/face_predictions.csv" --model "$EXP/stageb_dev_adapt3/weights/best.pt" --out-csv "$EXP/holdout_detections.csv" --imgsz 512 --batch 128 --device 0
"$PY" "$BASE/ddli_detector_v1/eval_yolo_e2e_holdout.py" --calib-det "$EXP/calib_detections.csv" --holdout-det "$EXP/holdout_detections.csv" --calib-manifest "$META/dev_faces_localization_calib12k_seed20260524.csv" --holdout-manifest "$META/dev_faces_localization_holdout12k_seed20260524.csv" --calib-face-pred "$CE/calib/face_predictions.csv" --holdout-face-pred "$CE/holdout/face_predictions.csv" --baseline-report "$CE/final_official_score_candidate_report.json" --out "$EXP/candidate_holdout_vs_adapt384.json"
"$PY" - <<PY
import json
from pathlib import Path
candidate = json.loads(Path('$EXP/candidate_holdout_vs_adapt384.json').read_text())['holdout_detector']
reference_report = json.loads(Path('$REFERENCE').read_text())
reference = reference_report['full_detector']
pilot = reference_report['pilot_detector']
keys = ['fake_image_region_iou_mean', 'micro_area_region_iou_over_all_images', 'real_false_box_rate', 'official_proxy_no_text']
out = {
    'candidate_stageb3': candidate,
    'current_best_detector': reference,
    'pilot_detector': pilot,
    'delta_candidate_minus_current_best': {k: candidate[k] - reference[k] for k in keys},
    'recommend_candidate_for_test': candidate['official_proxy_no_text'] > reference['official_proxy_no_text'],
}
Path('$EXP/final_comparison_vs_current_best.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
print(json.dumps(out, indent=2))
PY
echo "[$(date '+%F %T')] [done] Fullmask continuation, Stage B 3 epochs, and holdout comparison completed"
