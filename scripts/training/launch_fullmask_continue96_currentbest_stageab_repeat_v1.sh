#!/usr/bin/env bash
set -euo pipefail

BASE=/home/pengsiran/projects_data/luyihang
PY=/home/pengsiran/anaconda3/envs/vmamba/bin/python
YPY=$BASE/envs/ultralytics_pilot/bin/python
D=/media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-I
DATA=$D/bbox_detector_fullmask_divisible96_continue_v1/full_divisible96_fastval_separate.yaml
MODEL=$BASE/experiments/ddli_bbox_detector_fullmask_continue96_bestdet_stageb3_6gpu_v3/stageb_dev_adapt3/weights/best.pt
REFERENCE=$BASE/experiments/ddli_bbox_detector_fullmask_continue96_bestdet_stageb3_6gpu_v3/score_gate_iou07_sweep_dev24k_v1/summary.json
EXP=$BASE/experiments/ddli_bbox_detector_fullmask_continue96_currentbest_stageab_repeat_v1
TRAINER=$BASE/ddli_detector_v1/train_yolo_with_midstep_checkpoints_6gpu.py
META=$D/dev/metadata_v1
CE=$BASE/experiments/ddli_cls_dev_adapt_e2e_cleanup_v1

test -s "$MODEL"
test -s "$REFERENCE"
test -s "$DATA"
test -s "$TRAINER"
test ! -e "$EXP"
mkdir -p "$EXP/source_snapshot"
cp -p "$MODEL" "$EXP/source_snapshot/current_best_before_repeat_stageab.pt"
cp -p "$REFERENCE" "$EXP/source_snapshot/current_best_score_sweep_summary.json"
cd "$BASE"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

exec > >(tee -a "$EXP/pipeline.log") 2>&1
echo "[$(date '+%F %T')] Repeat old Stage A start: current best -> fullmask continuation"
echo "[$(date '+%F %T')] DDP: GPUs 0-5, global batch=96, workers=6/rank, checkpoints each 4000 steps"
"$YPY" -m torch.distributed.run --standalone --nproc_per_node 6 "$TRAINER" \
  --model "$MODEL" \
  --data "$DATA" \
  --project "$EXP" \
  --name fullmask_continue_repeat \
  --checkpoint-steps 4000,8000,12000,16000,20000 \
  --device 0,1,2,3,4,5 \
  --workers 6 \
  --batch 96 \
  --imgsz 512 \
  --epochs 1 \
  --optimizer AdamW \
  --lr0 0.0002 \
  --seed 20260528

echo "[$(date '+%F %T')] Repeat old Stage B start: dev adaptation, 3 epochs, AdamW lr0=0.0002"
"$YPY" - <<PY
from ultralytics import YOLO
m = YOLO('$EXP/fullmask_continue_repeat/weights/last.pt')
m.train(data='$D/bbox_detector_pilot_yolov8s_devweighted_v1/finetune.yaml',
        imgsz=512, epochs=3, batch=96, device='0,1,2,3,4,5', workers=6,
        project='$EXP', name='stageb_dev_adapt3_repeat', exist_ok=False, cache=False,
        seed=20260528, patience=0, optimizer='AdamW', lr0=0.0002, amp=False)
PY

echo "[$(date '+%F %T')] Predict calib/holdout full detections for score sweep"
export CUDA_VISIBLE_DEVICES=0
"$YPY" "$BASE/ddli_detector_v1/predict_yolo_crop_manifest_all.py" \
  --manifest "$D/dev_bbox_face_crops_v1/calib12k/crop_manifest.csv" \
  --model "$EXP/stageb_dev_adapt3_repeat/weights/best.pt" \
  --out-csv "$EXP/calib_detections.csv" \
  --imgsz 512 --batch 128 --device 0 --predict-conf 0.001 --predict-iou 0.7 --max-det 20
"$YPY" "$BASE/ddli_detector_v1/predict_yolo_crop_manifest_all.py" \
  --manifest "$D/dev_bbox_face_crops_v1/holdout12k/crop_manifest.csv" \
  --model "$EXP/stageb_dev_adapt3_repeat/weights/best.pt" \
  --out-csv "$EXP/holdout_detections.csv" \
  --imgsz 512 --batch 128 --device 0 --predict-conf 0.001 --predict-iou 0.7 --max-det 20

echo "[$(date '+%F %T')] Run score sweep against calib/holdout"
cp "$BASE/ddli_detector_v1/sweep_dev_score_threshold_gate.py" "$BASE/ddli_detector_v1/sweep_dev_score_threshold_gate_repeat_stageab_v1.py"
"$PY" - <<'PY'
from pathlib import Path
p = Path('/home/pengsiran/projects_data/luyihang/ddli_detector_v1/sweep_dev_score_threshold_gate_repeat_stageab_v1.py')
s = p.read_text(encoding='utf-8')
s = s.replace(
    'EXP = Path("/home/pengsiran/projects_data/luyihang/experiments/ddli_bbox_detector_fullmask_continue96_bestdet_stageb3_6gpu_v3")',
    'EXP = Path("/home/pengsiran/projects_data/luyihang/experiments/ddli_bbox_detector_fullmask_continue96_currentbest_stageab_repeat_v1")',
)
s = s.replace(
    'OUT = EXP / "score_gate_iou07_sweep_dev24k_v1"',
    'OUT = EXP / "score_gate_iou07_sweep_dev24k_repeat_stageab_v1"',
)
p.write_text(s, encoding='utf-8')
PY
"$PY" "$BASE/ddli_detector_v1/sweep_dev_score_threshold_gate_repeat_stageab_v1.py"

"$PY" - <<PY
import json
from pathlib import Path
old = json.loads(Path('$REFERENCE').read_text())
new = json.loads(Path('$EXP/score_gate_iou07_sweep_dev24k_repeat_stageab_v1/summary.json').read_text())
old_best = old['best_dev24k_by_score_if_bert07_and_iou_gt_07']
new_best = new['best_dev24k_by_score_if_bert07_and_iou_gt_07']
keys = ['score_est_gt07_bert07', 'score_no_text_per_image', 'fake_iou_gt_07',
        'fake_image_region_iou_mean', 'all_image_iou_mean', 'real_false_box_rate']
out = {
    'old_current_best': old_best,
    'repeat_stageab_best': new_best,
    'delta_repeat_minus_old': {k: new_best[k] - old_best[k] for k in keys},
    'recommend_repeat_for_test': new_best['score_est_gt07_bert07'] > old_best['score_est_gt07_bert07'],
}
Path('$EXP/final_score_comparison_vs_current_best.json').write_text(json.dumps(out, indent=2) + '\\n', encoding='utf-8')
print(json.dumps(out, indent=2))
PY

echo "[$(date '+%F %T')] [done] Repeat old Stage A/Stage B and score sweep completed"
