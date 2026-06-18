#!/usr/bin/env bash
set -euo pipefail

BASE=/home/pengsiran/projects_data/luyihang
YPY=$BASE/envs/ultralytics_pilot/bin/python
D=/media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-I
DATA=$D/bbox_detector_fullmask_divisible96_continue_v1/full_divisible96_fastval_separate.yaml
MODEL=$BASE/model_weights/yolov8m.pt
EXP=$BASE/experiments/ddli_bbox_detector_hetero_yolov8m512_stageab_v1
TRAINER=$BASE/ddli_detector_v1/train_yolo_with_midstep_checkpoints_6gpu.py
RUN=stagea_fullmask_b96

test -s "$MODEL"
test -s "$DATA"
test -s "$TRAINER"
test -d "$EXP"
test -s "$EXP/smoke_result.json"
test ! -e "$EXP/$RUN"

mkdir -p "$EXP/source_snapshot"
cp -p "$MODEL" "$EXP/source_snapshot/yolov8m_coco_init.pt"
cp -p "$EXP/smoke_result.json" "$EXP/source_snapshot/smoke_result_before_stagea.json"
cd "$BASE"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

exec > >(tee -a "$EXP/stagea_pipeline.log") 2>&1
echo "[$(date '+%F %T')] StageA start: YOLOv8m COCO init -> formal fullmask"
echo "[$(date '+%F %T')] DDP: GPUs 0-5, global batch=96, imgsz=512, AdamW lr0=0.0002, amp=True"

"$YPY" -m torch.distributed.run --standalone --nproc_per_node 6 "$TRAINER" \
  --model "$MODEL" \
  --data "$DATA" \
  --project "$EXP" \
  --name "$RUN" \
  --checkpoint-steps 4000,8000,12000,16000,20000 \
  --device 0,1,2,3,4,5 \
  --workers 6 \
  --batch 96 \
  --imgsz 512 \
  --epochs 1 \
  --optimizer AdamW \
  --lr0 0.0002 \
  --seed 20260529

test -s "$EXP/$RUN/weights/last.pt"
test -s "$EXP/$RUN/weights/best.pt"
test -s "$EXP/$RUN/results.csv"

"$YPY" - <<PY
import csv, json, math
from pathlib import Path

exp = Path("$EXP")
run = exp / "$RUN"
rows = list(csv.DictReader((run / "results.csv").open(newline="", encoding="utf-8")))
loss_has_nan = False
for row in rows:
    for key in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]:
        loss_has_nan = loss_has_nan or math.isnan(float(row[key]))
payload = {
    "stage": "stagea",
    "run": "$RUN",
    "model_init": "$MODEL",
    "data": "$DATA",
    "batch": 96,
    "imgsz": 512,
    "epochs_expected": 1,
    "epochs_recorded": len(rows),
    "loss_has_nan": loss_has_nan,
    "last_pt": str(run / "weights/last.pt"),
    "last_pt_exists": (run / "weights/last.pt").is_file(),
    "best_pt": str(run / "weights/best.pt"),
    "best_pt_exists": (run / "weights/best.pt").is_file(),
    "results_csv": str(run / "results.csv"),
    "last_row": rows[-1] if rows else None,
}
(exp / "stagea_result.json").write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(payload, indent=2), flush=True)
if loss_has_nan or len(rows) < 1:
    raise SystemExit(1)
PY

echo "[$(date '+%F %T')] [pause] StageA checkpoint reached"
