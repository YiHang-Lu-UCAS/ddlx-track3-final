#!/usr/bin/env bash
set -euo pipefail

BASE=/home/pengsiran/projects_data/luyihang
YPY=$BASE/envs/ultralytics_pilot/bin/python
D=/media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-I
DATA=$D/bbox_detector_pilot_yolov8s_devweighted_v1/finetune.yaml
EXP=$BASE/experiments/ddli_bbox_detector_hetero_yolov8m512_stageab_v1
STAGEA_RUN=stagea_fullmask_b96
STAGEB_RUN=stageb_dev_adapt3_b96
STAGEA_RESULT=$EXP/stagea_result.json
MODEL=$EXP/$STAGEA_RUN/weights/last.pt

test -d "$EXP"
test -s "$DATA"
test ! -e "$EXP/$STAGEB_RUN"

exec > >(tee -a "$EXP/stageb_watch_pipeline.log") 2>&1
echo "[$(date '+%F %T')] Watch StageA then launch StageB start"

while true; do
  if [ -s "$STAGEA_RESULT" ]; then
    break
  fi
  if ! pgrep -f "launch_hetero_yolov8m512_stagea_v1.sh" >/dev/null && ! pgrep -f "stagea_fullmask_b96" >/dev/null; then
    echo "[$(date '+%F %T')] StageA process not found and no stagea_result.json; abort"
    exit 1
  fi
  sleep 60
done

"$YPY" - <<PY
import json
from pathlib import Path

p = Path("$STAGEA_RESULT")
payload = json.loads(p.read_text(encoding="utf-8"))
if payload.get("loss_has_nan"):
    raise SystemExit("StageA loss_has_nan=true")
if not payload.get("last_pt_exists"):
    raise SystemExit("StageA last_pt missing")
print(json.dumps(payload, indent=2), flush=True)
PY

test -s "$MODEL"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

echo "[$(date '+%F %T')] StageB start: StageA last.pt -> dev adapt 3 epochs"
"$YPY" - <<PY
from ultralytics import YOLO

m = YOLO("$MODEL")
m.train(
    data="$DATA",
    imgsz=512,
    epochs=3,
    batch=96,
    device="0,1,2,3,4,5",
    workers=6,
    project="$EXP",
    name="$STAGEB_RUN",
    exist_ok=False,
    cache=False,
    seed=20260529,
    patience=0,
    optimizer="AdamW",
    lr0=0.0002,
    amp=True,
    plots=False,
)
PY

test -s "$EXP/$STAGEB_RUN/weights/last.pt"
test -s "$EXP/$STAGEB_RUN/weights/best.pt"
test -s "$EXP/$STAGEB_RUN/results.csv"

"$YPY" - <<PY
import csv, json, math
from pathlib import Path

exp = Path("$EXP")
run = exp / "$STAGEB_RUN"
rows = list(csv.DictReader((run / "results.csv").open(newline="", encoding="utf-8")))
loss_has_nan = False
for row in rows:
    for key in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]:
        loss_has_nan = loss_has_nan or math.isnan(float(row[key]))
payload = {
    "stage": "stageb",
    "run": "$STAGEB_RUN",
    "model_init": "$MODEL",
    "data": "$DATA",
    "batch": 96,
    "imgsz": 512,
    "epochs_expected": 3,
    "epochs_recorded": len(rows),
    "loss_has_nan": loss_has_nan,
    "last_pt": str(run / "weights/last.pt"),
    "last_pt_exists": (run / "weights/last.pt").is_file(),
    "best_pt": str(run / "weights/best.pt"),
    "best_pt_exists": (run / "weights/best.pt").is_file(),
    "results_csv": str(run / "results.csv"),
    "last_row": rows[-1] if rows else None,
}
(exp / "stageb_result.json").write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(payload, indent=2), flush=True)
if loss_has_nan or len(rows) < 3:
    raise SystemExit(1)
PY

echo "[$(date '+%F %T')] [pause] StageB checkpoint reached"
