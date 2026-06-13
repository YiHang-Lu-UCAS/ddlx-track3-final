#!/usr/bin/env bash
set -euo pipefail

BASE=/home/pengsiran/projects_data/luyihang
PY=/home/pengsiran/anaconda3/envs/vmamba/bin/python
YPY=$BASE/envs/ultralytics_pilot/bin/python
SRC=/media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-X/metadata_clsadapt020_adapt384_cleanup_top3_v1
RAW_FACE=/media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-X/metadata_fast/face_shard_outputs
SCRIPT=$BASE/ddli_detector_v1

OLD_WORK=/media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-X/metadata_detector_fullmask_continue96_stageb3_newdet_textsame_v4_6gpu
REPEAT2_WORK=/media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-X/metadata_detector_repeat2_lr1e4_conf015_nms025_textsame_v1_6gpu
Y8M_WORK=/media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-X/metadata_detector_hetero_yolov8m512_stageab_v1_6gpu
WBF_WORK=/media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-X/metadata_detector_wbf_old_repeat2_yolov8m_v1

Y8M_MODEL=$BASE/experiments/ddli_bbox_detector_hetero_yolov8m512_stageab_v1/stageb_dev_adapt3_b96/weights/best.pt
NORMAL_TEXT_ZIP=$BASE/submissions/ddl_x_test_detector_repeat2_lr1e4_conf015_nms025_textsame_v1_5variants_6gpu/normal/submission_ddl_x_test_repeat2_lr1e4_conf015_nms025_textsame_v1_normal.zip
EYES_TEXT_ZIP=$BASE/submissions/ddl_x_test_detector_repeat2_lr1e4_conf015_nms025_textrerun_fake_nobox_v3_fallback4_6gpu/fake_nobox_eyes/submission_ddl_x_test_repeat2_lr1e4_conf015_nms025_textrerun_fake_nobox_v3_fake_nobox_eyes_fake_nobox_textrerun.zip
EYES_MOUTH_TEXT_ZIP=$BASE/submissions/ddl_x_test_detector_repeat2_lr1e4_conf015_nms025_textrerun_fake_nobox_v3_fallback4_6gpu/fake_nobox_eyes_mouth/submission_ddl_x_test_repeat2_lr1e4_conf015_nms025_textrerun_fake_nobox_v3_fake_nobox_eyes_mouth_fake_nobox_textrerun.zip
NOSE_EYES_MOUTH_TEXT_ZIP=$BASE/submissions/ddl_x_test_detector_repeat2_lr1e4_conf015_nms025_textrerun_fake_nobox_v3_fallback4_6gpu/fake_nobox_nose_eyes_mouth/submission_ddl_x_test_repeat2_lr1e4_conf015_nms025_textrerun_fake_nobox_v3_fake_nobox_nose_eyes_mouth_fake_nobox_textrerun.zip
TAG=wbf_old_repeat2_yolov8m_pr125_iou035_post175_req2_textreuse_v1
SUB=$BASE/submissions/ddl_x_test_detector_${TAG}_4variants
BATCH=64

test -s "$Y8M_MODEL"
test -s "$SRC/image_scores.csv"
test -s "$SRC/test_images.csv"
test -d "$SRC/seg_face_shards"
test -d "$RAW_FACE"
test -d "$OLD_WORK/det_shards"
test -d "$REPEAT2_WORK/det_shards"
test -s "$NORMAL_TEXT_ZIP"
test -s "$EYES_TEXT_ZIP"
test -s "$EYES_MOUTH_TEXT_ZIP"
test -s "$NOSE_EYES_MOUTH_TEXT_ZIP"
test ! -e "$SUB"

mkdir -p "$Y8M_WORK/det_face_shards" "$Y8M_WORK/det_shards" "$WBF_WORK" "$SUB/logs"
exec > >(tee -a "$SUB/pipeline.log") 2>&1

echo "[$(date '+%F %T')] WBF full-test 4 variants start tag=$TAG"
echo "[$(date '+%F %T')] WBF config: old+repeat2+yolov8m favor_repeat2 pre_conf=0.125 wbf_iou=0.35 post_conf=0.175 max_boxes=3 require_models=2"

if ! ls "$Y8M_WORK/det_shards"/shard_*.csv >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] Build YOLOv8m six face shards"
  "$PY" - <<PY
import csv
from pathlib import Path
src = Path('$SRC/seg_face_shards')
rows = []
fieldnames = None
for p in sorted(src.glob('shard_*.csv')):
    with p.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows.extend(reader)
out = Path('$Y8M_WORK/det_face_shards')
counts = []
for i in range(6):
    shard = rows[i::6]
    path = out / f'shard_{i:02d}.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shard)
    counts.append(len(shard))
print({'total_face_rows': len(rows), 'six_shards': counts})
PY

  echo "[$(date '+%F %T')] Predict YOLOv8m full-test detections"
  pids=()
  for i in $(seq 0 5); do
    printf -v shard '%02d' "$i"
    echo "[yolov8m] launch visible physical GPU $i shard $shard"
    CUDA_VISIBLE_DEVICES="$i" "$YPY" "$SCRIPT/predict_yolo_test_faces.py" \
      --manifest "$Y8M_WORK/det_face_shards/shard_${shard}.csv" \
      --model "$Y8M_MODEL" --out-csv "$Y8M_WORK/det_shards/shard_${shard}.csv" \
      --imgsz 512 --batch "$BATCH" --device "" --predict-conf 0.001 --predict-iou 0.7 --max-det 20 \
      > "$SUB/logs/yolov8m_det_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  [ "$fail" = 0 ] || { echo "[FAILED] yolov8m detector shard inference"; exit 1; }
else
  echo "[$(date '+%F %T')] Reuse existing YOLOv8m det shards in $Y8M_WORK/det_shards"
fi

echo "[$(date '+%F %T')] Merge WBF full-test boxes"
"$PY" "$SCRIPT/merge_wbf_test_boxes_hetero_v1.py" \
  --old-glob "$OLD_WORK/det_shards/shard_*.csv" \
  --repeat2-glob "$REPEAT2_WORK/det_shards/shard_*.csv" \
  --yolov8m-glob "$Y8M_WORK/det_shards/shard_*.csv" \
  --pre-conf 0.125 --wbf-iou 0.35 --post-conf 0.175 --max-boxes 3 --require-models 2 \
  --out-json "$WBF_WORK/detector_pred_boxes.json" \
  --out-summary "$SUB/logs/wbf_detector_boxes_summary.json" \
  | tee "$SUB/logs/merge_wbf_boxes.log"

echo "[$(date '+%F %T')] Build 4 zip variants with reused text"
"$PY" "$SCRIPT/build_wbf_text_from_variant_zips.py" \
  --normal-text-zip "$NORMAL_TEXT_ZIP" \
  --eyes-text-zip "$EYES_TEXT_ZIP" \
  --eyes-mouth-text-zip "$EYES_MOUTH_TEXT_ZIP" \
  --nose-eyes-mouth-text-zip "$NOSE_EYES_MOUTH_TEXT_ZIP" \
  --image-scores "$SRC/image_scores.csv" \
  --test-images "$SRC/test_images.csv" \
  --detector-boxes "$WBF_WORK/detector_pred_boxes.json" \
  --cls-preds-dir "$SRC/cls_preds" \
  --raw-face-dir "$RAW_FACE" \
  --out-root "$SUB" \
  --tag "$TAG" \
  | tee "$SUB/logs/build_four_variants.log"

echo "[$(date '+%F %T')] [done] WBF full-test 4 variants completed"
