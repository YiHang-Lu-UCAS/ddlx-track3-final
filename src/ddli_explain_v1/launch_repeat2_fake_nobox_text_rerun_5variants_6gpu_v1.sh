#!/usr/bin/env bash
set -euo pipefail

BASE=/home/pengsiran/projects_data/luyihang
EXP=$BASE/experiments/ddlx_test_repeat2_fake_nobox_explain_qwen25vl3b_checkpoint1500_fallback4_6gpu_v3
MODEL=$BASE/qwen25vl_sft_repro/models/Qwen2.5-VL-3B-Instruct
CKPT=$BASE/experiments/ddli_explain_qwen25vl3b_lora_v1/formal_run_v1/v0-20260526-222731/checkpoint-1500
SRC=/media/omnisky/sdb/pengsiran/projects_data/luyihang/datasets/DDL-X/metadata_clsadapt020_adapt384_cleanup_top3_v1
SOURCE_SUB=$BASE/submissions/ddl_x_test_detector_repeat2_lr1e4_conf015_nms025_textsame_v1_5variants_6gpu
OUT_SUB=$BASE/submissions/ddl_x_test_detector_repeat2_lr1e4_conf015_nms025_textrerun_fake_nobox_v3_fallback4_6gpu
SCRIPT=$BASE/ddli_explain_v1
INPUT=$EXP/inputs
OUTPUT=$EXP/raw_predictions
LOG=$EXP/logs
ENV=/home/pengsiran/anaconda3/envs/qwen25vl_sft
TAG=repeat2_lr1e4_conf015_nms025_textrerun_fake_nobox_v3
VARIANTS=(fake_nobox_eyes_mouth fake_nobox_eyes fake_nobox_nose_eyes fake_nobox_nose_eyes_mouth)

test -d "$SOURCE_SUB/normal/json"
test -s "$SRC/test_images.csv"
test -s "$CKPT/adapter_model.safetensors"
test -s "$MODEL/config.json"
test ! -e "$EXP"
test ! -e "$OUT_SUB"
mkdir -p "$OUTPUT" "$LOG"

exec > >(tee -a "$EXP/pipeline.log") 2>&1
echo "[$(date '+%F %T')] repeat2 fake detector-nobox explanation rerun start"

/home/pengsiran/anaconda3/envs/vmamba/bin/python "$SCRIPT/build_fake_nobox_variant_explain_shards.py" \
  --source-root "$SOURCE_SUB" \
  --variants "${VARIANTS[@]}" \
  --test-images "$SRC/test_images.csv" \
  --out-dir "$INPUT" \
  --num-shards 6

source /home/pengsiran/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"

run_shard() {
  local gpu=$1
  local shard
  printf -v shard '%02d' "$gpu"
  local input="$INPUT/shard_${shard}.jsonl"
  local output="$OUTPUT/shard_${shard}_predictions.jsonl"
  local log="$LOG/shard_${shard}.log"
  local status="$LOG/shard_${shard}.status"
  local batch=128
  rm -f "$output" "$log" "$status"
  echo "[$(date '+%F %T')] launch shard=$shard gpu=$gpu batch=$batch" > "$status"
  if ! CUDA_VISIBLE_DEVICES="$gpu" swift infer \
      --model "$MODEL" --adapters "$CKPT" --template qwen2_5_vl \
      --val_dataset "$input" --infer_backend pt --torch_dtype float16 \
      --max_length 4096 --max_pixels 602112 --max_new_tokens 640 \
      --temperature 0 --num_beams 1 --max_batch_size "$batch" --stream false \
      --result_path "$output" > "$log" 2>&1; then
    batch=64
    rm -f "$output"
    echo "[$(date '+%F %T')] retry shard=$shard gpu=$gpu batch=$batch" >> "$status"
    CUDA_VISIBLE_DEVICES="$gpu" swift infer \
      --model "$MODEL" --adapters "$CKPT" --template qwen2_5_vl \
      --val_dataset "$input" --infer_backend pt --torch_dtype float16 \
      --max_length 4096 --max_pixels 602112 --max_new_tokens 640 \
      --temperature 0 --num_beams 1 --max_batch_size "$batch" --stream false \
      --result_path "$output" > "$log" 2>&1
  fi
  local rows
  rows=$(wc -l < "$output")
  echo "[$(date '+%F %T')] done shard=$shard gpu=$gpu batch=$batch rows=$rows" >> "$status"
}

pids=()
for gpu in $(seq 0 5); do
  run_shard "$gpu" &
  pids+=("$!")
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
if [ "$fail" != 0 ]; then
  echo "[$(date '+%F %T')] FAILED: generation shard failed"
  exit 1
fi

/home/pengsiran/anaconda3/envs/vmamba/bin/python "$SCRIPT/apply_fake_nobox_variant_explanations.py" \
  --source-root "$SOURCE_SUB" \
  --input-dir "$INPUT" \
  --prediction-dir "$OUTPUT" \
  --out-root "$OUT_SUB" \
  --tag "$TAG" \
  --variants "${VARIANTS[@]}" \
  --num-shards 6

echo "[$(date '+%F %T')] DONE: fake detector-nobox text rerun packages completed"
