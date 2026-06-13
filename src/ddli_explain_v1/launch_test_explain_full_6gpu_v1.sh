#!/usr/bin/env bash
set -euo pipefail

BASE=/home/pengsiran/projects_data/luyihang
EXP=$BASE/experiments/ddlx_test_explain_qwen25vl3b_checkpoint1500_full_6gpu_v1
MODEL=$BASE/qwen25vl_sft_repro/models/Qwen2.5-VL-3B-Instruct
CKPT=$BASE/experiments/ddli_explain_qwen25vl3b_lora_v1/formal_run_v1/v0-20260526-222731/checkpoint-1500
INPUT=$EXP/inputs
OUTPUT=$EXP/raw_predictions
LOG=$EXP/logs
ENV=/home/pengsiran/anaconda3/envs/qwen25vl_sft

mkdir -p "$OUTPUT" "$LOG"
test -s "$INPUT/input_summary.json"
test -s "$CKPT/adapter_model.safetensors"
test -s "$MODEL/config.json"

source /home/pengsiran/anaconda3/etc/profile.d/conda.sh
conda activate qwen25vl_sft

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

echo "[$(date '+%F %T')] full DDL-X explanation generation start; GPUs=0-5 max_new_tokens=640 batch=128 fallback=64"
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
  echo "[$(date '+%F %T')] FAILED: at least one generation shard failed"
  exit 1
fi
echo "[$(date '+%F %T')] DONE: all six raw text generation shards completed; no conclusion repair applied"
