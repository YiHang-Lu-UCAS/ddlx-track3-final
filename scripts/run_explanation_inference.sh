#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'EOF'
Usage:
  bash scripts/run_explanation_inference.sh /path/to/explain_inputs.jsonl /path/to/output_dir

This is the method-level explanation reproduction path. It invokes Qwen2.5-VL
with the LoRA checkpoint-1500 adapter. The generated text may not be byte-identical
to the cached final artifact text.

Environment overrides:
  QWEN_BASE_MODEL
  QWEN_LORA_ADAPTER
  QWEN_MAX_NEW_TOKENS
  CUDA_VISIBLE_DEVICES
EOF
  exit 2
fi

INPUT_JSONL="$1"
OUT_DIR="$2"

QWEN_BASE_MODEL="${QWEN_BASE_MODEL:-models/explanation/qwen2_5_vl_3b_instruct}"
QWEN_LORA_ADAPTER="${QWEN_LORA_ADAPTER:-models/explanation/qwen2_5_vl_3b_lora_checkpoint1500}"
QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-2048}"

if [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "Missing input JSONL: ${INPUT_JSONL}" >&2
  exit 2
fi

if [[ ! -d "${QWEN_BASE_MODEL}" ]]; then
  echo "Missing Qwen base model directory: ${QWEN_BASE_MODEL}" >&2
  exit 2
fi

if [[ ! -d "${QWEN_LORA_ADAPTER}" ]]; then
  echo "Missing Qwen LoRA adapter directory: ${QWEN_LORA_ADAPTER}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

swift infer \
  --model "${QWEN_BASE_MODEL}" \
  --adapters "${QWEN_LORA_ADAPTER}" \
  --template qwen2_5_vl \
  --val_dataset "${INPUT_JSONL}" \
  --result_path "${OUT_DIR}/qwen_explanations.jsonl" \
  --max_new_tokens "${QWEN_MAX_NEW_TOKENS}" \
  --max_length 4096 \
  --max_pixels 602112 \
  --temperature 0 \
  --num_beams 1

echo "Qwen explanation output written to ${OUT_DIR}/qwen_explanations.jsonl"
