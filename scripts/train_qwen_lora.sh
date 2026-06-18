#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/train_qwen_lora.sh configs/qwen_lora_sft.env

The config file must define MODEL_NAME_OR_PATH, TRAIN_DATA, VAL_DATA, and
OUTPUT_DIR. Environment variables set by the caller override config values.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/qwen_lora_sft.env}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  usage
  exit 2
fi
CONFIG_PATH="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)/$(basename "${CONFIG_PATH}")"

OVERRIDE_KEYS=(
  CUDA_VISIBLE_DEVICES NPROC_PER_NODE MASTER_PORT MODEL_NAME_OR_PATH TRAIN_DATA
  VAL_DATA OUTPUT_DIR TUNER_TYPE LORA_RANK LORA_ALPHA LORA_DROPOUT
  TARGET_MODULES NUM_TRAIN_EPOCHS PER_DEVICE_TRAIN_BATCH_SIZE
  PER_DEVICE_EVAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS LEARNING_RATE
  WARMUP_RATIO WEIGHT_DECAY LR_SCHEDULER_TYPE SAVE_STEPS EVAL_STEPS
  LOGGING_STEPS MAX_LENGTH MAX_PIXELS DATALOADER_NUM_WORKERS DATASET_NUM_PROC
  BF16 FP16 TORCH_DTYPE GRADIENT_CHECKPOINTING LAZY_TOKENIZE
  SAVE_TOTAL_LIMIT REPORT_TO EVAL_STRATEGY SAVE_STRATEGY RESUME_FROM_CHECKPOINT
)
declare -A ENV_OVERRIDES=()
for key in "${OVERRIDE_KEYS[@]}"; do
  if [[ "${!key+x}" ]]; then
    ENV_OVERRIDES["${key}"]="${!key}"
  fi
done

set -a
# shellcheck disable=SC1090
source "${CONFIG_PATH}"
set +a
for key in "${!ENV_OVERRIDES[@]}"; do
  printf -v "${key}" '%s' "${ENV_OVERRIDES[$key]}"
done
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-29636}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH.}"
TRAIN_DATA="${TRAIN_DATA:?Set TRAIN_DATA.}"
VAL_DATA="${VAL_DATA:?Set VAL_DATA.}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR.}"

for path in "${MODEL_NAME_OR_PATH}" "${TRAIN_DATA}" "${VAL_DATA}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required training input not found: ${path}" >&2
    exit 1
  fi
done
if ! command -v swift >/dev/null 2>&1; then
  echo "swift CLI not found; activate environment-qwen.yml first." >&2
  exit 1
fi

python -m src.ddli_explain_v1.validate_swift_jsonl --jsonl "${TRAIN_DATA}" --check-images
python -m src.ddli_explain_v1.validate_swift_jsonl --jsonl "${VAL_DATA}" --check-images
mkdir -p "${OUTPUT_DIR}"

CMD=(
  swift sft
  --model "${MODEL_NAME_OR_PATH}"
  --dataset "${TRAIN_DATA}"
  --val_dataset "${VAL_DATA}"
  --output_dir "${OUTPUT_DIR}"
  --tuner_type "${TUNER_TYPE:-lora}"
  --lora_rank "${LORA_RANK:-16}"
  --lora_alpha "${LORA_ALPHA:-32}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --target_modules "${TARGET_MODULES:-all-linear}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
  --learning_rate "${LEARNING_RATE:-5e-5}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}"
  --warmup_ratio "${WARMUP_RATIO:-0.03}"
  --weight_decay "${WEIGHT_DECAY:-0.1}"
  --save_strategy "${SAVE_STRATEGY:-steps}"
  --save_steps "${SAVE_STEPS:-250}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-4}"
  --eval_strategy "${EVAL_STRATEGY:-steps}"
  --eval_steps "${EVAL_STEPS:-250}"
  --logging_steps "${LOGGING_STEPS:-10}"
  --max_length "${MAX_LENGTH:-4096}"
  --max_pixels "${MAX_PIXELS:-602112}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}"
  --bf16 "${BF16:-false}"
  --fp16 "${FP16:-true}"
  --torch_dtype "${TORCH_DTYPE:-float16}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}"
  --dataset_num_proc "${DATASET_NUM_PROC:-2}"
  --lazy_tokenize "${LAZY_TOKENIZE:-true}"
  --report_to "${REPORT_TO:-none}"
)
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

echo "Running Qwen2.5-VL LoRA SFT with global batch size $((NPROC_PER_NODE * ${PER_DEVICE_TRAIN_BATCH_SIZE:-1} * ${GRADIENT_ACCUMULATION_STEPS:-4}))"
printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
