#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  cat >&2 <<'EOF'
Usage:
  bash scripts/rebuild_from_saved_wbf_boxes.sh \
    /path/to/image_scores.json \
    /path/to/test_images.json \
    /path/to/cls_preds_dir \
    /path/to/raw_face_dir \
    /path/to/output_dir

This rebuilds the WBF submission variants from saved WBF boxes and cached
Qwen-generated text source zips. It is for exact artifact reconstruction, not
method-level explanation generation.

Expected inputs:
  evidence/detector_pred_boxes.json
  text_sources/*normal*.zip
  text_sources/*fake_nobox_eyes*.zip
  text_sources/*fake_nobox_eyes_mouth*.zip
  text_sources/*fake_nobox_nose_eyes_mouth*.zip

Override exact text zip names with:
  NORMAL_TEXT_ZIP
  EYES_TEXT_ZIP
  EYES_MOUTH_TEXT_ZIP
  NOSE_EYES_MOUTH_TEXT_ZIP
EOF
  exit 2
fi

IMAGE_SCORES="$1"
TEST_IMAGES="$2"
CLS_PREDS_DIR="$3"
RAW_FACE_DIR="$4"
OUT_DIR="$5"

DETECTOR_BOXES="${DETECTOR_BOXES:-evidence/detector_pred_boxes.json}"
TEXT_SOURCES_DIR="${TEXT_SOURCES_DIR:-text_sources}"
NORMAL_TEXT_ZIP="${NORMAL_TEXT_ZIP:-${TEXT_SOURCES_DIR}/normal.zip}"
EYES_TEXT_ZIP="${EYES_TEXT_ZIP:-${TEXT_SOURCES_DIR}/fake_nobox_eyes.zip}"
EYES_MOUTH_TEXT_ZIP="${EYES_MOUTH_TEXT_ZIP:-${TEXT_SOURCES_DIR}/fake_nobox_eyes_mouth.zip}"
NOSE_EYES_MOUTH_TEXT_ZIP="${NOSE_EYES_MOUTH_TEXT_ZIP:-${TEXT_SOURCES_DIR}/fake_nobox_nose_eyes_mouth.zip}"

if [[ ! -f "${DETECTOR_BOXES}" ]]; then
  echo "Missing saved boxes: ${DETECTOR_BOXES}" >&2
  echo "Place detector_pred_boxes.json under evidence/ or set DETECTOR_BOXES." >&2
  exit 2
fi

for f in "${NORMAL_TEXT_ZIP}" "${EYES_TEXT_ZIP}" "${EYES_MOUTH_TEXT_ZIP}" "${NOSE_EYES_MOUTH_TEXT_ZIP}" "${IMAGE_SCORES}" "${TEST_IMAGES}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing required file: ${f}" >&2
    exit 2
  fi
done

for d in "${CLS_PREDS_DIR}" "${RAW_FACE_DIR}"; do
  if [[ ! -d "${d}" ]]; then
    echo "Missing required directory: ${d}" >&2
    exit 2
  fi
done

mkdir -p "${OUT_DIR}"

python src/ddli_detector_v1/build_wbf_text_from_variant_zips.py \
  --normal-text-zip "${NORMAL_TEXT_ZIP}" \
  --eyes-text-zip "${EYES_TEXT_ZIP}" \
  --eyes-mouth-text-zip "${EYES_MOUTH_TEXT_ZIP}" \
  --nose-eyes-mouth-text-zip "${NOSE_EYES_MOUTH_TEXT_ZIP}" \
  --image-scores "${IMAGE_SCORES}" \
  --test-images "${TEST_IMAGES}" \
  --detector-boxes "${DETECTOR_BOXES}" \
  --cls-preds-dir "${CLS_PREDS_DIR}" \
  --raw-face-dir "${RAW_FACE_DIR}" \
  --out-root "${OUT_DIR}" \
  --tag "wbf_rebuild"

echo "Rebuild output written to ${OUT_DIR}"
