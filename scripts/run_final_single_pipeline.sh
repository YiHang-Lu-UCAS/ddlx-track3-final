#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-verify}"

cat_usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/run_final_single_pipeline.sh verify [FINAL_ZIP]
  bash scripts/run_final_single_pipeline.sh rebuild IMAGE_SCORES TEST_IMAGES CLS_PREDS_DIR RAW_FACE_DIR OUT_DIR
  bash scripts/run_final_single_pipeline.sh qwen EXPLAIN_INPUTS_JSONL OUT_DIR
  bash scripts/run_final_single_pipeline.sh all [FINAL_ZIP]

Purpose:
  Organizer-facing entrypoint for the single submitted DDL-X Track 3 system.
  The final system is one fixed inference pipeline with classification,
  localization, and explanation branches that jointly produce the final JSON.

Modes:
  verify   Check the exact selected leaderboard artifact SHA256.
  rebuild  Rebuild final WBF variants from saved WBF boxes and cached
           Qwen-generated text sources. This is exact artifact reconstruction.
  qwen     Rerun the Qwen2.5-VL + LoRA explanation branch on prepared JSONL
           inputs. This is method-level explanation verification.
  all      Run verify, then optionally run rebuild/qwen when the required
           environment variables are set.

Environment variables for all:
  IMAGE_SCORES
  TEST_IMAGES
  CLS_PREDS_DIR
  RAW_FACE_DIR
  REBUILD_OUT_DIR
  EXPLAIN_INPUTS_JSONL
  QWEN_OUT_DIR
EOF
}

case "${MODE}" in
  verify)
    FINAL_ZIP="${2:-${FINAL_ZIP:-}}"
    if [[ -n "${FINAL_ZIP}" ]]; then
      bash scripts/verify_bundle.sh "${FINAL_ZIP}"
    else
      bash scripts/verify_bundle.sh
    fi
    ;;

  rebuild)
    if [[ $# -lt 6 ]]; then
      cat_usage
      exit 2
    fi
    bash scripts/rebuild_from_saved_wbf_boxes.sh "$2" "$3" "$4" "$5" "$6"
    ;;

  qwen)
    if [[ $# -lt 3 ]]; then
      cat_usage
      exit 2
    fi
    bash scripts/run_explanation_inference.sh "$2" "$3"
    ;;

  all)
    FINAL_ZIP="${2:-${FINAL_ZIP:-}}"
    if [[ -n "${FINAL_ZIP}" ]]; then
      bash scripts/verify_bundle.sh "${FINAL_ZIP}"
    else
      bash scripts/verify_bundle.sh
    fi

    if [[ -n "${IMAGE_SCORES:-}" && -n "${TEST_IMAGES:-}" && -n "${CLS_PREDS_DIR:-}" && -n "${RAW_FACE_DIR:-}" && -n "${REBUILD_OUT_DIR:-}" ]]; then
      bash scripts/rebuild_from_saved_wbf_boxes.sh "${IMAGE_SCORES}" "${TEST_IMAGES}" "${CLS_PREDS_DIR}" "${RAW_FACE_DIR}" "${REBUILD_OUT_DIR}"
    else
      echo "Skipping rebuild: set IMAGE_SCORES, TEST_IMAGES, CLS_PREDS_DIR, RAW_FACE_DIR, and REBUILD_OUT_DIR to enable it."
    fi

    if [[ -n "${EXPLAIN_INPUTS_JSONL:-}" && -n "${QWEN_OUT_DIR:-}" ]]; then
      bash scripts/run_explanation_inference.sh "${EXPLAIN_INPUTS_JSONL}" "${QWEN_OUT_DIR}"
    else
      echo "Skipping Qwen rerun: set EXPLAIN_INPUTS_JSONL and QWEN_OUT_DIR to enable it."
    fi
    ;;

  -h|--help|help)
    cat_usage
    ;;

  *)
    echo "Unknown mode: ${MODE}" >&2
    cat_usage
    exit 2
    ;;
esac
