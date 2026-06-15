#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/run_end_to_end_from_images.sh \
    --image-dir /path/to/test/images \
    --model-root /path/to/model_package/models \
    --out-dir /path/to/output \
    [--gpus auto|cpu|0|0,1,2] \
    [--swift-command swift] \
    [--num-explain-shards 1] \
    [--skip-qwen] \
    [--force]

This is the single public inference entrypoint for the DDL-X Track 3 model
package. It scans input images and regenerates JSON outputs containing:
classification, bounding boxes, and visible-forgery explanations.
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python -m src.ddlx_full_infer_v1.run_end_to_end "$@"
