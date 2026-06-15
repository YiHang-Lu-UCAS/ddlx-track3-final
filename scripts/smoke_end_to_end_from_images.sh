#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'EOF'
Usage:
  bash scripts/smoke_end_to_end_from_images.sh /path/to/test/images /path/to/model_package/models /tmp/ddlx_smoke

Runs the end-to-end pipeline with --skip-qwen on the provided image directory.
Use a small directory, for example 8-16 images, for a quick verifier smoke test.
EOF
  exit 2
fi

IMAGE_DIR="$1"
MODEL_ROOT="$2"
OUT_DIR="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

bash scripts/run_end_to_end_from_images.sh \
  --image-dir "${IMAGE_DIR}" \
  --model-root "${MODEL_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --gpus auto \
  --skip-qwen \
  --force
