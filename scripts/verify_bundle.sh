#!/usr/bin/env bash
set -euo pipefail

EXPECTED_SHA256="a00d0f7e81d0742c03842eb45a8b010498b5bd502bf9c17d25620cdf89f11e97"

FINAL_ZIP="${1:-${FINAL_ZIP:-final_artifact/submission_fake_nobox_nose_eyes_mouth.zip}}"

if [[ ! -f "${FINAL_ZIP}" ]]; then
  cat >&2 <<EOF
Final zip not found: ${FINAL_ZIP}

Usage:
  bash scripts/verify_bundle.sh /path/to/submission_fake_nobox_nose_eyes_mouth.zip

or set:
  FINAL_ZIP=/path/to/submission_fake_nobox_nose_eyes_mouth.zip

Expected SHA256:
  ${EXPECTED_SHA256}
EOF
  exit 2
fi

ACTUAL_SHA256="$(sha256sum "${FINAL_ZIP}" | awk '{print $1}')"

echo "Expected: ${EXPECTED_SHA256}"
echo "Actual:   ${ACTUAL_SHA256}"

if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  echo "SHA256 mismatch" >&2
  exit 1
fi

echo "OK: final zip matches the selected artifact."
