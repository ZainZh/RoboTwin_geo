#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SEED VARIANT" >&2
  exit 2
fi
SEED="$1"
VARIANT="$2"
[[ "${SEED}" == "20260806" ]] || {
  echo "remote stage42 is frozen to training seed 20260806" >&2
  exit 2
}
case "${VARIANT}" in
  full_fixed|no_ce|no_supcon|no_consistency) ;;
  *) echo "unknown stage42 variant: ${VARIANT}" >&2; exit 2 ;;
esac
[[ "${RUN_STAGE42:-0}" == "1" ]] || {
  echo "refusing remote stage42 without explicit RUN_STAGE42=1" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_ROOT="/workspace/RoboTwin_geo/outputs/paper_revision/hammer_stage42_independent_remote6000ada_seed${SEED}_e4000_split424242_20260806"
RUN_NAME="hammer_${VARIANT}_e4000_split424242_seed${SEED}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${OUTPUT_ROOT}/logs/${RUN_NAME}.log"
MANIFEST="${OUTPUT_ROOT}/manifests/${RUN_NAME}.sha256"
OUTER_COMPLETION="${OUTPUT_ROOT}/completed/${RUN_NAME}.complete"

mkdir -p "${OUTPUT_ROOT}/completed" "${OUTPUT_ROOT}/manifests" "${OUTPUT_ROOT}/logs"
if [[ -f "${OUTER_COMPLETION}" ]]; then
  /workspace/venvs/geo-utonia/bin/python \
    "${SCRIPT_DIR}/hammer_stage42_protocol.py" \
    --role remote --seed "${SEED}" --variant "${VARIANT}" \
    --mode audit --require-outer
  echo "[stage42-remote] verified already complete: ${RUN_NAME}"
  exit 0
fi

STAGE42_ROLE=remote \
RUN_STAGE42=1 \
STAGE42_SEEDS="${SEED}" \
STAGE42_VARIANTS="${VARIANT}" \
STAGE42_NUM_WORKERS="${STAGE42_NUM_WORKERS:-8}" \
  /bin/bash "${SCRIPT_DIR}/run_hammer_stage42_independent_matrix.sh" \
  2>&1 | tee "${LOG_PATH}"

# Audit every inner checkpoint and SHA before publishing the outer marker used
# by the remote queue/sync layer.
/workspace/venvs/geo-utonia/bin/python \
  "${SCRIPT_DIR}/hammer_stage42_protocol.py" \
  --role remote --seed "${SEED}" --variant "${VARIANT}" --mode audit

manifest_tmp="${MANIFEST}.tmp-${BASHPID}"
completion_tmp="${OUTER_COMPLETION}.tmp-${BASHPID}"
trap 'rm -f "${manifest_tmp}" "${completion_tmp}"' EXIT
find "${RUN_DIR}" -type f -print0 | sort -z | xargs -0 sha256sum > "${manifest_tmp}"
mv "${manifest_tmp}" "${MANIFEST}"
{
  date -Is
  echo "protocol=hammer_stage42_independent_test_v1"
  echo "seed=${SEED}"
  echo "variant=${VARIANT}"
  echo "split_seed=424242"
  echo "val_ratio=0.1"
  echo "test_ratio=0.2"
  sha256sum "${MANIFEST}"
} > "${completion_tmp}"
mv "${completion_tmp}" "${OUTER_COMPLETION}"
trap - EXIT

/workspace/venvs/geo-utonia/bin/python \
  "${SCRIPT_DIR}/hammer_stage42_protocol.py" \
  --role remote --seed "${SEED}" --variant "${VARIANT}" \
  --mode audit --require-outer
echo "[stage42-remote] verified complete: ${RUN_NAME}"
