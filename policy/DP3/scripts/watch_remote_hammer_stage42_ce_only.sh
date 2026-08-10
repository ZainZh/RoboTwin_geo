#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-root@112.69.3.12}"
REMOTE_PORT="${REMOTE_PORT:-43678}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REMOTE_BASE="/workspace/RoboTwin_geo/outputs/paper_revision"
ROOT_NAME="hammer_stage42_independent_remote6000ada_seed20260806_e4000_split424242_20260806"
RUN_NAME="hammer_ce_only_e4000_split424242_seed20260806"
EVAL_NAME="hammer_stage42_independent_test_evaluation_e1750_v1"
LABEL="seed20260806_ce_only"
REMOTE_ROOT="${REMOTE_BASE}/${ROOT_NAME}"
LOCAL_ROOT="${REPO_ROOT}/outputs/paper_revision/${ROOT_NAME}"
REMOTE_EVAL="${REMOTE_BASE}/${EVAL_NAME}"
LOCAL_EVAL="${REPO_ROOT}/outputs/paper_revision/${EVAL_NAME}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=12 -p "${REMOTE_PORT}")
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=12 -p ${REMOTE_PORT}"

if ! [[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

remote_file_exists() {
  local path="$1"
  "${SSH[@]}" "${REMOTE_HOST}" test -f "${path}"
}

wait_for_remote_file() {
  local path="$1"
  while ! remote_file_exists "${path}"; do
    sleep "${POLL_SECONDS}"
  done
}

verify_tree() {
  local remote_path="$1"
  local local_path="$2"
  local differences
  differences="$(rsync -ani --checksum -e "${RSYNC_SSH}" \
    "${REMOTE_HOST}:${remote_path}/" "${local_path}/" | grep -E '^[<>ch.*]' || true)"
  if [[ -n "${differences}" ]]; then
    echo "checksum verification failed for ${remote_path}: ${differences}" >&2
    return 1
  fi
}

OUTER_COMPLETION="${REMOTE_ROOT}/completed/${RUN_NAME}.complete"
wait_for_remote_file "${OUTER_COMPLETION}"
mkdir -p "${LOCAL_ROOT}/completed" "${LOCAL_ROOT}/manifests" "${LOCAL_ROOT}/logs"
rsync -a --checksum --partial -e "${RSYNC_SSH}" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/${RUN_NAME}/" "${LOCAL_ROOT}/${RUN_NAME}/"
rsync -a --checksum -e "${RSYNC_SSH}" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/completed/${RUN_NAME}.complete" "${LOCAL_ROOT}/completed/"
rsync -a --checksum -e "${RSYNC_SSH}" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/manifests/${RUN_NAME}.sha256" "${LOCAL_ROOT}/manifests/"
rsync -a --checksum -e "${RSYNC_SSH}" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/logs/${RUN_NAME}.log" "${LOCAL_ROOT}/logs/"
verify_tree "${REMOTE_ROOT}/${RUN_NAME}" "${LOCAL_ROOT}/${RUN_NAME}"
echo "[stage42-sync] checksum-verified training run: ${RUN_NAME}"

R1_RESULT="${REMOTE_EVAL}/${LABEL}/r1_test_r5_fp32/results.json"
R2_RESULT="${REMOTE_EVAL}/${LABEL}/r2_test_t5_fp32/results.json"
wait_for_remote_file "${R1_RESULT}"
wait_for_remote_file "${R2_RESULT}"
mkdir -p "${LOCAL_EVAL}/${LABEL}"
rsync -a --checksum --partial -e "${RSYNC_SSH}" \
  "${REMOTE_HOST}:${REMOTE_EVAL}/${LABEL}/" "${LOCAL_EVAL}/${LABEL}/"
verify_tree "${REMOTE_EVAL}/${LABEL}" "${LOCAL_EVAL}/${LABEL}"
echo "[stage42-sync] checksum-verified evaluation: ${LABEL}"
